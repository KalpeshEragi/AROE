"""Gateway tests against a fake Ollama running as a real HTTP server (own thread).

Real sockets on both sides, so disconnect handling is tested end to end:
client -> gateway -> aiohttp -> fake backend. No GPU or Ollama needed.
"""
import asyncio
import json
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from lrm.adapters.ollama import OllamaAdapter
from lrm.api.app import create_app
from lrm.core.events import EventLog
from lrm.core.scheduler import Scheduler

CHAT = {"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------- fake Ollama
class FakeOllama:
    def __init__(self):
        self.reset()

    def reset(self, n_tokens=3, delay=0.0, fail_status=None, stream_error=False):
        self.n, self.delay, self.fail_status, self.stream_error = n_tokens, delay, fail_status, stream_error
        self.closed_early = threading.Event()  # backend saw the gateway hang up mid-stream
        self.finished = threading.Event()      # stream ran to the end
        self.requests = []

    def app(self):
        return Starlette(routes=[Route("/api/chat", self.chat, methods=["POST"]),
                                 Route("/api/embed", self.embed, methods=["POST"]),
                                 Route("/api/tags", self.tags)])

    def _fail(self):
        return JSONResponse({"error": "llama-server process has terminated: CUDA error: "
                                      "shared object initialization failed"}, status_code=self.fail_status)

    async def chat(self, request: Request):
        body = await request.json()
        self.requests.append(("/api/chat", body))
        if self.fail_status:
            return self._fail()
        done = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                "prompt_eval_count": 4, "eval_count": self.n, "load_duration": 1_000_000,
                "eval_duration": 2_000_000}
        if not body.get("stream"):
            return JSONResponse({**done, "message": {"role": "assistant", "content": "ok"}})
        fake = self

        async def gen():
            completed = False
            try:
                for i in range(fake.n):
                    if fake.delay:
                        await asyncio.sleep(fake.delay)
                    yield json.dumps({"message": {"role": "assistant", "content": f"t{i} "}, "done": False}) + "\n"
                yield json.dumps({"error": "runner crashed"} if fake.stream_error else done) + "\n"
                completed = True
            finally:
                (fake.finished if completed else fake.closed_early).set()
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    async def embed(self, request: Request):
        body = await request.json()
        self.requests.append(("/api/embed", body))
        if self.fail_status:
            return self._fail()
        return JSONResponse({"embeddings": [[0.5, 0.25]] * len(body["input"]), "prompt_eval_count": 7})

    async def tags(self, request: Request):
        return JSONResponse({"models": [{"name": "qwen3.5:4b"}, {"name": "bge-m3:latest"}]})


@contextmanager
def serve(app, lifespan="off"):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           lifespan=lifespan))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        th.join(10)


@pytest.fixture(scope="module")
def backend():
    fake = FakeOllama()
    with serve(fake.app()) as url:
        yield fake, url


def build(url, **sched_kw):
    events = EventLog()
    sched = Scheduler(events, **sched_kw)
    adapter = OllamaAdapter(url)
    return create_app(adapter, sched, events, disconnect_poll_s=0.05), sched, events, adapter


def types(events, job=None):
    return [e["type"] for e in events.recent(1000, job_id=job)]


def sse_payloads(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


# ---------------------------------------------------------------- in-process gateway tests
@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def gateway(url, **sched_kw):
    app, sched, ev, adapter = build(url, **sched_kw)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
            yield c, sched, ev
    finally:
        await adapter.aclose()


@pytest.mark.anyio
async def test_chat_non_streaming(backend):
    fake, url = backend
    fake.reset()
    async with gateway(url) as (c, sched, ev):
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 16, "temperature": 0},
                         headers={"X-LRM-Priority": "interactive"})
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "t0 t1 t2 "
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 3
    job = r.headers["X-LRM-Job-Id"]
    assert float(r.headers["X-LRM-Queue-Wait-Ms"]) >= 0
    assert fake.requests[0][1]["options"] == {"num_predict": 16, "temperature": 0}
    assert fake.requests[0][1]["stream"] is True  # always streamed upstream, assembled here
    assert types(ev, job) == ["job.arrived", "job.dispatched", "job.first_token", "job.completed"]
    assert ev.recent(job_id=job)[0]["priority"] == "interactive"
    done = ev.recent(job_id=job)[-1]
    assert done["eval_count"] == 3 and done["load_s"] == 0.001
    assert sched.status()["running"] == []


@pytest.mark.anyio
async def test_chat_streaming_openai_format(backend):
    fake, url = backend
    fake.reset(n_tokens=3)
    async with gateway(url) as (c, sched, ev):
        r = await c.post("/v1/chat/completions",
                         json={**CHAT, "stream": True, "stream_options": {"include_usage": True}})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    payloads = sse_payloads(r.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"])
    assert text == "t0 t1 t2 "
    assert [ch for ch in chunks if ch["choices"]][-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    assert types(ev)[-1] == "job.completed"
    assert fake.finished.is_set()
    assert sched.status()["running"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_backend_load_failure_becomes_502_and_releases_slot(backend, stream):
    fake, url = backend
    fake.reset(fail_status=500)
    async with gateway(url) as (c, sched, ev):
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": stream})
    assert r.status_code == 502
    assert "CUDA error" in r.json()["error"]["message"]
    failed = [e for e in ev.recent() if e["type"] == "job.failed"]
    assert len(failed) == 1 and failed[0]["backend_status"] == 500
    assert sched.status()["running"] == []


@pytest.mark.anyio
async def test_backend_unreachable_is_502(backend):
    with socket.socket() as s:  # find a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
    async with gateway(dead) as (c, sched, ev):
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "backend_unreachable"
    assert types(ev)[-1] == "job.failed"
    assert sched.status()["running"] == []


@pytest.mark.anyio
async def test_error_inside_stream_is_forwarded_and_job_failed(backend):
    fake, url = backend
    fake.reset(n_tokens=2, stream_error=True)
    async with gateway(url) as (c, sched, ev):
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    payloads = sse_payloads(r.text)
    assert json.loads(payloads[-2])["error"]["message"] == "runner crashed"
    assert payloads[-1] == "[DONE]"
    assert types(ev)[-1] == "job.failed"
    assert sched.status()["running"] == []


@pytest.mark.anyio
async def test_error_inside_stream_fails_non_streaming_request(backend):
    fake, url = backend
    fake.reset(n_tokens=2, stream_error=True)
    async with gateway(url) as (c, sched, ev):
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "runner crashed"
    assert types(ev)[-1] == "job.failed"
    assert sched.status()["running"] == []


async def wait_running(sched, n=1, timeout=3.0):
    deadline = time.time() + timeout
    while len(sched.status()["running"]) < n:
        assert time.time() < deadline, "job never started"
        await asyncio.sleep(0.01)


@pytest.mark.anyio
async def test_interactive_preempts_background_non_streaming_chat(backend):
    fake, url = backend
    fake.reset(n_tokens=60, delay=0.02)  # ~1.2 s per generation
    async with gateway(url, preempt=True) as (c, sched, ev):
        bg_task = asyncio.create_task(c.post("/v1/chat/completions", json=CHAT,
                                             headers={"X-LRM-Priority": "background"}))
        await wait_running(sched)
        await asyncio.sleep(0.2)  # let the background job generate some tokens
        r_i = await c.post("/v1/chat/completions", json={**CHAT, "stream": True},
                           headers={"X-LRM-Priority": "interactive"})
        r_bg = await bg_task
    assert r_i.status_code == 200 and r_bg.status_code == 200
    # Background still gets its complete answer, from the rerun.
    assert r_bg.json()["choices"][0]["message"]["content"] == "".join(f"t{i} " for i in range(60))
    preempted = [e for e in ev.recent() if e["type"] == "job.preempted"]
    assert len(preempted) == 1 and preempted[0]["wasted_tokens"] > 0
    done_bg = [e for e in ev.recent() if e["type"] == "job.completed" and e["job"] == r_bg.headers["X-LRM-Job-Id"]]
    assert done_bg[0]["preemptions"] == 1
    assert len([p for p, _ in fake.requests if p == "/api/chat"]) == 3  # bg, interactive, bg rerun
    # Interactive waited for a cancel, not for the ~1 s of background generation left.
    wait_ms = float(r_i.headers["X-LRM-Queue-Wait-Ms"])
    assert wait_ms < 300, wait_ms
    assert sched.status()["running"] == [] and sched.status()["queued"] == []
    print(f"interactive queue wait {wait_ms:.0f} ms; background preempted after "
          f"{preempted[0]['wasted_tokens']} tokens")


@pytest.mark.anyio
async def test_background_streaming_chat_is_not_preempted(backend):
    fake, url = backend
    fake.reset(n_tokens=30, delay=0.02)
    async with gateway(url, preempt=True) as (c, sched, ev):
        bg_task = asyncio.create_task(c.post("/v1/chat/completions", json={**CHAT, "stream": True},
                                             headers={"X-LRM-Priority": "background"}))
        await wait_running(sched)
        r_i = await c.post("/v1/chat/completions", json=CHAT, headers={"X-LRM-Priority": "interactive"})
        r_bg = await bg_task
    assert r_i.status_code == r_bg.status_code == 200
    assert not [e for e in ev.recent() if e["type"] in ("job.preempt_requested", "job.preempted")]


@pytest.mark.anyio
async def test_bad_requests_are_400_without_creating_jobs(backend):
    fake, url = backend
    fake.reset()
    async with gateway(url) as (c, _, ev):
        r1 = await c.post("/v1/chat/completions", json={**CHAT, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
        r2 = await c.post("/v1/chat/completions", json=CHAT, headers={"X-LRM-Priority": "urgent"})
        r3 = await c.post("/v1/chat/completions", content=b"not json")
    assert [r1.status_code, r2.status_code, r3.status_code] == [400, 400, 400]
    assert "unknown priority" in r2.json()["error"]["message"]
    assert not [t for t in types(ev) if t.startswith("job.")]
    assert fake.requests == []


@pytest.mark.anyio
async def test_embeddings_and_models(backend):
    fake, url = backend
    fake.reset()
    async with gateway(url) as (c, _, ev):
        r = await c.post("/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b"]},
                         headers={"X-LRM-Priority": "background"})
        m = await c.get("/v1/models")
    assert r.status_code == 200
    assert [d["index"] for d in r.json()["data"]] == [0, 1]
    assert r.json()["usage"]["prompt_tokens"] == 7
    assert ev.recent(job_id=r.headers["X-LRM-Job-Id"])[-1]["n_inputs"] == 2
    assert [d["id"] for d in m.json()["data"]] == ["qwen3.5:4b", "bge-m3:latest"]


# ---------------------------------------------------------------- real gateway server: disconnects
def wait_until(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_disconnect_mid_stream_closes_backend_and_frees_slot(backend):
    fake, url = backend
    fake.reset(n_tokens=200, delay=0.05)  # ~10 s of tokens if not cancelled
    app, _, _, _ = build(url)
    with serve(app, lifespan="on") as gw, httpx.Client(base_url=gw, timeout=5) as c:
        with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as r:
            got = 0
            for line in r.iter_lines():
                if line.startswith("data: "):
                    got += 1
                    if got == 2:
                        break  # leaving the block closes the connection mid-stream
        t0 = time.time()
        assert fake.closed_early.wait(3), "backend never saw the disconnect"
        closed_after = time.time() - t0
        assert not fake.finished.is_set()
        assert wait_until(lambda: c.get("/lrm/status").json()["running"] == [])
        evs = c.get("/lrm/events").json()
        cancelled = [e for e in evs if e["type"] == "job.cancelled"]
        assert len(cancelled) == 1 and cancelled[0]["stage"] == "client_disconnected"
    print(f"backend saw disconnect {closed_after * 1000:.0f} ms after the client left")


def test_disconnect_while_queued_leaves_queue(backend):
    fake, url = backend
    fake.reset(n_tokens=200, delay=0.05)
    app, _, _, _ = build(url)
    stop = threading.Event()

    def hold_slot(gw):
        with httpx.Client(base_url=gw, timeout=10) as c:
            with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as r:
                for _ in r.iter_lines():
                    if stop.is_set():
                        break

    with serve(app, lifespan="on") as gw, httpx.Client(base_url=gw, timeout=5) as c:
        holder = threading.Thread(target=hold_slot, args=(gw,), daemon=True)
        holder.start()
        assert wait_until(lambda: len(c.get("/lrm/status").json()["running"]) == 1)
        # A second request for the same model has to queue; give up after 0.5 s.
        with httpx.Client(base_url=gw, timeout=httpx.Timeout(5, read=0.5)) as c2:
            with pytest.raises(httpx.ReadTimeout):
                c2.post("/v1/chat/completions", json={**CHAT, "stream": True})
        assert wait_until(lambda: c.get("/lrm/status").json()["queued"] == [])
        evs = c.get("/lrm/events").json()
        assert any(e["type"] == "job.cancelled" and e["stage"] == "queued" for e in evs)
        assert len(c.get("/lrm/status").json()["running"]) == 1  # first request unaffected
        assert len([p for p, _ in fake.requests if p == "/api/chat"]) == 1  # queued one never reached backend
        stop.set()
        holder.join(5)
        assert wait_until(lambda: c.get("/lrm/status").json()["running"] == [])
