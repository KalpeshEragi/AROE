"""HTTP gateway: OpenAI-compatible routes in front of the scheduler and the Ollama adapter.

Request flow: parse -> create Job -> wait for a slot (scheduler) -> call backend ->
stream or return -> release the slot. Every step is an event in the EventLog.

Client disconnects are checked while queued, while the backend call is pending, and
during streaming, so an abandoned request never keeps a slot or keeps Ollama busy
(Ollama stops generating when its connection closes; measured in Phase 1, E5).
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..adapters.ollama import BackendError, ChatStream
from ..core.events import EventLog
from ..core.job import Job, JobState, Priority
from ..core.scheduler import Scheduler
from . import openai_compat as oa

CLIENT_CLOSED = 499  # nginx convention: client closed the request before a response


class ManagedStream(StreamingResponse):
    """StreamingResponse that always runs `cleanup`, however the stream ends.

    If the client disconnects before the body generator even starts, the generator's own
    `finally` never runs, so cleanup cannot live there. And during cancellation, anyio keeps
    cancelling every await in the cancelled scope (level-triggered), so cleanup must be
    shielded or `await upstream.aclose()` would itself be cancelled.
    """

    def __init__(self, content, cleanup: Callable[[], Awaitable[None]], **kw):
        super().__init__(content, **kw)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._cleanup()


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def collect_chat(adapter, body: dict[str, Any], progress: dict[str, int]) -> dict[str, Any]:
    """Non-streaming chat, done by streaming from the backend and assembling the answer.

    Why not Ollama's non-streaming mode: cancelling a *streaming* request is verified to
    stop generation (Phase 1, E5), and streaming lets us count tokens as they arrive
    (`progress["tokens"]`), which is the work lost if the request is preempted.
    Returns a dict shaped like Ollama's non-streaming /api/chat response.
    """
    stream = await adapter.open_chat_stream(body)
    content: list[str] = []
    thinking: list[str] = []
    final = None
    try:
        async for line in stream.lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise BackendError(500, chunk["error"])
            msg = chunk.get("message") or {}
            for key, sink in (("content", content), ("thinking", thinking)):
                if msg.get(key):
                    sink.append(msg[key])
                    progress["tokens"] = progress.get("tokens", 0) + 1
            if chunk.get("done"):
                final = chunk
                break
    finally:
        with anyio.CancelScope(shield=True):
            await stream.aclose()
    if final is None:
        raise BackendError(502, "backend stream ended without done", "backend_connection_lost")
    message = {"role": "assistant", "content": "".join(content)}
    if thinking:
        message["thinking"] = "".join(thinking)
    return {**final, "message": message}


def create_app(adapter, scheduler: Scheduler, events: EventLog,
               *, disconnect_poll_s: float = 0.25, embed_chunk: int = 32) -> FastAPI:
    """`embed_chunk`: max inputs per backend embedding call (0 = never split)."""

    @asynccontextmanager
    async def lifespan(app):
        events.emit("gateway.started")
        yield
        await adapter.aclose()
        events.emit("gateway.stopped")
        events.close()

    app = FastAPI(title="Local AI Resource Manager", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------ helpers
    async def until_done_or_disconnect(request: Request, aw: Awaitable[Any]) -> tuple[bool, Any]:
        """Await `aw`, but give up (cancelling it) if the client disconnects first.
        Returns (finished, result). Exceptions from `aw` propagate."""
        task = asyncio.ensure_future(aw)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=disconnect_poll_s)
                if done:
                    return True, task.result()
                if await request.is_disconnected():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    return False, None
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def wait_for_slot(request: Request, job: Job, keep_seniority: bool = False) -> bool:
        got, _ = await until_done_or_disconnect(request, scheduler.acquire(job, keep_seniority))
        if not got and job.state is JobState.RUNNING:
            # The slot was granted in the same instant the disconnect was noticed.
            scheduler.release(job, JobState.CANCELLED, stage="dispatched_then_disconnected")
        return got

    def job_headers(job: Job) -> dict[str, str]:
        return {"X-LRM-Job-Id": job.id, "X-LRM-Queue-Wait-Ms": f"{(job.wait_s or 0) * 1000:.1f}"}

    def backend_error_response(e: BackendError, job: Job | None) -> JSONResponse:
        status = 502 if e.status >= 500 else e.status
        return JSONResponse(oa.error_body(e.message, e.kind, str(e.status)), status_code=status,
                            headers=job_headers(job) if job else None)

    async def parse(request: Request, translate) -> tuple[dict, dict, Priority] | JSONResponse:
        try:
            req = await request.json()
        except ValueError:
            return JSONResponse(oa.error_body("request body must be JSON"), status_code=400)
        if not isinstance(req, dict):
            return JSONResponse(oa.error_body("request body must be a JSON object"), status_code=400)
        try:
            prio = Priority.parse(request.headers.get("x-lrm-priority"))
            body = translate(req)
        except ValueError as e:  # BadRequest is a ValueError
            return JSONResponse(oa.error_body(str(e)), status_code=400)
        return req, body, prio

    async def run_simple(request: Request, job: Job, call: Awaitable[dict]) -> dict | Response:
        """Non-streaming backend call holding the job's slot. Returns the backend JSON
        or an error Response; always releases the slot."""
        try:
            finished, result = await until_done_or_disconnect(request, call)
        except BackendError as e:
            scheduler.release(job, JobState.FAILED, error=e.message, backend_status=e.status)
            return backend_error_response(e, job)
        except asyncio.CancelledError:
            scheduler.release(job, JobState.CANCELLED, stage="handler_cancelled")
            raise
        except Exception as e:
            scheduler.release(job, JobState.FAILED, error=f"gateway error: {e!r}")
            raise
        if not finished:
            scheduler.release(job, JobState.CANCELLED, stage="client_disconnected")
            return Response(status_code=CLIENT_CLOSED)
        return result

    async def run_preemptible_chat(request: Request, job: Job, body: dict) -> dict | Response | None:
        """Like run_simple for a non-streaming chat, but the scheduler may preempt it.
        Returns None if preempted (slot already given up; caller re-acquires and retries)."""
        progress = {"tokens": 0}
        task = asyncio.ensure_future(collect_chat(adapter, body, progress))
        started = time.monotonic()
        if job.preemptible:
            job.on_preempt = task.cancel
        try:
            finished, result = await until_done_or_disconnect(request, task)
        except asyncio.CancelledError:
            me = asyncio.current_task()
            if job.preempt_requested and not (me and me.cancelling()):
                # The backend call was cancelled by preemption, not this handler.
                scheduler.requeue(job, "preempted", wasted_tokens=progress["tokens"],
                                  wasted_s=round(time.monotonic() - started, 3),
                                  preemptions=job.preemptions)
                return None
            scheduler.release(job, JobState.CANCELLED, stage="handler_cancelled")
            raise
        except BackendError as e:
            scheduler.release(job, JobState.FAILED, error=e.message, backend_status=e.status)
            return backend_error_response(e, job)
        except Exception as e:
            scheduler.release(job, JobState.FAILED, error=f"gateway error: {e!r}")
            raise
        finally:
            job.on_preempt = None
        if not finished:
            scheduler.release(job, JobState.CANCELLED, stage="client_disconnected")
            return Response(status_code=CLIENT_CLOSED)
        return result

    # ------------------------------------------------------------ routes
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        parsed = await parse(request, oa.chat_request_to_ollama)
        if isinstance(parsed, Response):
            return parsed
        req, body, prio = parsed
        job = scheduler.new_job("chat", body["model"], prio)
        # Only background, non-streaming chat can be preempted: its client sees nothing until
        # the end, so a rerun is invisible apart from the delay. Streamed tokens can't be taken back.
        job.preemptible = not req.get("stream") and prio is Priority.BACKGROUND
        if not await wait_for_slot(request, job):
            return Response(status_code=CLIENT_CLOSED)
        cid, created, model = f"chatcmpl-{job.id}", int(time.time()), body["model"]

        if not req.get("stream"):
            while True:
                result = await run_preemptible_chat(request, job, body)
                if result is not None:
                    break
                if not await wait_for_slot(request, job, keep_seniority=True):
                    return Response(status_code=CLIENT_CLOSED)
            if isinstance(result, Response):
                return result
            scheduler.mark_first_token(job)  # non-streaming: first byte = whole answer
            scheduler.release(job, JobState.COMPLETED, preemptions=job.preemptions, **oa.timings_from(result))
            return JSONResponse(oa.chat_response(result, cid, created, model), headers=job_headers(job))

        # Streaming. Open the upstream stream first (this is where a model load happens and
        # where load failures surface), so errors can still become a proper HTTP status.
        result = await run_simple(request, job, adapter.open_chat_stream(body))
        if isinstance(result, Response):
            return result
        upstream: ChatStream = result
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        end = {"outcome": JobState.CANCELLED, "info": {"stage": "client_disconnected"}}

        async def body_iter():
            first = True
            try:
                async for line in upstream.lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        end.update(outcome=JobState.FAILED, info={"error": chunk["error"]})
                        yield _sse(oa.error_body(chunk["error"], "backend_error"))
                        yield "data: [DONE]\n\n"
                        return
                    msg = chunk.get("message") or {}
                    if msg.get("content") or msg.get("thinking"):
                        scheduler.mark_first_token(job)
                    out = oa.stream_chunk(chunk, cid, created, model, first)
                    if out is not None:
                        first = False
                        yield _sse(out)
                    if chunk.get("done"):
                        end.update(outcome=JobState.COMPLETED, info=oa.timings_from(chunk))
                        if include_usage:
                            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                                        "model": model, "choices": [], "usage": oa.usage_from(chunk)})
                        yield "data: [DONE]\n\n"
                        return
                end.update(outcome=JobState.FAILED, info={"error": "backend stream ended without done"})
            except BackendError as e:
                end.update(outcome=JobState.FAILED, info={"error": e.message})
                yield _sse(oa.error_body("backend connection lost", "backend_error"))

        async def cleanup():
            await upstream.aclose()
            scheduler.release(job, end["outcome"], **end["info"])

        return ManagedStream(body_iter(), cleanup, media_type="text/event-stream",
                             headers={**job_headers(job), "Cache-Control": "no-cache"})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        parsed = await parse(request, oa.embed_request_to_ollama)
        if isinstance(parsed, Response):
            return parsed
        _req, body, prio = parsed
        job = scheduler.new_job("embed", body["model"], prio)
        # Large inputs run as chunks. Between chunks the job gives up its slot and queues
        # again with a *fresh* enqueue time, so a long indexing job never ages up to the
        # interactive level; an interactive query waits at most one chunk.
        inputs = body["input"]
        size = embed_chunk if embed_chunk and len(inputs) > embed_chunk else len(inputs)
        parts = [inputs[i:i + size] for i in range(0, len(inputs), size)]
        vectors: list = []
        prompt_tokens = 0
        for i, part in enumerate(parts):
            if len(parts) > 1:
                job.label = f"chunk {i + 1}/{len(parts)}"
            if i > 0:
                scheduler.requeue(job, "yielded", chunk=f"{i}/{len(parts)}")
            if not await wait_for_slot(request, job):
                return Response(status_code=CLIENT_CLOSED)
            result = await run_simple(request, job, adapter.embed({**body, "input": part}))
            if isinstance(result, Response):
                return result
            got = result.get("embeddings") or []
            if len(got) != len(part):
                e = BackendError(502, f"backend returned {len(got)} embeddings for {len(part)} inputs")
                scheduler.release(job, JobState.FAILED, error=e.message)
                return backend_error_response(e, job)
            vectors.extend(got)
            prompt_tokens += result.get("prompt_eval_count") or 0
        scheduler.release(job, JobState.COMPLETED, n_inputs=len(inputs), chunks=len(parts),
                          prompt_eval_count=prompt_tokens)
        return JSONResponse(oa.embed_response({"embeddings": vectors, "prompt_eval_count": prompt_tokens},
                                              body["model"]), headers=job_headers(job))

    @app.get("/v1/models")
    async def models():
        try:
            ms = await adapter.list_models()
        except BackendError as e:
            return backend_error_response(e, None)
        return {"object": "list", "data": [{"id": m["name"], "object": "model", "created": 0,
                                            "owned_by": "ollama"} for m in ms]}

    @app.get("/lrm/status")
    async def status():
        return scheduler.status()

    @app.get("/lrm/events")
    async def recent_events(n: int = 200, job: str | None = None):
        return events.recent(n=n, job_id=job)

    return app
