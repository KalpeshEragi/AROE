import asyncio
import random

import pytest

from lrm.core.events import EventLog
from lrm.core.job import JobState, Priority
from lrm.core.scheduler import Scheduler

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"  # the scheduler uses asyncio futures


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(limit=1, limits=None):
    clock = FakeClock()
    ev = EventLog(now=clock)
    return Scheduler(ev, default_limit=limit, model_limits=limits, now=clock), ev, clock


def dispatched_order(ev):
    return [e["job"] for e in ev.recent(10_000) if e["type"] == "job.dispatched"]


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_same_model_runs_in_arrival_order():
    s, ev, _ = make()
    jobs = [s.new_job("chat", "qwen", Priority.NORMAL) for _ in range(3)]
    tasks = [asyncio.create_task(s.acquire(j)) for j in jobs]
    await settle()
    assert [j.state for j in jobs] == [JobState.RUNNING, JobState.QUEUED, JobState.QUEUED]
    s.release(jobs[0], JobState.COMPLETED)
    await settle()
    s.release(jobs[1], JobState.COMPLETED)
    await settle()
    assert dispatched_order(ev) == [j.id for j in jobs]
    await asyncio.gather(*tasks)


async def test_busy_model_does_not_block_other_models():
    s, _, _ = make()
    a = s.new_job("chat", "qwen", Priority.NORMAL)
    b = s.new_job("embed", "bge-m3", Priority.NORMAL)
    await s.acquire(a)
    await asyncio.wait_for(s.acquire(b), timeout=1)  # would hang if b waited behind a
    assert a.state is b.state is JobState.RUNNING


async def test_name_without_tag_shares_slot_with_latest():
    s, _, _ = make()
    a = s.new_job("embed", "bge-m3", Priority.NORMAL)
    b = s.new_job("embed", "bge-m3:latest", Priority.NORMAL)
    await s.acquire(a)
    t = asyncio.create_task(s.acquire(b))
    await settle()
    assert b.state is JobState.QUEUED
    s.release(a, JobState.COMPLETED)
    await asyncio.wait_for(t, 1)


async def test_cancel_while_queued_leaves_queue_and_next_job_runs():
    s, ev, _ = make()
    a, b, c = (s.new_job("chat", "qwen", Priority.NORMAL) for _ in range(3))
    await s.acquire(a)
    tb = asyncio.create_task(s.acquire(b))
    tc = asyncio.create_task(s.acquire(c))
    await settle()
    tb.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tb
    assert b.state is JobState.CANCELLED
    assert [q["id"] for q in s.status()["queued"]] == [c.id]
    s.release(a, JobState.COMPLETED)
    await asyncio.wait_for(tc, 1)
    assert c.state is JobState.RUNNING
    assert any(e["type"] == "job.cancelled" and e["stage"] == "queued" for e in ev.recent())


async def test_cancel_in_same_instant_as_grant_does_not_leak_slot():
    s, _, _ = make()
    a, b, c = (s.new_job("chat", "qwen", Priority.NORMAL) for _ in range(3))
    await s.acquire(a)
    tb = asyncio.create_task(s.acquire(b))
    await settle()
    s.release(a, JobState.COMPLETED)   # grants b's slot (future resolved)...
    tb.cancel()                        # ...but b is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await tb
    assert b.state is JobState.CANCELLED
    assert s.status()["running"] == []
    await asyncio.wait_for(s.acquire(c), 1)  # slot is free again


async def test_release_is_idempotent():
    s, ev, _ = make()
    a = s.new_job("chat", "qwen", Priority.NORMAL)
    await s.acquire(a)
    s.release(a, JobState.COMPLETED)
    s.release(a, JobState.FAILED)
    assert a.state is JobState.COMPLETED
    assert sum(e["type"].startswith("job.completed") or e["type"] == "job.failed" for e in ev.recent()) == 1


async def test_wait_time_and_reason_recorded():
    s, ev, clock = make()
    a, b = (s.new_job("chat", "qwen", Priority.NORMAL) for _ in range(2))
    await s.acquire(a)
    tb = asyncio.create_task(s.acquire(b))
    await settle()
    clock.t = 3.5
    s.release(a, JobState.COMPLETED)
    await tb
    d = [e for e in ev.recent() if e["type"] == "job.dispatched" and e["job"] == b.id][0]
    assert d["wait_s"] == pytest.approx(3.5)
    assert "waited 3.50s" in d["reason"] and "qwen:latest" in d["reason"]


async def test_aging_policy_in_scheduler_lets_interactive_jump_then_ages_background():
    from lrm.core.policies import make_aging
    clock = FakeClock()
    ev = EventLog(now=clock)
    s = Scheduler(ev, policy=make_aging(10), now=clock)
    running = s.new_job("chat", "qwen", Priority.NORMAL)
    await s.acquire(running)
    bg = s.new_job("chat", "qwen", Priority.BACKGROUND)
    t_bg = asyncio.create_task(s.acquire(bg))
    await settle()
    clock.t = 1.0
    inter = s.new_job("chat", "qwen", Priority.INTERACTIVE)
    t_i = asyncio.create_task(s.acquire(inter))
    await settle()
    s.release(running, JobState.COMPLETED)   # t=1: bg is level 2, interactive level 0
    await settle()
    assert inter.state is JobState.RUNNING and bg.state is JobState.QUEUED
    clock.t = 25.0                            # bg waited 25 s -> level 0
    late = s.new_job("chat", "qwen", Priority.INTERACTIVE)
    t_late = asyncio.create_task(s.acquire(late))
    await settle()
    s.release(inter, JobState.COMPLETED)
    await settle()
    assert bg.state is JobState.RUNNING and late.state is JobState.QUEUED
    assert s.status()["policy"] == "priority_aging"
    s.release(bg, JobState.COMPLETED)
    await asyncio.gather(t_bg, t_i, t_late)


@pytest.mark.parametrize("seed", range(30))
async def test_random_workload_invariants(seed):
    """Random arrivals, cancels and completions. Checks after every step:
    per-model limit never exceeded; each job dispatched at most once; among jobs of one
    model that were dispatched, dispatch order == arrival order (FCFS per model)."""
    rng = random.Random(seed)
    limits = {"m0:latest": 1, "m1:latest": 2}
    s, ev, clock = make(limit=1, limits=limits)
    tasks, jobs, running = {}, [], []
    for step in range(120):
        clock.t += rng.random()
        r = rng.random()
        if r < 0.45:
            j = s.new_job("chat", f"m{rng.randrange(3)}", Priority.NORMAL)
            jobs.append(j)
            tasks[j.id] = asyncio.create_task(s.acquire(j))
        elif r < 0.6:
            waiting = [j for j in jobs if j.state is JobState.QUEUED]
            if waiting:
                tasks[rng.choice(waiting).id].cancel()
        else:
            running = [j for j in jobs if j.state is JobState.RUNNING]
            if running:
                s.release(rng.choice(running), JobState.COMPLETED)
        await settle()
        per_model = {}
        for j in jobs:
            if j.state is JobState.RUNNING:
                per_model[j.key] = per_model.get(j.key, 0) + 1
        for k, n in per_model.items():
            assert n <= s.limit(k), (seed, step, k, n)
    order = dispatched_order(ev)
    assert len(order) == len(set(order))
    by_id = {j.id: j for j in jobs}
    for key in {j.key for j in jobs}:
        d = [by_id[i] for i in order if by_id[i].key == key]
        assert d == sorted(d, key=lambda j: j.arrival_t), (seed, key)
    for t in tasks.values():
        t.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
