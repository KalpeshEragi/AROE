"""Policy tests: pure functions first, then a deterministic run showing starvation vs aging."""
import pytest

from lrm.core.job import Job, Priority
from lrm.core.policies import by_name, effective_level, fcfs, make_aging, strict_priority

I, N, B = Priority.INTERACTIVE, Priority.NORMAL, Priority.BACKGROUND


def job(prio, t, model="m"):
    j = Job(kind="chat", model=model, priority=prio, arrival_t=t)
    j.enqueue_t = t
    return j


def anyone(_):
    return True


def test_fcfs_ignores_priority():
    q = [job(B, 0), job(I, 1)]
    assert fcfs(q, anyone, 5) is q[0]


def test_strict_priority_prefers_class_then_age():
    old_bg, new_i, old_i = job(B, 0), job(I, 2), job(I, 1)
    assert strict_priority([old_bg, new_i, old_i], anyone, 10) is old_i


def test_policies_skip_jobs_whose_model_is_busy():
    busy, free = job(I, 0, model="qwen"), job(B, 1, model="bge-m3")
    only_free = lambda j: j.model == "bge-m3"
    for policy in (fcfs, strict_priority, make_aging(15)):
        assert policy([busy, free], only_free, 2) is free


@pytest.mark.parametrize("waited, level", [(0, 2), (14.9, 2), (15, 1), (29.9, 1), (30, 0), (500, 0)])
def test_effective_level_rises_with_wait(waited, level):
    assert effective_level(job(B, 0), waited, 15) == level


def test_aged_background_beats_newer_interactive_but_not_older():
    aging = make_aging(15)
    bg = job(B, 0)
    older_i, newer_i = job(I, -1), job(I, 20)
    assert aging([bg, newer_i], anyone, 31) is bg          # bg at level 0, older than newer_i
    assert aging([bg, newer_i], anyone, 25) is newer_i     # bg only at level 1 yet
    assert aging([bg, older_i], anyone, 31) is older_i     # equal level, older_i waited longer


def test_by_name():
    assert by_name("fcfs") is fcfs and by_name("priority") is strict_priority
    assert by_name("aging", 7).promote_after_s == 7
    with pytest.raises(ValueError):
        by_name("lottery")
    with pytest.raises(ValueError):
        make_aging(0)


def simulate(policy, horizon=600.0, service=4.0, interactive_every=3.0):
    """One model, one slot, every job takes `service` s. Interactive requests arrive every
    `interactive_every` s, which is more than the slot can serve (overload). One background
    job is queued at t=0. Returns when the background job started, or None."""
    bg = job(B, 0.0)
    queue, t, next_i, busy_until = [bg], 0.0, 0.0, 0.0
    while t <= horizon:
        while next_i <= t:
            queue.append(job(I, next_i))
            next_i += interactive_every
        if busy_until <= t and queue:
            chosen = policy(queue, anyone, t)
            queue.remove(chosen)
            if chosen is bg:
                return t
            busy_until = t + service
        t = min(next_i, max(busy_until, t + 1e-9))
    return None


def test_strict_priority_starves_background_under_overload():
    assert simulate(strict_priority) is None


def test_aging_bounds_background_wait_under_overload():
    started = simulate(make_aging(15))
    # Reaches level 0 at 30 s; then only interactive jobs that arrived before it (none,
    # it arrived at t=0) and the job in service can delay it: bound = 30 + one service.
    assert started is not None and started <= 30 + 4.0, started
