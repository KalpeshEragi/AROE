import pytest

from lrm.cli import build, parse_args


def scheduler_for(*argv):
    return build(parse_args(["--events", "", *argv])).state.scheduler


def test_defaults_are_aging_with_preemption():
    s = scheduler_for()
    assert s.policy.__name__ == "priority_aging" and s.policy.promote_after_s == 15.0
    assert s.preempt is True and s.max_preemptions == 2
    assert build(parse_args(["--events", ""])).state.embed_chunk == 32


def test_baseline_flags():
    s = scheduler_for("--policy", "fcfs", "--no-preempt", "--model-limit", "bge-m3=2")
    assert s.policy.__name__ == "fcfs" and s.preempt is False
    assert s.limit("bge-m3:latest") == 2 and s.limit("qwen3.5:4b") == 1


def test_tuning_flags():
    s = scheduler_for("--policy", "aging", "--promote-after", "5", "--max-preemptions", "0")
    assert s.policy.promote_after_s == 5.0 and s.max_preemptions == 0
    assert build(parse_args(["--events", "", "--embed-chunk", "0"])).state.embed_chunk == 0


def test_bad_policy_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--policy", "lottery"])
