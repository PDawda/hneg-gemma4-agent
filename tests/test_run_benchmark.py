"""Resume / offset / summary logic of the harness, with a fake runner and episodes."""
import csv
import importlib.util
import json
import sys

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "experiments" / "run_benchmark.py")
rb = importlib.util.module_from_spec(spec)
sys.modules["run_benchmark"] = rb
spec.loader.exec_module(rb)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"instance_id": f"b{i}", "repo_path": str(repo), "target_file": "x.py",
                                           "issue": "", "test_cmd": "true"}) for i in range(4)))
    calls = []

    def fake_episode(task, config, runner, cfg, args):
        calls.append((task["instance_id"], config))
        if task["instance_id"] == "b2" and config == "full":
            raise ValueError("prompt too long")
        r = rb.BugResult(task["instance_id"], config, resolved=(config == "full"), turns=2,
                         total_tokens=100, final_patch_tokens=40 if config == "full" else 0, latency_s=1.0)
        return r

    monkeypatch.setattr(rb, "run_episode", fake_episode)
    monkeypatch.setattr(rb, "make_runner", lambda args, cfg: object())
    meta = {"model": "m.gguf", "model_sha256": "aaa"}
    monkeypatch.setattr(rb, "collect_meta", lambda args, cfg, runner, tasks: dict(meta))
    out = tmp_path / "out"

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["rb", "--tasks", str(tasks), "--out-dir", str(out),
                                          "--configs", "baseline", "full", "--backend", "gguf", *extra])
        try:
            rb.main()
        except SystemExit as e:
            return e.code
        return 0

    run.meta = meta
    return run, calls, out


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_resume_offset_and_summary(setup):
    run, calls, out = setup
    assert run("--limit", "2") == 0
    assert calls == [("b0", "baseline"), ("b0", "full"), ("b1", "baseline"), ("b1", "full")]

    calls.clear()
    assert run() == 0                                   # resumes: only b2, b3 are new
    assert [c[0] for c in calls] == ["b2", "b2", "b3", "b3"]
    errors = [json.loads(l) for l in (out / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["instance_id"] == "b2"

    calls.clear()
    assert run() == 0                                   # nothing left (b2 errored, others done)
    assert calls == []

    summary = {r["config_key"]: r for r in read(out / "summary.csv")}
    assert summary["full"]["n_bugs"] == "3"              # b2 excluded: not done by every config
    assert summary["full"]["pass_at_1_pct"] == "100.0" and summary["baseline"]["pass_at_1_pct"] == "0.0"
    assert summary["full"]["mean_token_waste"] == "0.6"

    calls.clear()
    assert run("--fresh", "--offset", "3") == 0
    assert calls == [("b3", "baseline"), ("b3", "full")]


def test_time_budget_stops_cleanly(setup):
    run, calls, out = setup
    assert run("--time-budget-hours", "0") == 3
    assert calls == []


def test_refuses_to_resume_with_a_different_model(setup):
    run, calls, out = setup
    assert run("--limit", "1") == 0
    assert json.loads((out / "run_meta.json").read_text())["model_sha256"] == "aaa"
    run.meta["model_sha256"] = "bbb"
    assert "Refusing to resume" in str(run("--limit", "2"))
    assert run("--limit", "2", "--fresh") == 0


class _FakeRunner:
    """Always proposes the same patch; the repo's tests always fail."""

    def __init__(self):
        from conftest import FakeTokenizer
        self.tok = FakeTokenizer(["a", "b"])

    def generate(self, messages, processor=None):
        if processor is not None:
            processor.reset(0)
        return "```python\ndef f():\n    return 1\n```", [2, 3, 2], 50, 0.01


def _episode(tmp_path, config):
    repo = tmp_path / "r"
    repo.mkdir(exist_ok=True)
    (repo / "x.py").write_text("def f():\n    return 0\n")
    task = {"instance_id": "t", "repo_path": str(repo), "target_file": "x.py", "issue": "", "test_cmd": "false"}
    cfg = rb.load_config(ROOT / "hneg_config.yaml")

    class A:
        max_turns, test_timeout, trace_dir = 10, 30, None
    return rb.run_episode(task, config, _FakeRunner(), cfg, A())


def test_baseline_k3_uses_retry_cap_and_loop_metric_is_uniform(tmp_path):
    base = _episode(tmp_path, "baseline")
    k3 = _episode(tmp_path, "baseline_k3")
    full = _episode(tmp_path, "full")
    assert (base.turns, k3.turns, full.turns) == (10, 3, 3)
    assert k3.hit_turn_cap and not k3.halted
    assert full.halted and not full.hit_turn_cap
    # identical patch every turn -> counted as a loop for every config, including full
    assert base.infinite_loop and k3.infinite_loop and full.infinite_loop
    assert (base.repeated_patches, k3.repeated_patches) == (9, 2)   # same patch every turn after the first
    assert k3.total_tokens < base.total_tokens


class _OverflowRunner(_FakeRunner):
    def __init__(self, overflow_at):
        super().__init__()
        self.calls, self.overflow_at = 0, overflow_at

    def generate(self, messages, processor=None):
        self.calls += 1
        if self.calls >= self.overflow_at:
            raise rb.ContextOverflow("too long")
        return super().generate(messages, processor)


def test_context_overflow_counts_as_failure_after_turn_one(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "x.py").write_text("x = 1\n")
    task = {"instance_id": "t", "repo_path": str(repo), "target_file": "x.py", "issue": "", "test_cmd": "false"}
    cfg = rb.load_config(ROOT / "hneg_config.yaml")

    class A:
        max_turns, test_timeout, trace_dir = 10, 30, None
    r = rb.run_episode(task, "baseline", _OverflowRunner(overflow_at=3), cfg, A())
    assert r.context_overflow and not r.resolved and r.turns == 2 and r.total_tokens > 0
    with pytest.raises(rb.ContextOverflow):                     # does not fit at all -> excluded
        rb.run_episode(task, "baseline", _OverflowRunner(overflow_at=1), cfg, A())


class _EditRunner(_FakeRunner):
    """First reply is a malformed edit, second a correct SEARCH/REPLACE fix."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def generate(self, messages, processor=None):
        if processor is not None:
            processor.reset(0)
        self.n += 1
        assert "<<<<<<< SEARCH" in messages[0]["content"]            # edit instructions in the prompt
        if self.n == 1:
            return "```python\n<<<<<<< SEARCH\nreturn 99\n=======\nreturn 1\n>>>>>>> REPLACE\n```", [5, 6], 40, 0.01
        return "```python\n<<<<<<< SEARCH\n    return 0\n=======\n    return 1\n>>>>>>> REPLACE\n```", [7, 8], 40, 0.01


def test_edit_mode_episode_applies_patch_and_passes_tests(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "x.py").write_text("def f():\n    return 0\n")
    cmd = f"{sys.executable} -c 'import x; assert x.f() == 1'"
    task = {"instance_id": "t", "repo_path": str(repo), "target_file": "x.py", "issue": "f should return 1",
            "test_cmd": cmd}
    cfg = rb.load_config(ROOT / "hneg_config.yaml")

    class A:
        max_turns, test_timeout, trace_dir, patch_format, context_chars = 10, 60, None, "edit", 24000
    for config in ("baseline", "full"):
        r = rb.run_episode(task, config, _EditRunner(), cfg, A())
        assert r.resolved and r.turns == 2, config
        assert r.attempts[0].apply_error and "not found" in r.attempts[0].apply_error
        assert r.parse_failures == 1 and r.attempts[1].test_exit == 0


class _RecordingEditRunner(_EditRunner):
    def __init__(self):
        super().__init__()
        self.seen, self.procs = [], []

    def generate(self, messages, processor=None):
        self.seen.append(list(messages))
        self.procs.append(processor)
        return super().generate(messages, processor)


def test_full_edit_mode_uses_replay_guard_and_shared_feedback(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "x.py").write_text("def f():\n    return 0\n")
    cmd = f"{sys.executable} -c 'import x; assert x.f() == 1'"
    task = {"instance_id": "t", "repo_path": str(repo), "target_file": "x.py", "issue": "f should return 1",
            "test_cmd": cmd}
    cfg = rb.load_config(ROOT / "hneg_config.yaml")

    class A:
        max_turns, test_timeout, trace_dir, patch_format, context_chars = 10, 60, None, "edit", 24000
    run = _RecordingEditRunner()
    r = rb.run_episode(task, "full", run, cfg, A())
    proc = run.procs[0]
    assert proc.failed_trie is None and proc.replay_guard is not None     # edit mode: guard, not token trie
    assert "return 99" in proc.replay_guard.failed_search                  # unfound SEARCH is excluded
    retry = run.seen[1][-1]["content"]
    assert "could not be applied" in retry and "return 0" in retry          # closest-lines hint
    assert r.resolved


def test_retry_instruction_is_identical_for_every_config(tmp_path):
    seen = {}
    for config in ("baseline", "baseline_k3", "full"):
        run = _RecordingEditRunner()
        repo = tmp_path / config
        repo.mkdir()
        (repo / "x.py").write_text("def f():\n    return 0\n")
        task = {"instance_id": "t", "repo_path": str(repo), "target_file": "x.py", "issue": "f should return 1",
                "test_cmd": f"{sys.executable} -c 'import x; assert x.f() == 1'"}

        class A:
            max_turns, test_timeout, trace_dir, patch_format, context_chars = 10, 60, None, "edit", 24000
        rb.run_episode(task, config, run, rb.load_config(ROOT / "hneg_config.yaml"), A())
        seen[config] = run.seen[1][-1]["content"]
    assert len(set(seen.values())) == 1 and seen["full"].endswith(rb.RETRY_SUFFIX)
