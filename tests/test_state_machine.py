import json

import pytest

from src.state_machine import (
    DynamicStateManager,
    Event,
    Phase,
    TransitionError,
    apply_calibration,
    calibrate_from_traces,
    load_config,
)
from conftest import ROOT


@pytest.fixture
def cfg():
    return load_config(ROOT / "hneg_config.yaml")


def test_happy_path(cfg):
    sm = DynamicStateManager.from_config(cfg)
    sm.start("a.py")
    sm.fire(Event.TARGET_LOCATED, ast_node="f")
    assert sm.current.phase == Phase.CODE_EDIT and sm.current.ast_node == "f"
    sm.fire(Event.PATCH_EMITTED)
    sm.fire(Event.BUILD_PASSED)
    sm.fire(Event.TESTS_PASSED)
    assert sm.done and not sm.halted and sm.k == 0
    with pytest.raises(TransitionError):
        sm.fire(Event.PATCH_EMITTED)


def test_push_pop_restores_parent_scope(cfg):
    sm = DynamicStateManager.from_config(cfg)
    sm.start("a.py")
    sm.fire(Event.TARGET_LOCATED, ast_node="f")
    sm.fire(Event.MISSING_DEP, file_scope="b.py", ast_node="g")
    assert sm.current.phase == Phase.RESOLVE_DEP and sm.current.depth == 1 and len(sm.stack) == 2
    assert sm.tool_allowed("resolve_external_symbol") and not sm.tool_allowed("emit_structured_patch")
    sm.fire(Event.DEP_RESOLVED)
    assert sm.current.phase == Phase.CODE_EDIT
    assert (sm.current.file_scope, sm.current.ast_node, sm.current.depth) == ("a.py", "f", 0)


def test_graceful_halt_at_k_max(cfg):
    sm = DynamicStateManager.from_config(cfg)
    sm.start("a.py")
    for attempt in range(1, 4):
        sm.fire(Event.TARGET_LOCATED)
        sm.fire(Event.PATCH_EMITTED)
        sm.fire(Event.BUILD_FAILED, payload={"error": "boom"})
        assert sm.k == attempt
    assert sm.halted and sm.current.phase == Phase.GRACEFUL_HALT


def test_on_failure_callback(cfg):
    seen = []
    sm = DynamicStateManager.from_config(cfg, on_failure=lambda m, p: seen.append(p["tokens"]))
    sm.start()
    sm.fire(Event.TARGET_LOCATED)
    sm.fire(Event.PATCH_EMITTED)
    sm.fire(Event.BUILD_FAILED, payload={"tokens": [1, 2]})
    assert seen == [[1, 2]]


def test_strict_mode_rejects_illegal_transition(cfg):
    sm = DynamicStateManager.from_config(cfg)
    sm.start()
    with pytest.raises(TransitionError):
        sm.fire(Event.BUILD_PASSED)
    lax = DynamicStateManager(strict_mode=False)
    lax.start()
    assert lax.fire(Event.BUILD_PASSED).phase == Phase.REPO_ANALYSIS
    assert lax.tool_allowed("unknown_tool")


def test_tool_gating(cfg):
    sm = DynamicStateManager.from_config(cfg)
    sm.start()
    assert sm.tool_allowed("query_cpg_graph")
    with pytest.raises(TransitionError):
        sm.require_tool("run_test_suite")
    assert not sm.tool_allowed("not_in_matrix")   # strict mode denies unknown tools


def test_stack_depth_bound_halts():
    sm = DynamicStateManager(max_depth=1)
    sm.start()
    sm.fire(Event.TARGET_LOCATED)
    sm.fire(Event.MISSING_DEP)
    assert sm.halted


def test_trace_and_calibration(tmp_path, cfg):
    trace = tmp_path / "t.jsonl"
    # Episode 1: success on attempt 2. Episode 2: halt. Episode 3: success on attempt 1.
    for script in (["fail", "pass"], ["fail", "fail", "fail"], ["pass"]):
        sm = DynamicStateManager.from_config(cfg)
        sm.start("a.py")
        for step in script:
            sm.fire(Event.TARGET_LOCATED)
            sm.fire(Event.PATCH_EMITTED)
            if step == "pass":
                sm.fire(Event.BUILD_PASSED)
                sm.fire(Event.TESTS_PASSED)
            else:
                sm.fire(Event.BUILD_FAILED, payload={"error": "AssertionError: x"})
        sm.dump_trace(trace)

    cal = calibrate_from_traces(tmp_path, coverage=0.5)
    assert cal["n_episodes"] == 3
    assert cal["attempts_to_success_hist"] == {1: 1, 2: 1}
    assert cal["halt_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert cal["recommended"]["max_convergence_retries"] == 1
    assert cal["top_failure_messages"]["AssertionError: x"] == 4
    assert calibrate_from_traces(trace, coverage=1.0)["recommended"]["max_convergence_retries"] == 2

    new_cfg = apply_calibration(cfg, cal)
    assert new_cfg["governance"]["max_convergence_retries"] == 1
    assert cfg["governance"]["max_convergence_retries"] == 3     # original untouched
    json.dumps(cal)
