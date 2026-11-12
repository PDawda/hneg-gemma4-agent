"""Macro-governance layer: dynamic pushdown statechart.

States are attributed execution frames S_t = <Phase, File_Scope, AST_Node, Depth>
kept on a bounded stack. Sub-tasks (e.g. resolving a missing dependency while
editing) push a child frame; on resolution the child is popped and the parent
frame resumes with its original scope.

The transition table mirrors Section 4 of the README. A global guard sends the
machine to GRACEFUL_HALT once the convergence counter k reaches K_max.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class Phase(str, enum.Enum):
    REPO_ANALYSIS = "REPO_ANALYSIS"
    CODE_EDIT = "CODE_EDIT"
    RESOLVE_DEP = "RESOLVE_DEP"
    LOCAL_COMPILATION = "LOCAL_COMPILATION"
    SYMBOLIC_VALIDATION = "SYMBOLIC_VALIDATION"
    DONE = "DONE"
    GRACEFUL_HALT = "GRACEFUL_HALT"


class Event(str, enum.Enum):
    TARGET_LOCATED = "TARGET_LOCATED"
    MISSING_DEP = "MISSING_DEP"
    DEP_RESOLVED = "DEP_RESOLVED"
    PATCH_EMITTED = "PATCH_EMITTED"
    BUILD_FAILED = "BUILD_FAILED"
    BUILD_PASSED = "BUILD_PASSED"
    TESTS_PASSED = "TESTS_PASSED"
    TESTS_FAILED = "TESTS_FAILED"


class TransitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionFrame:
    """S_t = <Phi, F, Sigma_AST, delta>."""

    phase: Phase
    file_scope: Optional[str] = None
    ast_node: Optional[str] = None
    depth: int = 0
    symbol_allowlist: Tuple[str, ...] = ()

    def with_phase(self, phase: Phase, **changes: Any) -> "ExecutionFrame":
        data = {**asdict(self), "phase": phase, **changes}
        return ExecutionFrame(**data)


# (source phase, event) -> (target phase, stack op)
#   stack op: "replace" swaps the top frame, "push" adds a child, "pop" resumes parent
TRANSITIONS: Dict[Tuple[Phase, Event], Tuple[Phase, str]] = {
    (Phase.REPO_ANALYSIS, Event.TARGET_LOCATED): (Phase.CODE_EDIT, "replace"),
    (Phase.CODE_EDIT, Event.MISSING_DEP): (Phase.RESOLVE_DEP, "push"),
    (Phase.RESOLVE_DEP, Event.DEP_RESOLVED): (Phase.CODE_EDIT, "pop"),
    (Phase.CODE_EDIT, Event.PATCH_EMITTED): (Phase.LOCAL_COMPILATION, "replace"),
    (Phase.LOCAL_COMPILATION, Event.BUILD_FAILED): (Phase.REPO_ANALYSIS, "replace"),
    (Phase.LOCAL_COMPILATION, Event.BUILD_PASSED): (Phase.SYMBOLIC_VALIDATION, "replace"),
    (Phase.SYMBOLIC_VALIDATION, Event.TESTS_PASSED): (Phase.DONE, "replace"),
    (Phase.SYMBOLIC_VALIDATION, Event.TESTS_FAILED): (Phase.REPO_ANALYSIS, "replace"),
}

# Events that count as a failed convergence attempt (k <- k + 1).
FAILURE_EVENTS = {Event.BUILD_FAILED, Event.TESTS_FAILED}
TERMINAL = {Phase.DONE, Phase.GRACEFUL_HALT}


def load_config(path: Union[str, Path] = "hneg_config.yaml") -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml is required to load hneg_config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class TraceEvent:
    t: float
    event: str
    source: str
    target: str
    depth: int
    k: int
    detail: Dict[str, Any] = field(default_factory=dict)


class DynamicStateManager:
    """Hierarchical pushdown automaton driving the agent's macro phases.

    Example
    -------
    >>> sm = DynamicStateManager(max_retries=3, max_depth=4)
    >>> sm.start("src/auth/jwt.py")
    >>> sm.fire(Event.TARGET_LOCATED, ast_node="verify_token")
    >>> sm.current.phase
    <Phase.CODE_EDIT: 'CODE_EDIT'>
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_depth: int = 4,
        strict_mode: bool = True,
        enable_graceful_halt: bool = True,
        tool_state_matrix: Optional[List[Dict[str, str]]] = None,
        on_failure: Optional[Callable[["DynamicStateManager", Dict[str, Any]], None]] = None,
    ) -> None:
        self.max_retries = max_retries
        self.max_depth = max_depth
        self.strict_mode = strict_mode
        self.enable_graceful_halt = enable_graceful_halt
        self.tool_permissions: Dict[str, Phase] = {
            row["tool_name"]: Phase(row["allowed_state"]) for row in (tool_state_matrix or [])
        }
        self.on_failure = on_failure
        self.stack: List[ExecutionFrame] = []
        self.k = 0
        self.trace: List[TraceEvent] = []

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs: Any) -> "DynamicStateManager":
        gov = cfg.get("governance", {})
        return cls(
            max_retries=gov.get("max_convergence_retries", 3),
            max_depth=gov.get("max_pda_stack_depth", 4),
            strict_mode=gov.get("strict_mode", True),
            enable_graceful_halt=gov.get("enable_graceful_halt", True),
            tool_state_matrix=cfg.get("tool_state_matrix"),
            **kwargs,
        )

    # ---- stack ---------------------------------------------------------- #
    @property
    def current(self) -> ExecutionFrame:
        if not self.stack:
            raise TransitionError("state machine not started")
        return self.stack[-1]

    @property
    def halted(self) -> bool:
        return bool(self.stack) and self.current.phase == Phase.GRACEFUL_HALT

    @property
    def done(self) -> bool:
        return bool(self.stack) and self.current.phase in TERMINAL

    def start(self, file_scope: Optional[str] = None, symbol_allowlist: Tuple[str, ...] = ()) -> None:
        self.stack = [ExecutionFrame(Phase.REPO_ANALYSIS, file_scope=file_scope, symbol_allowlist=tuple(symbol_allowlist))]
        self.k = 0
        self.trace = []
        self._log("START", Phase.REPO_ANALYSIS, Phase.REPO_ANALYSIS, {})

    def push(self, frame: ExecutionFrame) -> None:
        if len(self.stack) >= self.max_depth:
            raise TransitionError(f"PDA stack depth bound exceeded (delta_max={self.max_depth})")
        self.stack.append(frame)

    def pop(self) -> ExecutionFrame:
        if len(self.stack) <= 1:
            raise TransitionError("cannot pop the root frame")
        return self.stack.pop()

    # ---- transitions ---------------------------------------------------- #
    def fire(self, event: Union[Event, str], payload: Optional[Dict[str, Any]] = None, **frame_changes: Any) -> ExecutionFrame:
        """Apply ``event``. ``frame_changes`` update ExecutionFrame fields;
        ``payload`` is free-form data (e.g. failed patch tokens) passed to
        ``on_failure`` and recorded in the trace."""
        event = Event(event)
        payload = payload or {}
        src = self.current
        if src.phase in TERMINAL:
            raise TransitionError(f"machine is terminal ({src.phase.value})")

        key = (src.phase, event)
        if key not in TRANSITIONS:
            if self.strict_mode:
                raise TransitionError(f"illegal transition: {src.phase.value} --{event.value}-->")
            self._log(event.value, src.phase, src.phase, {"ignored": True})
            return src

        target, op = TRANSITIONS[key]
        detail = {kk: v for kk, v in frame_changes.items() if kk != "symbol_allowlist"}
        detail.update({kk: v for kk, v in payload.items() if isinstance(v, (str, int, float, bool))})

        if event in FAILURE_EVENTS:
            self.k += 1
            if self.on_failure:
                self.on_failure(self, payload)
            if self.enable_graceful_halt and self.k >= self.max_retries:
                return self._halt(event, src, f"retry counter k={self.k} >= K_max={self.max_retries}", detail)

        if op == "replace":
            self.stack[-1] = src.with_phase(target, **frame_changes)
        elif op == "push":
            try:
                self.push(src.with_phase(target, depth=src.depth + 1, **frame_changes))
            except TransitionError as exc:
                return self._halt(event, src, str(exc), detail)
        elif op == "pop":
            self.pop()
            self.stack[-1] = self.current.with_phase(target)
        self._log(event.value, src.phase, self.current.phase, detail)
        return self.current

    def _halt(self, event: Event, src: ExecutionFrame, reason: str, detail: Optional[Dict[str, Any]] = None) -> ExecutionFrame:
        self.stack = [src.with_phase(Phase.GRACEFUL_HALT, depth=0)]
        self._log(event.value, src.phase, Phase.GRACEFUL_HALT, {**(detail or {}), "reason": reason})
        return self.current

    # ---- tool gating ---------------------------------------------------- #
    def tool_allowed(self, tool_name: str) -> bool:
        required = self.tool_permissions.get(tool_name)
        if required is None:
            return not self.strict_mode
        return bool(self.stack) and self.current.phase == required

    def require_tool(self, tool_name: str) -> None:
        if not self.tool_allowed(tool_name):
            phase = self.current.phase.value if self.stack else "UNSTARTED"
            raise TransitionError(f"tool '{tool_name}' not permitted in state {phase}")

    # ---- trace (Loop 3 input) ------------------------------------------ #
    def _log(self, event: str, src: Phase, dst: Phase, detail: Dict[str, Any]) -> None:
        self.trace.append(TraceEvent(time.time(), event, src.value, dst.value, len(self.stack) - 1, self.k, detail))

    def dump_trace(self, path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            for ev in self.trace:
                fh.write(json.dumps(asdict(ev)) + "\n")


# --------------------------------------------------------------------------- #
# Loop 3: offline calibration over trace logs
# --------------------------------------------------------------------------- #
def _split_episodes(events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    for ev in events:
        if ev.get("event") == "START" or not episodes:
            episodes.append([])
        episodes[-1].append(ev)
    return episodes


def calibrate_from_traces(
    trace_paths: Union[str, Path, List[Union[str, Path]]],
    coverage: float = 0.95,
) -> Dict[str, Any]:
    """Loop 3 (Calibration): derive routing heuristics from completed traces.

    Reads JSONL traces written by :meth:`DynamicStateManager.dump_trace` (a
    directory, a file, or a list of files; files may hold several episodes,
    each beginning with a ``START`` event) and returns summary statistics plus
    a recommended ``max_convergence_retries``: the smallest K such that at
    least ``coverage`` of the observed successes needed <= K attempts.

    Model weights are never touched; the output is meant to be merged into
    ``hneg_config.yaml`` (see :func:`apply_calibration`). Note that traces are
    right-censored at the K_max in force when they were recorded, so the
    recommendation can lower K_max confidently but can only suggest raising it
    (``halt_rate`` high and ``success_at_last_attempt_rate`` high).
    """
    if isinstance(trace_paths, (str, Path)):
        p = Path(trace_paths)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    else:
        files = [Path(f) for f in trace_paths]

    episodes: List[List[Dict[str, Any]]] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        episodes.extend(e for e in _split_episodes(events) if e)

    attempts_to_success: List[int] = []
    halts = 0
    max_depth = 0
    halt_reasons: Dict[str, int] = {}
    failure_msgs: Dict[str, int] = {}
    max_k_seen = 0
    for ep in episodes:
        final = ep[-1]["target"]
        n_patches = sum(1 for e in ep if e["event"] == Event.PATCH_EMITTED.value)
        max_depth = max(max_depth, max(e.get("depth", 0) for e in ep))
        max_k_seen = max(max_k_seen, max(e.get("k", 0) for e in ep))
        for e in ep:
            if e["event"] in (Event.BUILD_FAILED.value, Event.TESTS_FAILED.value):
                msg = str(e.get("detail", {}).get("error", "")).strip().splitlines()
                key = (msg[0] if msg else "<no message>")[:120]
                failure_msgs[key] = failure_msgs.get(key, 0) + 1
        if final == Phase.DONE.value:
            attempts_to_success.append(max(n_patches, 1))
        elif final == Phase.GRACEFUL_HALT.value:
            halts += 1
            reason = str(ep[-1].get("detail", {}).get("reason", "unknown"))
            halt_reasons[reason] = halt_reasons.get(reason, 0) + 1

    n = len(episodes)
    hist: Dict[int, int] = {}
    for a in attempts_to_success:
        hist[a] = hist.get(a, 0) + 1
    recommended = None
    if attempts_to_success:
        need = coverage * len(attempts_to_success)
        acc = 0
        for k in sorted(hist):
            acc += hist[k]
            if acc >= need:
                recommended = k
                break
    last = max(hist) if hist else 0
    return {
        "n_episodes": n,
        "success_rate": round(len(attempts_to_success) / n, 4) if n else 0.0,
        "halt_rate": round(halts / n, 4) if n else 0.0,
        "attempts_to_success_hist": dict(sorted(hist.items())),
        "success_at_last_attempt_rate": round(hist.get(last, 0) / len(attempts_to_success), 4) if hist else 0.0,
        "max_stack_depth_observed": max_depth,
        "max_k_observed": max_k_seen,
        "halt_reasons": halt_reasons,
        "top_failure_messages": dict(sorted(failure_msgs.items(), key=lambda kv: -kv[1])[:10]),
        "recommended": {
            "max_convergence_retries": recommended,
            "max_pda_stack_depth": max_depth + 1 if n else None,
        },
    }


def apply_calibration(cfg: Dict[str, Any], calibration: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``cfg`` with non-null recommended governance values merged in."""
    out = json.loads(json.dumps(cfg))
    gov = out.setdefault("governance", {})
    rec = calibration.get("recommended", {})
    if rec.get("max_convergence_retries"):
        gov["max_convergence_retries"] = int(rec["max_convergence_retries"])
    if rec.get("max_pda_stack_depth"):
        gov["max_pda_stack_depth"] = max(int(rec["max_pda_stack_depth"]), gov.get("max_pda_stack_depth", 1))
    return out


if __name__ == "__main__":  # python -m src.state_machine <trace_dir> [coverage]
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.state_machine <trace_dir_or_file> [coverage]")
    cov = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    print(json.dumps(calibrate_from_traces(sys.argv[1], coverage=cov), indent=2))
