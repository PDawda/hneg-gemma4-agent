#!/usr/bin/env python3
"""Ablation benchmark harness for H-NEG (README Section 6).

Runs each bug in a task file under one or more system configurations and logs
per-bug metrics plus an aggregate table in the same schema as
``results_ablation.csv``.

Configurations
--------------
baseline      Unconstrained generation; tracebacks accumulate in context; no
              pre-test validation; stops only at --max-turns.
baseline_k3   Unconstrained generation with the same retry cap (K_max) and the same
              "latest diagnostic only" context as `full`, but no reflex check, mask,
              trie or state machine. Comparing `full` against it isolates what the
              governance layers add beyond simply stopping early.
reflex_only   + Loop 1 reflex check (AST/indent/scope) before running tests.
static_mask   + inline identifier-scope logit mask (no trie, no PDA halt).
full          + failure trie, pushdown state machine and GRACEFUL_HALT (K_max).

Task file (JSONL, one object per bug)
-------------------------------------
{"instance_id": "django__django-11099",
 "repo_path": "/data/swe/django__django-11099",   # checked-out repo at base commit
 "target_file": "django/contrib/auth/validators.py",
 "issue": "UsernameValidator allows trailing newline ...",
 "test_cmd": "python -m pytest tests/auth_tests/test_validators.py -x -q"}

Build the task file from SWE-bench Lite with ``prepare_tasks.py``.

Usage
-----
python experiments/run_benchmark.py --tasks experiments/tasks.jsonl \
    --configs baseline reflex_only static_mask full --out-dir experiments/results \
    --trace-dir experiments/traces --ablation-csv experiments/results_ablation.csv

Results are appended per bug to ``<out-dir>/per_bug_<config>.csv``; rerunning
the same command resumes where it stopped (``--fresh`` starts over). Use
``--time-budget-hours`` to stop cleanly before a session limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import run_tests  # noqa: E402
from experiments.edit_format import (EDIT_INSTRUCTIONS, apply_edits, build_code_context, edit_feedback,  # noqa: E402
                                     failed_block_index, parse_edit_blocks)
from src.cpg_parser import CodePropertyGraph, reflex_check  # noqa: E402
from src.llamacpp_backend import ContextOverflow  # noqa: E402
from src.logit_masker import EditReplayGuard, FailedDiffTrie, HNEGLogitsProcessor, IdentifierScopeGate  # noqa: E402
from src.state_machine import DynamicStateManager, Event, load_config  # noqa: E402

CONFIGS = ("baseline", "baseline_k3", "reflex_only", "static_mask", "full")
# Same retry instruction for every configuration: in the first E4B run, the baseline's
# "Try again." vs the others' "Produce a different fix." alone changed a greedy reply.
RETRY_SUFFIX = ("\nProduce a different fix. Your previous edit was not kept: the file is unchanged, "
                "so SEARCH lines must match the original file shown above.")

CONFIG_LABELS = {
    "baseline": "Baseline Gemma (Unconstrained)",
    "reflex_only": "Gemma + Loop 1 Reflex Only",
    "static_mask": "Gemma + Identifier-Scope Logit Masking",
    "full": "Full H-NEG Pipeline (CPG + PDA + Mask + Anti-Replay)",
    "baseline_k3": "Baseline Gemma + K_max Retry Cap (no masking)",
}
ABLATION_COLUMNS = [
    "system_configuration", "config_key", "n_bugs", "parse_failure_rate_pct",
    "infinite_loop_rate_pct", "mean_token_footprint", "mean_run_latency_s", "pass_at_1_pct", "model",
]
CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a software engineering agent. Fix the bug described in the issue. "
    "Return the COMPLETE corrected contents of the target file in a single ```python code block "
    "and nothing else."
)


@dataclass
class Attempt:
    turn: int
    tokens_in: int
    tokens_out: int
    parse_ok: bool
    reflex_error: Optional[str]
    test_exit: Optional[int]
    patch_hash: str
    gen_seconds: float
    apply_error: Optional[str] = None
    mask_fallbacks: int = 0
    fallback_contexts: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None      # "eog" (which token in stop_token) or "max_tokens"
    stop_token: Optional[str] = None
    reply: str = ""                        # raw reply (special tokens visible), first 8000 chars
    oos_names: List[str] = field(default_factory=list)   # out-of-scope names the patch introduced
    repeated: bool = False                 # same (whitespace-normalised) patch as an earlier attempt


@dataclass
class BugResult:
    instance_id: str
    config: str
    resolved: bool = False
    halted: bool = False
    hit_turn_cap: bool = False
    context_overflow: bool = False   # transcript outgrew the context window after turn 1
    repeated_patches: int = 0        # attempts that re-submitted an earlier patch
    oos_attempts: int = 0            # attempts whose patch introduced out-of-scope names
    infinite_loop: bool = False
    turns: int = 0
    parse_failures: int = 0
    total_tokens: int = 0
    final_patch_tokens: int = 0
    latency_s: float = 0.0
    sr_id: str = ""
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def token_waste(self) -> float:
        return (self.total_tokens - self.final_patch_tokens) / self.total_tokens if self.total_tokens else 0.0


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class GemmaRunner:
    def __init__(self, model_id: str, max_new_tokens: int, temperature: float) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def encode(self, messages: List[Dict[str, str]]):
        """Apply the chat template, folding the system turn into the first user
        turn for templates that reject a ``system`` role, and normalising the
        return type across transformers versions (tensor vs. BatchEncoding)."""
        try:
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            if messages and messages[0]["role"] == "system":
                merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                enc = self.tok.apply_chat_template(merged + messages[2:], add_generation_prompt=True, return_tensors="pt")
            else:
                raise
        if not isinstance(enc, self.torch.Tensor):
            enc = enc["input_ids"]
        return enc.to(self.model.device)

    def generate(self, messages: List[Dict[str, str]], processor: Optional[HNEGLogitsProcessor] = None):
        from transformers import LogitsProcessorList

        ids = self.encode(messages)
        if processor is not None:
            processor.reset(patch_start=ids.shape[1])
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0)
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        if processor is not None:
            kwargs["logits_processor"] = LogitsProcessorList([processor])
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), **kwargs)
        dt = time.perf_counter() - t0
        new_ids = out[0, ids.shape[1]:].tolist()
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        return text, new_ids, ids.shape[1], dt


def make_runner(args, cfg: Dict):
    if args.backend == "hf":
        return GemmaRunner(args.model or cfg["agent"]["model_id"], args.max_new_tokens, args.temperature)
    from src.llamacpp_backend import GGUFRunner

    sb = cfg.get("sandbox", {})
    return GGUFRunner(
        repo_id=args.gguf_repo or sb.get("repo_id"), filename=args.gguf_file or sb.get("filename"),
        model_path=args.gguf_path or sb.get("model_path"),
        n_ctx=args.n_ctx or sb.get("n_ctx", 8192),
        n_gpu_layers=args.n_gpu_layers if args.n_gpu_layers is not None else sb.get("n_gpu_layers", -1),
        n_threads=sb.get("n_threads"),
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )


def model_name(args, cfg: Dict) -> str:
    if args.backend == "gguf":
        sb = cfg.get("sandbox", {})
        return Path(args.gguf_path or sb.get("model_path") or args.gguf_file or sb.get("filename", "gguf")).name
    return args.model or cfg["agent"]["model_id"]


# --------------------------------------------------------------------------- #
# Episode
# --------------------------------------------------------------------------- #
def build_messages(task: Dict, source: str, edit_mode: bool = True, context_chars: int = 24000) -> List[Dict[str, str]]:
    """Initial prompt; identical for every config (the context cannot bias the ablation)."""
    target = task["target_file"]
    if edit_mode:
        shown, full_file = build_code_context(source, task["issue"], context_chars)
        note = "" if full_file else " (only the parts relevant to the issue are shown; omitted regions are marked)"
        return [
            {"role": "system", "content": EDIT_INSTRUCTIONS},
            {"role": "user", "content": f"Issue:\n{task['issue']}\n\nTarget file `{target}`{note}:\n```python\n{shown}\n```"},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Issue:\n{task['issue']}\n\nTarget file `{target}`:\n```python\n{source}\n```"},
    ]


_CPG_CACHE: Dict[str, CodePropertyGraph] = {}


def _cpg(repo: Path) -> CodePropertyGraph:
    """Build each repo's CPG once per process (episodes never modify the checkout)."""
    key = str(repo.resolve())
    if key not in _CPG_CACHE:
        _CPG_CACHE[key] = CodePropertyGraph.from_repo(repo)
    return _CPG_CACHE[key]


def run_episode(task: Dict, config: str, runner: GemmaRunner, cfg: Dict, args) -> BugResult:
    repo = Path(task["repo_path"])
    target = task["target_file"]
    source = (repo / target).read_text(encoding="utf-8")
    res = BugResult(task["instance_id"], config)
    t_start = time.perf_counter()

    use_reflex = config in {"reflex_only", "static_mask", "full"}
    use_mask = config in {"static_mask", "full"}
    use_trie = config == "full"
    use_pda = config == "full"
    k_max = cfg["governance"]["max_convergence_retries"]
    max_turns = k_max if config in {"full", "baseline_k3"} else args.max_turns

    edit_mode = getattr(args, "patch_format", "edit") == "edit"
    cpg = _cpg(repo)
    # Diagnostic allowlist, computed for EVERY config so out-of-scope names are measured
    # uniformly; only masked/reflex configs act on it.
    diag_allow = cpg.symbol_allowlist(target)
    orig_unknown = set(reflex_check(source, diag_allow).unknown_names)
    allowlist = diag_allow if (use_mask or use_reflex) else None
    if not (use_mask or use_reflex):
        cpg = None
    if allowlist is not None and edit_mode:
        allowlist = allowlist | {"SEARCH", "REPLACE"}   # edit-block marker words
    res.sr_id = cpg.sr_id if cpg else ""

    # Anti-loop exclusion: a token trie for whole-file patches; for SEARCH/REPLACE
    # patches the edit-level guard (a token trie would block correct anchor lines).
    trie = guard = None
    if use_trie and edit_mode:
        guard = EditReplayGuard(runner.tok)
    elif use_trie:
        trie = FailedDiffTrie(cfg["symbolic_engine"].get("trie_min_prefix_len", 8))
    gate = IdentifierScopeGate(runner.tok, allowlist) if use_mask else None
    processor = None
    if use_mask or use_trie:
        processor = HNEGLogitsProcessor(runner.tok, failed_trie=trie, grammar_gate=gate, replay_guard=guard,
                                        softcap=cfg["symbolic_engine"].get("logit_softcap"))

    sm = None
    if use_pda:
        sm = DynamicStateManager.from_config(cfg)
        sm.start(target, tuple(allowlist or ()))
        sm.fire(Event.TARGET_LOCATED, ast_node=target)

    messages = build_messages(task, source, edit_mode, getattr(args, "context_chars", 24000))
    seen_hashes: Dict[str, int] = {}

    for turn in range(1, max_turns + 1):
        try:
            text, out_ids, n_in, gen_s = runner.generate(messages, processor)
        except ContextOverflow:
            if turn == 1:
                raise          # the task itself does not fit: excluded for every config
            res.context_overflow = True   # accumulated context hit the ceiling: a failure
            break
        apply_error: Optional[str] = None
        if edit_mode:
            blocks = parse_edit_blocks(text)
            candidate, apply_error = apply_edits(source, blocks)
            fingerprint = "\n".join(s + "\n=>\n" + r for s, r in blocks) or text
        else:
            m = CODE_BLOCK.search(text)
            candidate = m.group(1) if m else text
            fingerprint = candidate
        phash = hashlib.sha1(re.sub(r"\s+", "", fingerprint).encode()).hexdigest()[:12]
        repeated = phash in seen_hashes
        res.repeated_patches += int(repeated)
        seen_hashes[phash] = seen_hashes.get(phash, 0) + 1
        new_oos: List[str] = []
        if candidate is not None:
            new_oos = sorted(set(reflex_check(candidate, diag_allow).unknown_names) - orig_unknown)
        res.oos_attempts += int(bool(new_oos))
        res.total_tokens += n_in + len(out_ids)

        exit_code: Optional[int] = None
        feedback: Optional[str] = None
        reflex_error: Optional[str] = None
        if use_pda:
            sm.fire(Event.PATCH_EMITTED)

        if candidate is None:
            # Malformed edit / SEARCH text not found: counts as a parse failure.
            parse_ok = False
            feedback = edit_feedback(source, blocks, apply_error) if edit_mode else f"Your edit could not be applied: {apply_error}"
        else:
            reflex = reflex_check(candidate, allowlist if use_mask else None, cfg["symbolic_engine"]["reflex_latency_cap_ms"])
            parse_ok = reflex_check(candidate).ok
            reflex_error = reflex.error
            if use_reflex and not reflex.ok:
                feedback = f"Reflex check failed (line {reflex.lineno}): {reflex.error}"
            else:
                exit_code, test_output, _ = run_tests(repo, target, candidate, task["test_cmd"], args.test_timeout)
                if exit_code != 0:
                    feedback = f"Tests failed (exit {exit_code}):\n{test_output}"
        res.parse_failures += int(not parse_ok)

        res.attempts.append(Attempt(turn, n_in, len(out_ids), parse_ok, reflex_error, exit_code, phash, gen_s,
                                    apply_error=apply_error,
                                    mask_fallbacks=processor.stats["fallbacks"] if processor else 0,
                                    fallback_contexts=list(processor.fallback_log[:5]) if processor else [],
                                    stop_reason=(getattr(runner, "last_stop", None) or {}).get("reason"),
                                    stop_token=(getattr(runner, "last_stop", None) or {}).get("text"),
                                    reply=(getattr(runner, "last_stop", None) or {}).get("raw_reply") or text[:8000],
                                    oos_names=new_oos, repeated=repeated))
        res.turns = turn

        if exit_code == 0:
            res.resolved = True
            res.final_patch_tokens = len(out_ids)
            if use_pda:
                sm.fire(Event.BUILD_PASSED)
                sm.fire(Event.TESTS_PASSED)
            break

        if trie is not None:
            trie.insert(out_ids)
        if guard is not None:
            k = failed_block_index(apply_error)
            if k is not None and k < len(blocks) and ("not found" in apply_error or "several places" in apply_error):
                guard.add_failed_search(blocks[k][0])
            elif candidate is not None:          # applied, but failed the reflex check or the tests
                for s_, r_ in blocks:
                    guard.add_failed_pair(s_, r_)
        if use_pda:
            sm.fire(Event.BUILD_FAILED, payload={"error": (feedback or "")[:200], "turn": turn})
            if sm.halted:
                res.halted = True
                break
            sm.fire(Event.TARGET_LOCATED, ast_node=target)

        retry = feedback + RETRY_SUFFIX         # identical wording for every config
        if config == "baseline":
            # Unconstrained reflection: the whole transcript keeps growing.
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": retry}]
        else:
            # Governed reflection: only the latest diagnostic is kept in context.
            messages = messages[:2] + [{"role": "assistant", "content": text}, {"role": "user", "content": retry}]
    else:
        res.hit_turn_cap = True

    # Same definition for every config: retries exhausted (turn cap or GRACEFUL_HALT)
    # without a fix, while re-submitting an identical (whitespace-normalised) patch.
    res.infinite_loop = (res.hit_turn_cap or res.halted) and not res.resolved and max(seen_hashes.values()) >= 2
    res.latency_s = time.perf_counter() - t_start
    if sm is not None and args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        sm.dump_trace(Path(args.trace_dir) / f"{task['instance_id']}.jsonl")
    return res


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _num(row: Dict, key: str) -> float:
    v = row[key]
    if isinstance(v, str):
        return float(v == "True") if v in ("True", "False") else float(v)
    return float(v)


def aggregate(rows: List[Dict], config: str) -> Dict[str, object]:
    """Table 6.3 metrics from per-bug rows (``BugResult`` dicts or CSV rows)."""
    n = len(rows)
    attempts = sum(_num(r, "turns") for r in rows) or 1
    mean = lambda k: sum(_num(r, k) for r in rows) / n  # noqa: E731
    return {
        "system_configuration": CONFIG_LABELS[config],
        "config_key": config,
        "n_bugs": n,
        "parse_failure_rate_pct": round(100 * sum(_num(r, "parse_failures") for r in rows) / attempts, 1),
        "infinite_loop_rate_pct": round(100 * mean("infinite_loop"), 1),
        "mean_token_footprint": round(mean("total_tokens")),
        "mean_run_latency_s": round(mean("latency_s"), 1),
        "pass_at_1_pct": round(100 * mean("resolved"), 1),
        "graceful_halt_rate_pct": round(100 * mean("halted"), 1),
        "mean_token_waste": round(mean("token_waste"), 3),
    }


def _read_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _append_row(path: Path, row: Dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _code_fingerprint() -> str:
    """Hash over the code and config that produce the results (works without git)."""
    h = hashlib.sha256()
    for f in sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "experiments").glob("*.py"))
                    + [ROOT / "hneg_config.yaml"]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def collect_meta(args, cfg: Dict, runner, tasks: List[Dict]) -> Dict:
    """Everything needed to say exactly what produced the results."""
    import platform

    meta: Dict[str, object] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "backend": args.backend,
        "model": model_name(args, cfg),
        "configs": list(args.configs),
        "n_tasks": len(tasks),
        "task_ids_sha256": hashlib.sha256("\n".join(t["instance_id"] for t in tasks).encode()).hexdigest()[:16],
        "code_sha256": _code_fingerprint(),
        "governance": cfg.get("governance"),
        "symbolic_engine": cfg.get("symbolic_engine"),
        "args": {k: v for k, v in vars(args).items() if k not in ("fresh",)},
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import subprocess
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                            text=True).stdout.strip() or None
    except Exception:
        meta["git_commit"] = None
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    if args.backend == "gguf":
        import llama_cpp
        meta["llama_cpp_python"] = llama_cpp.__version__
        meta["model_path"] = runner.model_path
        meta["model_sha256"] = _sha256_file(Path(runner.model_path))
        meta["n_ctx"] = runner.n_ctx
    else:
        import transformers
        meta["transformers"] = transformers.__version__
        meta["model_revision"] = getattr(runner.model.config, "_commit_hash", None)
    return meta


def write_summary(args, cfg: Dict, out: Path, configs: List[str], instance_ids: List[str]) -> List[Dict]:
    """Aggregate over the bugs completed by *every* selected config, so the
    configurations are compared on the same set."""
    per_config = {c: {r["instance_id"]: r for r in _read_rows(out / f"per_bug_{c}.csv")} for c in configs}
    common = [i for i in instance_ids if all(i in per_config[c] for c in configs)]
    if not common:
        print("No bug has been completed by all selected configs yet; no summary written.")
        return []
    rows = []
    for c in configs:
        row = {**aggregate([per_config[c][i] for i in common], c), "model": model_name(args, cfg), "backend": args.backend}
        row["system_configuration"] = row["system_configuration"].replace("Gemma", f"Gemma [{row['model']}]", 1)
        rows.append(row)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary over {len(common)} bug(s) written to {out / 'summary.csv'}")
    if args.ablation_csv:
        with open(args.ablation_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ABLATION_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Ablation table written to {args.ablation_csv}")
    for row in rows:
        print(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="JSONL task file")
    ap.add_argument("--config-file", default=str(ROOT / "hneg_config.yaml"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--model", default=None, help="override agent.model_id from the config (hf backend)")
    ap.add_argument("--backend", choices=("hf", "gguf"), default="hf",
                    help="hf = transformers (31B target); gguf = llama.cpp sandbox from the config's sandbox section")
    ap.add_argument("--gguf-path", default=None, help="local .gguf file (overrides sandbox.repo_id/filename)")
    ap.add_argument("--gguf-repo", default=None, help="override sandbox.repo_id")
    ap.add_argument("--gguf-file", default=None, help="override sandbox.filename")
    ap.add_argument("--n-gpu-layers", type=int, default=None, help="gguf: layers to offload (0 = CPU only)")
    ap.add_argument("--n-ctx", type=int, default=None, help="gguf: context window")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N tasks")
    ap.add_argument("--limit", type=int, default=None, help="only run N tasks (after --offset)")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="stop starting new bugs after this much wall time (e.g. 11 on Kaggle); rerun to resume")
    ap.add_argument("--fresh", action="store_true", help="discard previous results in --out-dir instead of resuming")
    ap.add_argument("--max-turns", type=int, default=10, help="turn cap for non-PDA configs")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--patch-format", choices=("edit", "full"), default="edit",
                    help="edit = SEARCH/REPLACE blocks (default); full = return the whole corrected file")
    ap.add_argument("--context-chars", type=int, default=24000,
                    help="edit mode: show files up to this size whole, else an issue-relevant excerpt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--test-timeout", type=int, default=600)
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results"))
    ap.add_argument("--trace-dir", default=None, help="write state-machine traces (Loop 3 input)")
    ap.add_argument("--ablation-csv", default=None,
                    help="also write the Table 6.3 columns to this path (e.g. experiments/results_ablation.csv)")
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config(args.config_file)
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    tasks = tasks[args.offset:]
    if args.limit:
        tasks = tasks[: args.limit]
    missing = [t["instance_id"] for t in tasks if not t.get("repo_path") or not Path(t["repo_path"]).is_dir()]
    if missing:
        sys.exit(f"{len(missing)} task(s) have no checked-out repo_path (e.g. {missing[0]}); "
                 "run experiments/prepare_tasks.py without --select-only first")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_files = [out / f"{kind}_{c}.{ext}" for c in args.configs
                    for kind, ext in (("per_bug", "csv"), ("attempts", "jsonl"))] + [out / "errors.jsonl", out / "run_meta.json"]
    if args.fresh:
        for f in result_files:
            f.unlink(missing_ok=True)
        if args.trace_dir:
            for f in Path(args.trace_dir).glob("*.jsonl"):
                f.unlink()
    done = {c: {r["instance_id"] for r in _read_rows(out / f"per_bug_{c}.csv")} for c in args.configs}
    errored = set()
    if (out / "errors.jsonl").exists():
        errored = {json.loads(l)["instance_id"] for l in (out / "errors.jsonl").read_text().splitlines() if l.strip()}
    todo = [t for t in tasks if t["instance_id"] not in errored and any(t["instance_id"] not in done[c] for c in args.configs)]
    n_complete = sum(all(t["instance_id"] in done[c] for c in args.configs) for t in tasks)
    print(f"{len(tasks)} task(s) selected: {n_complete} complete, {len(errored)} previously errored, "
          f"{len(todo)} to run", flush=True)

    runner = make_runner(args, cfg) if todo else None
    if runner is not None:
        meta = collect_meta(args, cfg, runner, tasks)
        meta_path = out / "run_meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            for key in ("model", "model_sha256", "model_revision", "code_sha256", "governance", "symbolic_engine"):
                if key in old and old.get(key) != meta.get(key):
                    sys.exit(f"Refusing to resume: '{key}' differs from the earlier run in {out} "
                             f"({old.get(key)!r} vs {meta.get(key)!r}). Use a new --out-dir or --fresh.")
            meta["first_created"] = old.get("first_created", old.get("created"))
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"run metadata: {meta_path} (model sha256 {str(meta.get('model_sha256', 'n/a'))[:16]}...)", flush=True)
    stopped_early = False
    # Bug-major order: every config runs on a bug before moving on, so a partial
    # run still compares all configurations on the same bugs.
    for i, task in enumerate(todo, 1):
        if args.time_budget_hours is not None and (time.time() - t_start) > args.time_budget_hours * 3600:
            print(f"Time budget of {args.time_budget_hours} h reached; rerun the same command to resume.")
            stopped_early = True
            break
        for config in args.configs:
            if task["instance_id"] in done[config]:
                continue
            try:
                r = run_episode(task, config, runner, cfg, args)
            except Exception as exc:  # e.g. prompt longer than the context window
                with open(out / "errors.jsonl", "a") as fh:
                    fh.write(json.dumps({"instance_id": task["instance_id"], "config": config,
                                         "error": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
                print(f"[{config}] {task['instance_id']}: ERROR {type(exc).__name__}: {exc}", flush=True)
                break
            row = {k: v for k, v in asdict(r).items() if k != "attempts"}
            row["token_waste"] = round(r.token_waste, 3)
            _append_row(out / f"per_bug_{config}.csv", row)
            with open(out / f"attempts_{config}.jsonl", "a") as fh:
                fh.write(json.dumps({"instance_id": r.instance_id, "attempts": [asdict(a) for a in r.attempts]}) + "\n")
            done[config].add(r.instance_id)
            n_apply_err = sum(1 for a in r.attempts if a.apply_error)
            n_fallback = sum(a.mask_fallbacks for a in r.attempts)
            print(f"[{i}/{len(todo)}] [{config}] {r.instance_id}: resolved={r.resolved} turns={r.turns} "
                  f"tokens={r.total_tokens} halted={r.halted} overflow={r.context_overflow} "
                  f"apply_errors={n_apply_err} mask_fallbacks={n_fallback} {r.latency_s:.0f}s", flush=True)

    write_summary(args, cfg, out, args.configs, [t["instance_id"] for t in tasks])
    if stopped_early:
        sys.exit(3)


if __name__ == "__main__":
    main()
