#!/usr/bin/env python3
"""GGUF sandbox smoke test for the H-NEG stack.

Default model: ``unsloth/gemma-4-E2B-it-GGUF`` / ``gemma-4-E2B-it-Q4_K_M.gguf``
(3.1 GB, from ``hneg_config.yaml: sandbox``), run through llama.cpp on one
Kaggle T4, an 8-12 GB GPU, or a laptop CPU. It verifies that the governance layers actually work with
the real tokenizer and decoder before you spend GPU hours on the 31B model.

Checks (any failure -> exit code 1; mask ``fallbacks`` are reported too)
-----------------------------------
1. gate      The identifier gate, built from the CPG allowlist, masks a token
             that would start the out-of-scope name ``secure_compare`` and
             keeps in-scope ``expected`` / ``signature`` legal.
2. trie      With every attempt forced to fail, no attempt reproduces a token
             sequence already in the failure trie.
3. halt      The forced-failure run ends in GRACEFUL_HALT with k == K_max.

Then one real governed repair of the demo bug is run and reported (its
outcome depends on the model, so it does not affect the exit code).

Usage
-----
python experiments/sandbox_smoke.py                       # downloads the E2B Q4_K_M GGUF (cached)
python experiments/sandbox_smoke.py --gguf-repo unsloth/gemma-4-E4B-it-GGUF --gguf-file gemma-4-E4B-it-Q4_K_M.gguf
python experiments/sandbox_smoke.py --gguf-path ~/models/model.gguf --n-gpu-layers 0
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import (  # noqa: E402
    CodePropertyGraph, DynamicStateManager, Event, FailedDiffTrie, HNEGLogitsProcessor,
    IdentifierScopeGate, Phase, load_config, reflex_check,
)
from src.llamacpp_backend import GGUFRunner  # noqa: E402

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
BUGGY = textwrap.dedent('''\
    import hmac
    import hashlib
    from auth.keys import load_secret


    def sign(payload: bytes) -> str:
        return hmac.new(load_secret(), payload, hashlib.sha256).hexdigest()


    def verify_token(payload: bytes, signature: str) -> bool:
        expected = sign(payload)
        return expected == signature
''')


def make_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="hneg_sandbox_")) / "demo"
    (repo / "auth").mkdir(parents=True)
    (repo / "auth" / "__init__.py").write_text("")
    (repo / "auth" / "jwt.py").write_text(BUGGY)
    (repo / "auth" / "keys.py").write_text("def load_secret() -> bytes:\n    return b'dev-secret'\n")
    return repo


def is_fixed(patch: str) -> bool:
    """Stand-in test suite: verify_token must use hmac.compare_digest and still work."""
    try:
        tree = ast.parse(patch)
    except SyntaxError:
        return False
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "verify_token"), None)
    if fn is None or not any(isinstance(n, ast.Attribute) and n.attr == "compare_digest" for n in ast.walk(fn)):
        return False
    ns = {"load_secret": lambda: b"dev-secret"}
    try:
        exec(compile(patch.replace("from auth.keys import load_secret", ""), "<patch>", "exec"), ns)
        good = ns["sign"](b"x")
        return ns["verify_token"](b"x", good) is True and ns["verify_token"](b"x", "0" * len(good)) is False
    except Exception:
        return False


def messages_for(source: str, feedback: str = None):
    user = ("Fix the timing-unsafe signature comparison in `verify_token`. Return the complete corrected "
            f"file `auth/jwt.py` in one ```python block and nothing else.\n\n```python\n{source}```")
    msgs = [{"role": "user", "content": user}]
    if feedback:
        msgs += [{"role": "assistant", "content": feedback[0]},
                 {"role": "user", "content": feedback[1] + " Produce a different fix."}]
    return msgs


def episode(runner, cfg, allow, force_fail: bool, label: str):
    gate = IdentifierScopeGate(runner.tok, allow)
    trie = FailedDiffTrie(cfg["symbolic_engine"]["trie_min_prefix_len"])
    proc = HNEGLogitsProcessor(runner.tok, failed_trie=trie, grammar_gate=gate,
                               softcap=cfg["symbolic_engine"].get("logit_softcap"))
    sm = DynamicStateManager.from_config(cfg)
    sm.start("auth/jwt.py", tuple(allow))
    feedback, replayed, fallbacks, patch = None, 0, 0, ""
    print(f"\n=== {label} ===")
    while not sm.done:
        if sm.current.phase == Phase.REPO_ANALYSIS:
            sm.fire(Event.TARGET_LOCATED, ast_node="verify_token")
        text, ids, n_in, dt = runner.generate(messages_for(BUGGY, feedback), proc)
        replayed += int(trie.contains(ids))
        fallbacks += proc.stats["fallbacks"]
        m = CODE_BLOCK.search(text)
        patch = m.group(1) if m else text
        sm.fire(Event.PATCH_EMITTED)
        r = reflex_check(patch, allowlist=allow, latency_cap_ms=cfg["symbolic_engine"]["reflex_latency_cap_ms"])
        ok = (not force_fail) and r.ok and is_fixed(patch)
        tps = len(ids) / dt if dt else 0.0
        print(f"attempt {sm.k + 1}: {len(ids)} tok in {dt:.1f}s ({tps:.1f} tok/s, H-NEG overhead "
              f"{runner.last_overhead_s:.2f}s) reflex_ok={r.ok} fixed={ok} mask={proc.stats}")
        if not r.ok:
            print(f"    reflex error: {r.error}")
        for ctx in proc.fallback_log[:3]:
            print(f"    mask fallback after: {ctx[-100:]!r}")
        if ok:
            sm.fire(Event.BUILD_PASSED)
            sm.fire(Event.TESTS_PASSED)
        else:
            trie.insert(ids)
            err = "forced failure (sandbox check)" if force_fail else (r.error or "tests failed: comparison is not constant-time")
            sm.fire(Event.BUILD_FAILED, payload={"error": err})
            feedback = (text, f"That failed: {err}.")
    return sm, replayed, fallbacks, patch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", default=str(ROOT / "hneg_config.yaml"))
    ap.add_argument("--gguf-path", default=None, help="local .gguf (skips the Hub)")
    ap.add_argument("--gguf-repo", default=None, help="override sandbox.repo_id, e.g. unsloth/gemma-4-E4B-it-GGUF")
    ap.add_argument("--gguf-file", default=None, help="override sandbox.filename, e.g. gemma-4-E4B-it-Q4_K_M.gguf")
    ap.add_argument("--n-gpu-layers", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--skip-repair", action="store_true", help="only run the deterministic checks")
    args = ap.parse_args()

    cfg = load_config(args.config_file)
    sb = cfg.get("sandbox", {})
    runner = GGUFRunner(
        repo_id=args.gguf_repo or sb.get("repo_id"), filename=args.gguf_file or sb.get("filename"),
        model_path=args.gguf_path or sb.get("model_path"), n_ctx=sb.get("n_ctx", 8192),
        n_gpu_layers=args.n_gpu_layers if args.n_gpu_layers is not None else sb.get("n_gpu_layers", -1),
        n_threads=sb.get("n_threads"), max_new_tokens=args.max_new_tokens or sb.get("max_new_tokens", 1024),
    )
    print(f"model: {runner.model_path}\nvocab: {runner.tok.n_vocab}  eog ids: {sorted(runner.eog_ids)[:8]}")

    repo = make_repo()
    cpg = CodePropertyGraph.from_repo(repo)
    allow = cpg.symbol_allowlist("auth/jwt.py", "verify_token")
    print(f"CPG {cpg.summary()}  SR_id={cpg.sr_id}")
    results = {}

    # 1. gate: probe with the real tokenizer
    gate = IdentifierScopeGate(runner.tok, allow)
    probe = HNEGLogitsProcessor(runner.tok, grammar_gate=gate)
    prefix = runner.tok.encode("```python\ndef verify_token(payload, signature):\n    return", special=False)
    mask = probe.allowed_mask(torch.tensor(prefix), runner.tok.n_vocab)
    bad = runner.tok.encode(" secure_compare", special=False)[0]
    good = [runner.tok.encode(f" {w}", special=False)[0] for w in ("expected", "signature")]
    results["gate"] = (not bool(mask[bad])) and all(bool(mask[g]) for g in good)
    print(f"\n[gate] ' secure_compare' first token {bad!r} ({runner.tok.decode([bad])!r}) allowed={bool(mask[bad])}; "
          f"in-scope tokens allowed={[bool(mask[g]) for g in good]}")

    # 2-4. forced failure: trie divergence + graceful halt
    sm, replayed, fallbacks, _ = episode(runner, cfg, allow, force_fail=True, label="forced-failure run")
    results["trie"] = replayed == 0
    results["halt"] = sm.current.phase == Phase.GRACEFUL_HALT and sm.k == cfg["governance"]["max_convergence_retries"]
    print(f"replayed failed sequences: {replayed}; mask fallbacks: {fallbacks}; final: {sm.current.phase.value} k={sm.k}")

    if not args.skip_repair:
        sm, _, _, patch = episode(runner, cfg, allow, force_fail=False, label="governed repair")
        print(f"repair outcome: {sm.current.phase.value} after {sm.k + (sm.current.phase == Phase.DONE)} attempt(s)")
        print(textwrap.indent(patch.strip(), "    "))

    print("\nSANDBOX CHECKS")
    for k, v in results.items():
        print(f"  {k:7s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
