#!/usr/bin/env python3
"""Check each claim of the paper against a finished ablation run.

Reads the ``run_benchmark.py`` output directory (``per_bug_<config>.csv``,
``attempts_<config>.jsonl``, ``run_meta.json``) and gives every claim one of
three verdicts:

    VERIFIED       the measurement supports the claim (CI / exact test excludes "no effect")
    NOT SUPPORTED  the measurement contradicts the claim
    INCONCLUSIVE   too little data to decide (e.g. no bug resolved, CI includes 0)

Claims fall into two groups:

* **Guarantees** (C1-C5) are properties of the governance mechanism. They hold
  or fail on every bug, so they can be checked at any sample size and with any
  model, resolved or not.
* **Effects** (C6-C10) are statistical. They need resolved bugs and enough of
  them, and they are measured for the model that was run: the absolute numbers
  are not those of the 31B model claimed in the paper's Table 6.3.

The decisive comparison is ``full vs baseline_k3``: same retry cap and same
short context, so any difference comes from masking, anti-replay and the
state machine rather than from stopping early.

Writes ``claims_verification.md`` and ``claims.json`` next to the inputs.

Usage
-----
python experiments/verify_claims.py --results-dir experiments/results --gpu-cost-per-hour 0.35
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("savings_report", HERE / "savings_report.py")
sr = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("savings_report", sr)
_spec.loader.exec_module(sr)

VERIFIED, NOT_SUPPORTED, INCONCLUSIVE = "VERIFIED", "NOT SUPPORTED", "INCONCLUSIVE"
ICON = {VERIFIED: "✅", NOT_SUPPORTED: "❌", INCONCLUSIVE: "⚠️"}


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def discordant(data, ids: Sequence[str], a: str, b: str):
    ra = {i: sr._f(data[a][i]["resolved"]) > 0 for i in ids}
    rb = {i: sr._f(data[b][i]["resolved"]) > 0 for i in ids}
    a_only = sum(1 for i in ids if ra[i] and not rb[i])
    b_only = sum(1 for i in ids if rb[i] and not ra[i])
    return a_only, b_only


def attempt_rates(results_dir: Path, config: str, ids: Sequence[str]) -> Dict[str, Optional[float]]:
    """Syntax-failure and edit-apply-failure rates from the attempt log."""
    path = results_dir / f"attempts_{config}.jsonl"
    if not path.exists():
        return {"attempts": None, "syntax_fail_pct": None, "apply_fail_pct": None}
    keep = set(ids)
    total = applied = syntax_bad = apply_bad = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["instance_id"] not in keep:
            continue
        for a in rec["attempts"]:
            total += 1
            if a.get("apply_error"):
                apply_bad += 1
            else:
                applied += 1
                syntax_bad += int(not a.get("parse_ok", True))
    return {
        "attempts": total,
        "syntax_fail_pct": 100 * syntax_bad / applied if applied else None,
        "apply_fail_pct": 100 * apply_bad / total if total else None,
        "applied_attempts": applied,
    }


def _max_turns(rows: Sequence[Dict]) -> int:
    return int(max(sr._f(r["turns"]) for r in rows)) if rows else 0


def _fmt(v, nd=1, suffix=""):
    return "n/a" if v is None else f"{v:,.{nd}f}{suffix}"


def _ci(ci, nd=1):
    return "" if not ci else f" [95% CI {ci[0]:,.{nd}f}, {ci[1]:,.{nd}f}]"


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #
def evaluate(results_dir: Path, gpu_cost: Optional[float] = None, price: Optional[float] = None,
             n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Dict[str, object]:
    data, ids = sr.load(results_dir)
    if not ids:
        raise SystemExit("no bug has been completed by every configuration yet")
    meta = json.loads((results_dir / "run_meta.json").read_text()) if (results_dir / "run_meta.json").exists() else {}
    k_max = (meta.get("governance") or {}).get("max_convergence_retries", 3)
    have = set(data)
    stats = {c: sr.config_stats([data[c][i] for i in ids]) for c in have}
    comps = {(a, b): sr.compare(data, ids, a, b, n_boot, seed) for a, b in sr.COMPARISONS if a in have and b in have}
    att = {c: attempt_rates(results_dir, c, ids) for c in have}
    roi_res = sr.roi(data, ids, [c for c in sr.ORDER if c in have], gpu_cost, price, 1000, n_boot, seed) \
        if (gpu_cost or price) else None
    n = len(ids)
    claims: List[Dict[str, object]] = []

    def add(cid, group, claim, paper, verdict, measured, why):
        claims.append({"id": cid, "group": group, "claim": claim, "paper_value": paper,
                       "verdict": verdict, "measured": measured, "reason": why})

    full = stats.get("full")
    if full is None:
        raise SystemExit("the 'full' configuration is required")
    loopers = {c: stats[c]["loop_pct"] for c in have if c != "full"}

    # C1 loop elimination
    worst = max(loopers.items(), key=lambda kv: kv[1]) if loopers else (None, None)
    if full["loop_pct"] > 0:
        v, why = NOT_SUPPORTED, "full H-NEG still re-submitted an identical failed patch until retries ran out"
    elif worst[1]:
        v, why = VERIFIED, f"0% for full H-NEG while {worst[0]} looped on {worst[1]:.1f}% of bugs"
    else:
        v, why = INCONCLUSIVE, "no configuration looped, so there was nothing to eliminate"
    add("C1", "guarantee", "Infinite-loop elimination (identical failed patch re-submitted until retries ran out)",
        "42.1% → 0.0%", v, f"full {full['loop_pct']:.1f}% · " + " · ".join(f"{c} {p:.1f}%" for c, p in loopers.items()), why)

    # C2 bounded attempts
    mt = _max_turns([data["full"][i] for i in ids])
    v = VERIFIED if mt <= k_max else NOT_SUPPORTED
    add("C2", "guarantee", f"Bounded convergence loop: at most K_max = {k_max} attempts, then GRACEFUL_HALT",
        f"k ≤ {k_max}", v, f"max attempts {mt}; halt rate {full['halt_pct']:.1f}%; max tokens/bug {_fmt(full['max_tokens'], 0)}"
        + (f" vs baseline {_fmt(stats['baseline']['max_tokens'], 0)}" if "baseline" in stats else ""),
        "hard cap held on every bug" if v == VERIFIED else "an episode exceeded K_max")

    # C3 anti-replay
    rep = full.get("repeated_attempts")
    others = {c: stats[c].get("repeated_attempts") for c in have if c != "full"}
    if rep is None:
        v, why = INCONCLUSIVE, "run predates the repeated-patch counter"
    elif rep == 0:
        v, why = (VERIFIED, "no failed patch was ever re-submitted under full H-NEG") if any(others.values()) else \
            (INCONCLUSIVE, "0 repeats, but no other configuration repeated either")
    else:
        v, why = NOT_SUPPORTED, f"{rep} attempts re-submitted an earlier patch"
    add("C3", "guarantee", "Failed-patch exclusion: a failed patch is never re-submitted", "trie pruning",
        v, f"full {rep} repeated attempts · " + " · ".join(f"{c} {r}" for c, r in others.items()), why)

    # C4 parse failure rate
    a_full = att["full"]
    syn = a_full["syntax_fail_pct"]
    if syn is None:
        v, why = INCONCLUSIVE, "no edit was applied, so syntax could not be measured"
    elif syn == 0:
        v, why = VERIFIED, f"all {a_full['applied_attempts']} applied patches parsed"
    else:
        v, why = NOT_SUPPORTED, "some applied patches did not parse"
    add("C4", "guarantee", "0.0% parse failure rate (syntactically valid patches)", "0.0%", v,
        f"syntax failures {_fmt(syn, 1, '%')} of applied patches; edits that could not be applied "
        f"{_fmt(a_full['apply_fail_pct'], 1, '%')} of attempts (not prevented by the mask; same for every config)", why)

    # C5 out-of-scope identifiers
    oos = full.get("oos_attempt_pct")
    oth = {c: stats[c].get("oos_attempt_pct") for c in have if c != "full"}
    if oos is None:
        v, why = INCONCLUSIVE, "run predates the out-of-scope counter"
    elif oos == 0:
        v, why = (VERIFIED, "the mask kept every patch inside the CPG scope") if any(oth.values()) else \
            (INCONCLUSIVE, "0% for full, but unconstrained configurations did not invent names either")
    else:
        v, why = NOT_SUPPORTED, "patches under full H-NEG introduced out-of-scope names"
    add("C5", "guarantee", "No out-of-scope identifiers (CPG-scoped logit mask)", "masked to −∞", v,
        f"full {_fmt(oos, 1, '%')} · " + " · ".join(f"{c} {_fmt(p, 1, '%')}" for c, p in oth.items()), why)

    # C6 / C7 token reduction and speedup vs baseline
    c = comps.get(("full", "baseline"))
    if c:
        lo = (c["token_reduction_ci"] or (None, None))[0]
        v = INCONCLUSIVE if lo is None else (VERIFIED if lo > 0 else NOT_SUPPORTED if c["token_reduction_ci"][1] < 0 else INCONCLUSIVE)
        add("C6", "effect", "Token reduction per bug vs unconstrained baseline", "−64.2%", v,
            f"{_fmt(c['token_reduction_pct'], 1, '%')}{_ci(c['token_reduction_ci'])}",
            "mostly the retry cap: see C8" if v == VERIFIED else "CI does not exclude zero")
        lo = (c["speedup_ci"] or (None, None))[0]
        v = INCONCLUSIVE if lo is None else (VERIFIED if lo > 1 else NOT_SUPPORTED if c["speedup_ci"][1] < 1 else INCONCLUSIVE)
        add("C7", "effect", "End-to-end speedup per bug vs unconstrained baseline", "3.93×", v,
            f"{_fmt(c['speedup_x'], 2, '×')}{_ci(c['speedup_ci'], 2)}",
            "CI excludes 1×" if v == VERIFIED else "CI includes 1×" if v == INCONCLUSIVE else "slower than baseline")

    # C8 governance beyond the retry cap (the decisive one)
    if ("full", "baseline_k3") in comps:
        ck = comps[("full", "baseline_k3")]
        fo, ko = discordant(data, ids, "full", "baseline_k3")
        p = mcnemar_exact(fo, ko)
        tf, tk = full["tokens_per_resolved"], stats["baseline_k3"]["tokens_per_resolved"]
        if full["resolved"] == 0 and stats["baseline_k3"]["resolved"] == 0:
            v, why = INCONCLUSIVE, "neither configuration resolved a bug: the model, not the governance, is the bottleneck"
        elif p < alpha and fo > ko:
            v, why = VERIFIED, f"full resolved {fo} bugs that baseline_k3 missed vs {ko} the other way (exact McNemar p = {p:.3g})"
        elif p < alpha and ko > fo:
            v, why = NOT_SUPPORTED, f"baseline_k3 resolved more (exact McNemar p = {p:.3g})"
        else:
            v, why = INCONCLUSIVE, f"discordant bugs {fo} vs {ko}, exact McNemar p = {p:.3g}; more bugs needed"
        add("C8", "effect", "Governance adds value beyond the retry cap (full vs baseline_k3: same K_max and context)",
            "implied by Table 6.3", v,
            f"Pass@1 {full['pass_at_1_pct']:.1f}% vs {stats['baseline_k3']['pass_at_1_pct']:.1f}% · "
            f"tokens/resolved {_fmt(tf, 0)} vs {_fmt(tk, 0)} · tokens/bug {_fmt(-ck['token_reduction_pct'] if ck['token_reduction_pct'] is not None else None, 1, '%')} change",
            why)

    # C9 Pass@1 vs baseline
    if "baseline" in have:
        fo, bo = discordant(data, ids, "full", "baseline")
        p = mcnemar_exact(fo, bo)
        if full["resolved"] == 0 and stats["baseline"]["resolved"] == 0:
            v, why = INCONCLUSIVE, "no bug resolved by either configuration"
        elif p < alpha:
            v, why = (VERIFIED if fo > bo else NOT_SUPPORTED), f"exact McNemar p = {p:.3g} ({fo} vs {bo} discordant bugs)"
        elif fo >= bo:
            v, why = INCONCLUSIVE, f"not significant (p = {p:.3g}, {fo} vs {bo}); the paper's claim is not contradicted"
        else:
            v, why = INCONCLUSIVE, f"baseline ahead but not significant (p = {p:.3g}, {fo} vs {bo})"
        add("C9", "effect", "Higher issue-resolution rate (Pass@1) than the unconstrained baseline", "19.2% → 31.8%",
            v, f"{full['pass_at_1_pct']:.1f}% ({full['resolved']}/{n}) vs {stats['baseline']['pass_at_1_pct']:.1f}% "
            f"({stats['baseline']['resolved']}/{n})", why)

    # C10 cost per resolved bug
    if roi_res:
        for sv in roi_res["savings"]:
            if (sv["a"], sv["b"]) not in (("full", "baseline"), ("full", "baseline_k3")):
                continue
            ci = sv["ci"]
            if sv["cost_per_resolved_saving_pct"] is None:
                v, why = INCONCLUSIVE, "a configuration resolved nothing, so cost per resolved bug is undefined"
            elif ci is None:
                v, why = INCONCLUSIVE, "too few resolved bugs for a confidence interval"
            else:
                v = VERIFIED if ci[0] > 0 else NOT_SUPPORTED if ci[1] < 0 else INCONCLUSIVE
                why = "CI excludes zero" if v != INCONCLUSIVE else "CI includes zero"
            add("C10" + ("a" if sv["b"] == "baseline" else "b"), "effect",
                f"Lower cost per resolved bug: full vs {sv['b']}", "−64.2% tokens, 3.93× per resolved patch", v,
                f"{_fmt(sv['cost_per_resolved_saving_pct'], 1, '%')} saving{_ci(ci)}", why)

    return {"n": n, "model": meta.get("model"), "model_sha256": meta.get("model_sha256"),
            "code_sha256": meta.get("code_sha256"), "task_ids_sha256": meta.get("task_ids_sha256"),
            "k_max": k_max, "claims": claims, "attempt_rates": att}


def render(res: Dict[str, object]) -> str:
    lines = [f"# Claim verification: {res['model'] or 'unknown model'}", "",
             f"n = **{res['n']}** bugs completed by every configuration · model SHA-256 `{res['model_sha256'] or 'n/a'}` · "
             f"code `{res['code_sha256'] or 'n/a'}` · task-set `{res['task_ids_sha256'] or 'n/a'}`", ""]
    counts = {v: sum(1 for c in res["claims"] if c["verdict"] == v) for v in (VERIFIED, NOT_SUPPORTED, INCONCLUSIVE)}
    lines.append(" · ".join(f"{ICON[v]} {v}: {k}" for v, k in counts.items()))
    for group, title in (("guarantee", "Guarantees (mechanism; checkable at any n)"),
                         ("effect", "Effects (statistical; measured for this model, not the 31B in Table 6.3)")):
        rows = [c for c in res["claims"] if c["group"] == group]
        if not rows:
            continue
        lines += ["", f"## {title}", "", "| # | Claim | Paper | Measured | Verdict | Why |", "| :-- | :-- | :-: | :-- | :-: | :-- |"]
        for c in rows:
            lines.append(f"| {c['id']} | {c['claim']} | {c['paper_value']} | {c['measured']} | "
                         f"{ICON[c['verdict']]} {c['verdict']} | {c['reason']} |")
    lines += ["", "_Verdicts use 95% paired bootstrap CIs over bugs and exact McNemar tests on resolved/unresolved pairs. "
              "INCONCLUSIVE means the data cannot decide, not that the claim is false._"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None)
    ap.add_argument("--price-per-mtok", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)
    rd = Path(args.results_dir)
    res = evaluate(rd, args.gpu_cost_per_hour, args.price_per_mtok, args.bootstrap, args.seed)
    md = render(res)
    (rd / "claims_verification.md").write_text(md)
    (rd / "claims.json").write_text(json.dumps(res, indent=2))
    print(md)
    print(f"written: {rd / 'claims_verification.md'} and {rd / 'claims.json'}")
    return res


if __name__ == "__main__":
    main()
