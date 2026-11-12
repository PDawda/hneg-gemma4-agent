#!/usr/bin/env python3
"""Savings report: the ablation table plus headline savings with 95% CIs.

Reads ``per_bug_<config>.csv`` (and ``run_meta.json``) from a
``run_benchmark.py`` output directory and reports, on the bugs completed by
*every* configuration present:

* the ablation table (PFR, loop rate, tokens/bug, latency/bug, Pass@1),
* per-resolved-bug cost (total tokens / total time divided by fixed bugs),
* paired comparisons with 95% bootstrap confidence intervals
  (resampling bugs with replacement, same bugs on both sides):

    full vs baseline        headline saving
    full vs baseline_k3     what the governance layers add beyond the retry cap
    baseline_k3 vs baseline what the retry cap alone buys

Token reduction is ``1 - mean_tokens(A) / mean_tokens(B)``; speedup is
``mean_latency(B) / mean_latency(A)``; dPass@1 is in percentage points.

Also reported: tail cost (p95 / max tokens and time per bug), the guarantee
counters (episodes that re-submitted a failed patch; attempts that introduced
out-of-scope names), and -- with ``--gpu-cost-per-hour`` and/or
``--price-per-mtok`` -- cost per resolved bug in dollars with 95% CIs and a
monthly projection for ``--bugs-per-month``.

Writes ``savings_report.md`` and ``savings.json`` next to the inputs.

Usage
-----
python experiments/savings_report.py --results-dir experiments/results
python experiments/savings_report.py --results-dir experiments/results \
    --gpu-cost-per-hour 0.35 --price-per-mtok 1.00 --bugs-per-month 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ORDER = ("baseline", "baseline_k3", "reflex_only", "static_mask", "full")
LABELS = {
    "baseline": "Baseline (unconstrained, up to --max-turns)",
    "baseline_k3": "Baseline + K_max retry cap (no masking)",
    "reflex_only": "+ Loop 1 reflex only",
    "static_mask": "+ Identifier-scope logit masking",
    "full": "Full H-NEG (CPG + PDA + mask + anti-replay)",
}
COMPARISONS = (("full", "baseline"), ("full", "baseline_k3"), ("baseline_k3", "baseline"))


def _f(v) -> float:
    if isinstance(v, str):
        return float(v == "True") if v in ("True", "False") else float(v)
    return float(v)


def load(results_dir: Path) -> Tuple[Dict[str, Dict[str, Dict]], List[str]]:
    data: Dict[str, Dict[str, Dict]] = {}
    for c in ORDER:
        p = results_dir / f"per_bug_{c}.csv"
        if p.exists():
            with open(p, newline="") as fh:
                data[c] = {r["instance_id"]: r for r in csv.DictReader(fh)}
    if not data:
        raise SystemExit(f"no per_bug_<config>.csv files in {results_dir}")
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    return data, common


def _pctl(vals: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile."""
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(-(-q * len(s) // 1)) - 1))]


def config_stats(rows: Sequence[Dict]) -> Dict[str, float]:
    n = len(rows)
    tot = lambda k: sum(_f(r[k]) for r in rows)  # noqa: E731
    has = lambda k: bool(rows) and k in rows[0]  # noqa: E731
    resolved = tot("resolved")
    toks = [_f(r["total_tokens"]) for r in rows]
    lats = [_f(r["latency_s"]) for r in rows]
    return {
        "p95_tokens": _pctl(toks, 0.95), "max_tokens": max(toks) if toks else None,
        "p95_latency_s": _pctl(lats, 0.95), "max_latency_s": max(lats) if lats else None,
        "repeat_episode_pct": 100 * sum(1 for r in rows if _f(r["repeated_patches"]) > 0) / n if has("repeated_patches") else None,
        "repeated_attempts": int(tot("repeated_patches")) if has("repeated_patches") else None,
        "oos_attempt_pct": 100 * tot("oos_attempts") / max(tot("turns"), 1) if has("oos_attempts") else None,
        "n": n,
        "pfr_pct": 100 * tot("parse_failures") / max(tot("turns"), 1),
        "loop_pct": 100 * tot("infinite_loop") / n,
        "halt_pct": 100 * tot("halted") / n,
        "overflow_pct": 100 * tot("context_overflow") / n if rows and "context_overflow" in rows[0] else 0.0,
        "tokens_per_bug": tot("total_tokens") / n,
        "latency_per_bug_s": tot("latency_s") / n,
        "pass_at_1_pct": 100 * resolved / n,
        "resolved": int(resolved),
        "tokens_per_resolved": tot("total_tokens") / resolved if resolved else None,
        "latency_per_resolved_s": tot("latency_s") / resolved if resolved else None,
        "mean_turns": tot("turns") / n,
    }


def _ratio_metrics(a: Sequence[Dict], b: Sequence[Dict]) -> Optional[Tuple[float, float, float]]:
    ta, tb = sum(_f(r["total_tokens"]) for r in a), sum(_f(r["total_tokens"]) for r in b)
    la, lb = sum(_f(r["latency_s"]) for r in a), sum(_f(r["latency_s"]) for r in b)
    if tb == 0 or la == 0:
        return None
    pa, pb = sum(_f(r["resolved"]) for r in a), sum(_f(r["resolved"]) for r in b)
    return 100 * (1 - ta / tb), lb / la, 100 * (pa - pb) / len(a)


def compare(data, ids: List[str], a: str, b: str, n_boot: int, seed: int) -> Dict[str, object]:
    rows_a = [data[a][i] for i in ids]
    rows_b = [data[b][i] for i in ids]
    point = _ratio_metrics(rows_a, rows_b)
    rng = random.Random(seed)
    samples: List[Tuple[float, float, float]] = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(ids)) for _ in ids]
        m = _ratio_metrics([rows_a[i] for i in idx], [rows_b[i] for i in idx])
        if m is not None:
            samples.append(m)

    def ci(k: int) -> Optional[Tuple[float, float]]:
        if len(samples) < 20:
            return None
        vals = sorted(s[k] for s in samples)
        return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]

    return {
        "a": a, "b": b, "n": len(ids),
        "token_reduction_pct": point[0] if point else None, "token_reduction_ci": ci(0),
        "speedup_x": point[1] if point else None, "speedup_ci": ci(1),
        "delta_pass_at_1_pp": point[2] if point else None, "delta_pass_at_1_ci": ci(2),
    }


def _cost_per_resolved(rows: Sequence[Dict], gpu_cost: Optional[float], price: Optional[float]) -> Optional[float]:
    resolved = sum(_f(r["resolved"]) for r in rows)
    if not resolved:
        return None
    cost = 0.0
    if gpu_cost:
        cost += sum(_f(r["latency_s"]) for r in rows) / 3600 * gpu_cost
    if price:
        cost += sum(_f(r["total_tokens"]) for r in rows) / 1e6 * price
    return cost / resolved


def roi(data, ids: List[str], configs: Sequence[str], gpu_cost: Optional[float], price: Optional[float],
        bugs_per_month: int, n_boot: int, seed: int) -> Dict[str, object]:
    """Dollar cost per resolved bug per config, savings with bootstrap CIs, monthly projection."""
    per_config = {}
    for c in configs:
        rows = [data[c][i] for i in ids]
        n = len(rows)
        cost_per_bug = ((sum(_f(r["latency_s"]) for r in rows) / 3600 * (gpu_cost or 0))
                        + (sum(_f(r["total_tokens"]) for r in rows) / 1e6 * (price or 0))) / n
        pass_rate = sum(_f(r["resolved"]) for r in rows) / n
        per_config[c] = {
            "cost_per_bug": cost_per_bug,
            "cost_per_resolved": _cost_per_resolved(rows, gpu_cost, price),
            "monthly_cost": cost_per_bug * bugs_per_month,
            "monthly_resolved": pass_rate * bugs_per_month,
        }
    savings = []
    rng = random.Random(seed)
    for a, b in COMPARISONS:
        if a not in configs or b not in configs:
            continue
        ra, rb = [data[a][i] for i in ids], [data[b][i] for i in ids]
        ca, cb = _cost_per_resolved(ra, gpu_cost, price), _cost_per_resolved(rb, gpu_cost, price)
        point = 100 * (1 - ca / cb) if ca is not None and cb else None
        samples = []
        for _ in range(n_boot):
            idx = [rng.randrange(len(ids)) for _ in ids]
            sa = _cost_per_resolved([ra[i] for i in idx], gpu_cost, price)
            sb = _cost_per_resolved([rb[i] for i in idx], gpu_cost, price)
            if sa is not None and sb:
                samples.append(100 * (1 - sa / sb))
        ci = None
        if len(samples) >= 20:
            samples.sort()
            ci = (samples[int(0.025 * len(samples))], samples[min(int(0.975 * len(samples)), len(samples) - 1)])
        savings.append({"a": a, "b": b, "cost_per_resolved_saving_pct": point, "ci": ci,
                        "bootstrap_samples_used": len(samples)})
    return {"gpu_cost_per_hour": gpu_cost, "price_per_mtok": price, "bugs_per_month": bugs_per_month,
            "per_config": per_config, "savings": savings}


def _fmt(v, nd=1, suffix=""):
    return "n/a" if v is None else f"{v:,.{nd}f}{suffix}"


def _fmt_ci(ci, nd=1):
    return "" if not ci else f" [{ci[0]:,.{nd}f}, {ci[1]:,.{nd}f}]"


def render(meta: Dict, stats: Dict[str, Dict], comps: List[Dict], n: int, roi_res: Optional[Dict] = None) -> str:
    model = meta.get("model", "unknown model")
    lines = [f"# H-NEG ablation: {model}", ""]
    lines.append(f"Bugs completed by every configuration: **n = {n}**  ")
    if meta:
        sha = str(meta.get("model_sha256") or meta.get("model_revision") or "n/a")
        lines.append(f"Model SHA-256: `{sha}`  ")
        lines.append(f"Code fingerprint: `{meta.get('code_sha256', 'n/a')}` · task-set: `{meta.get('task_ids_sha256', 'n/a')}` · "
                     f"GPU: {meta.get('gpu', 'n/a')} · K_max = {(meta.get('governance') or {}).get('max_convergence_retries', 'n/a')}")
    lines += ["", "| Configuration | PFR | Loop rate | Halt rate | Context overflow | Tokens / bug | Latency / bug | Pass@1 | Tokens / resolved | Latency / resolved |",
              "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |"]
    for c in ORDER:
        if c not in stats:
            continue
        s = stats[c]
        lines.append(f"| {LABELS[c]} | {_fmt(s['pfr_pct'], 1, '%')} | {_fmt(s['loop_pct'], 1, '%')} | {_fmt(s['halt_pct'], 1, '%')} | "
                     f"{_fmt(s['overflow_pct'], 1, '%')} | "
                     f"{_fmt(s['tokens_per_bug'], 0)} | {_fmt(s['latency_per_bug_s'], 1, ' s')} | {_fmt(s['pass_at_1_pct'], 1, '%')} "
                     f"({s['resolved']}/{s['n']}) | {_fmt(s['tokens_per_resolved'], 0)} | {_fmt(s['latency_per_resolved_s'], 1, ' s')} |")
    if comps:
        lines += ["", "## Paired comparisons (95% bootstrap CI over bugs)", "",
                  "| Comparison | Token reduction | Speedup | Δ Pass@1 (pp) |", "| :--- | :-: | :-: | :-: |"]
        for c in comps:
            lines.append(f"| {c['a']} vs {c['b']} | {_fmt(c['token_reduction_pct'], 1, '%')}{_fmt_ci(c['token_reduction_ci'])} | "
                         f"{_fmt(c['speedup_x'], 2, '×')}{_fmt_ci(c['speedup_ci'], 2)} | "
                         f"{_fmt(c['delta_pass_at_1_pp'], 1)}{_fmt_ci(c['delta_pass_at_1_ci'])} |")
    lines += ["", "## Tail cost and guarantees", "",
              "| Configuration | p95 tokens / bug | Max tokens / bug | p95 latency | Max latency | "
              "Episodes re-submitting a failed patch | Attempts introducing out-of-scope names |",
              "| :--- | :-: | :-: | :-: | :-: | :-: | :-: |"]
    for c in ORDER:
        if c not in stats:
            continue
        s = stats[c]
        rep = "n/a" if s["repeat_episode_pct"] is None else f"{s['repeat_episode_pct']:.1f}% ({s['repeated_attempts']} attempts)"
        lines.append(f"| {LABELS[c]} | {_fmt(s['p95_tokens'], 0)} | {_fmt(s['max_tokens'], 0)} | "
                     f"{_fmt(s['p95_latency_s'], 1, ' s')} | {_fmt(s['max_latency_s'], 1, ' s')} | {rep} | "
                     f"{_fmt(s['oos_attempt_pct'], 1, '%')} |")
    if roi_res:
        parts = []
        if roi_res["gpu_cost_per_hour"]:
            parts.append(f"GPU ${roi_res['gpu_cost_per_hour']:.2f}/h")
        if roi_res["price_per_mtok"]:
            parts.append(f"tokens ${roi_res['price_per_mtok']:.2f}/M")
        lines += ["", f"## ROI ({' + '.join(parts)}; projection for {roi_res['bugs_per_month']:,} bugs/month)", "",
                  "| Configuration | Cost / bug | Cost / resolved bug | Monthly cost | Bugs fixed / month |",
                  "| :--- | :-: | :-: | :-: | :-: |"]
        for c in ORDER:
            if c not in roi_res["per_config"]:
                continue
            p = roi_res["per_config"][c]
            cpr = "n/a (none resolved)" if p["cost_per_resolved"] is None else f"${p['cost_per_resolved']:,.4f}"
            lines.append(f"| {LABELS[c]} | ${p['cost_per_bug']:,.4f} | {cpr} | ${p['monthly_cost']:,.2f} | "
                         f"{p['monthly_resolved']:,.0f} |")
        if roi_res["savings"]:
            lines += ["", "| Comparison | Saving in cost per resolved bug (95% CI) |", "| :--- | :-: |"]
            for sv in roi_res["savings"]:
                lines.append(f"| {sv['a']} vs {sv['b']} | {_fmt(sv['cost_per_resolved_saving_pct'], 1, '%')}"
                             f"{_fmt_ci(sv['ci'])} |")
        lines += ["", "_Monthly figures extrapolate this sample's per-bug cost and fix rate; prices are inputs, not measurements._"]
    if n < 30:
        lines += ["", f"_Note: n = {n} is small; read the confidence intervals, not just the point estimates._"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None, help="e.g. cloud T4 hourly price")
    ap.add_argument("--price-per-mtok", type=float, default=None, help="price per million tokens (API comparison)")
    ap.add_argument("--bugs-per-month", type=int, default=1000)
    args = ap.parse_args(list(argv) if argv is not None else None)

    rd = Path(args.results_dir)
    data, ids = load(rd)
    if not ids:
        raise SystemExit("no bug has been completed by every configuration yet")
    meta = json.loads((rd / "run_meta.json").read_text()) if (rd / "run_meta.json").exists() else {}
    stats = {c: config_stats([data[c][i] for i in ids]) for c in data}
    comps = [compare(data, ids, a, b, args.bootstrap, args.seed) for a, b in COMPARISONS if a in data and b in data]
    roi_res = None
    if args.gpu_cost_per_hour or args.price_per_mtok:
        roi_res = roi(data, ids, [c for c in ORDER if c in data], args.gpu_cost_per_hour, args.price_per_mtok,
                      args.bugs_per_month, args.bootstrap, args.seed)
    report = render(meta, stats, comps, len(ids), roi_res)
    (rd / "savings_report.md").write_text(report)
    result = {"n": len(ids), "instance_ids": ids, "stats": stats, "comparisons": comps, "roi": roi_res,
              "model": meta.get("model"), "model_sha256": meta.get("model_sha256")}
    (rd / "savings.json").write_text(json.dumps(result, indent=2))
    print(report)
    print(f"written: {rd / 'savings_report.md'} and {rd / 'savings.json'}")
    return result


if __name__ == "__main__":
    main()
