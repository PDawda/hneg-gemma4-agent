import csv
import importlib.util
import json
import sys

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("savings_report", ROOT / "experiments" / "savings_report.py")
sr = importlib.util.module_from_spec(spec)
sys.modules["savings_report"] = sr
spec.loader.exec_module(sr)

FIELDS = ["instance_id", "config", "resolved", "halted", "hit_turn_cap", "infinite_loop", "turns",
          "parse_failures", "total_tokens", "final_patch_tokens", "latency_s", "token_waste",
          "repeated_patches", "oos_attempts"]


def write(d, config, rows):
    with open(d / f"per_bug_{config}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for iid, resolved, tokens, lat, *extra in rows:
            rp, oos = extra if extra else (0, 0)
            w.writerow({"instance_id": iid, "config": config, "resolved": resolved, "halted": False,
                        "hit_turn_cap": not resolved, "infinite_loop": False, "turns": 2, "parse_failures": 0,
                        "total_tokens": tokens, "final_patch_tokens": 0, "latency_s": lat, "token_waste": 1.0,
                        "repeated_patches": rp, "oos_attempts": oos})


def test_report_numbers(tmp_path):
    write(tmp_path, "baseline", [("a", True, 1000, 100.0, 0, 0), ("b", False, 3000, 300.0, 1, 1),
                                 ("c", False, 4000, 400.0, 2, 1)])
    write(tmp_path, "baseline_k3", [("a", True, 800, 80.0), ("b", False, 1200, 120.0), ("c", False, 1000, 100.0)])
    write(tmp_path, "full", [("a", True, 500, 50.0), ("b", True, 700, 70.0), ("c", False, 800, 80.0),
                             ("extra", True, 1, 1.0)])          # not run by the others -> excluded
    (tmp_path / "run_meta.json").write_text(json.dumps({"model": "gemma-4-E2B-it-Q4_K_M.gguf", "model_sha256": "abc"}))

    # $3.60/h -> $0.001 per second of latency; no token price
    res = sr.main(["--results-dir", str(tmp_path), "--bootstrap", "500", "--gpu-cost-per-hour", "3.6",
                   "--bugs-per-month", "1000"])
    assert res["n"] == 3 and res["instance_ids"] == ["a", "b", "c"]
    full_vs_base = next(c for c in res["comparisons"] if (c["a"], c["b"]) == ("full", "baseline"))
    assert full_vs_base["token_reduction_pct"] == pytest.approx(75.0)          # 1 - 2000/8000
    assert full_vs_base["speedup_x"] == pytest.approx(4.0)                     # 800 s / 200 s
    assert full_vs_base["delta_pass_at_1_pp"] == pytest.approx(100 / 3)        # 2/3 - 1/3
    lo, hi = full_vs_base["token_reduction_ci"]
    assert lo <= 75.0 <= hi
    k3 = next(c for c in res["comparisons"] if (c["a"], c["b"]) == ("full", "baseline_k3"))
    assert k3["token_reduction_pct"] == pytest.approx(100 * (1 - 2000 / 3000))
    assert res["stats"]["full"]["tokens_per_resolved"] == pytest.approx(1000.0)
    md = (tmp_path / "savings_report.md").read_text()
    assert "n = 3" in md and "abc" in md and "full vs baseline" in md

    b = res["stats"]["baseline"]
    assert b["p95_tokens"] == 4000 and b["max_latency_s"] == 400.0
    assert b["repeat_episode_pct"] == pytest.approx(200 / 3) and b["repeated_attempts"] == 3
    assert b["oos_attempt_pct"] == pytest.approx(100 * 2 / 6)          # 2 of 6 attempts
    assert res["stats"]["full"]["repeat_episode_pct"] == 0.0

    roi = res["roi"]
    assert roi["per_config"]["baseline"]["cost_per_resolved"] == pytest.approx(0.8)   # $0.80 / 1 fixed
    assert roi["per_config"]["full"]["cost_per_resolved"] == pytest.approx(0.1)       # $0.20 / 2 fixed
    assert roi["per_config"]["baseline"]["monthly_cost"] == pytest.approx(800 / 3)
    assert roi["per_config"]["full"]["monthly_resolved"] == pytest.approx(2000 / 3)
    sv = next(x for x in roi["savings"] if (x["a"], x["b"]) == ("full", "baseline"))
    assert sv["cost_per_resolved_saving_pct"] == pytest.approx(87.5)
    assert "## ROI" in md and "Tail cost and guarantees" in md
