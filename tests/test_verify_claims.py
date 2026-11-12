import csv
import importlib.util
import json
import sys

from conftest import ROOT

spec = importlib.util.spec_from_file_location("verify_claims", ROOT / "experiments" / "verify_claims.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["verify_claims"] = vc
spec.loader.exec_module(vc)

FIELDS = ["instance_id", "config", "resolved", "halted", "hit_turn_cap", "infinite_loop", "turns",
          "parse_failures", "total_tokens", "final_patch_tokens", "latency_s", "token_waste",
          "repeated_patches", "oos_attempts", "context_overflow"]


def write(d, config, rows, turns=3, loop=False):
    with open(d / f"per_bug_{config}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for iid, resolved, tokens, lat, rep, oos in rows:
            w.writerow({"instance_id": iid, "config": config, "resolved": resolved, "halted": config == "full" and not resolved,
                        "hit_turn_cap": not resolved, "infinite_loop": loop and not resolved, "turns": turns,
                        "parse_failures": 0, "total_tokens": tokens, "final_patch_tokens": 0, "latency_s": lat,
                        "token_waste": 1.0, "repeated_patches": rep, "oos_attempts": oos, "context_overflow": False})
    with open(d / f"attempts_{config}.jsonl", "w") as fh:
        for iid, *_ in rows:
            fh.write(json.dumps({"instance_id": iid, "attempts": [
                {"apply_error": None, "parse_ok": True}, {"apply_error": "edit block 1: SEARCH text not found", "parse_ok": False}]}) + "\n")


def test_mcnemar_exact():
    assert vc.mcnemar_exact(0, 0) == 1.0
    assert abs(vc.mcnemar_exact(6, 0) - 2 / 64) < 1e-12        # 2 * 0.5**6
    assert vc.mcnemar_exact(3, 3) == 1.0


def test_verdicts(tmp_path):
    ids = [f"b{i}" for i in range(12)]
    write(tmp_path, "baseline", [(i, False, 9000, 90.0, 2, 1) for i in ids], turns=10, loop=True)
    write(tmp_path, "baseline_k3", [(i, False, 3000, 30.0, 1, 0) for i in ids], loop=True)
    # full resolves 8 of 12 bugs that baseline_k3 misses -> significant
    write(tmp_path, "full", [(i, n < 8, 2500, 25.0, 0, 0) for n, i in enumerate(ids)])
    (tmp_path / "run_meta.json").write_text(json.dumps({"model": "m.gguf", "governance": {"max_convergence_retries": 3}}))

    res = vc.main(["--results-dir", str(tmp_path), "--bootstrap", "300", "--gpu-cost-per-hour", "0.35"])
    v = {c["id"]: c["verdict"] for c in res["claims"]}
    assert v["C1"] == vc.VERIFIED          # full 0% loops, baselines loop
    assert v["C2"] == vc.VERIFIED          # turns <= 3
    assert v["C3"] == vc.VERIFIED          # 0 repeats vs others > 0
    assert v["C4"] == vc.VERIFIED          # applied patches all parse
    assert v["C5"] == vc.VERIFIED          # baseline introduced oos names, full did not
    assert v["C6"] == vc.VERIFIED and v["C7"] == vc.VERIFIED
    assert v["C8"] == vc.VERIFIED          # 8 vs 0 discordant, p = 2/256
    assert v["C9"] == vc.VERIFIED
    assert v["C10a"] == vc.INCONCLUSIVE    # baseline resolved nothing -> undefined
    md = (tmp_path / "claims_verification.md").read_text()
    assert "VERIFIED" in md and "m.gguf" in md


def test_nothing_resolved_is_inconclusive_not_verified(tmp_path):
    ids = ["a", "b"]
    write(tmp_path, "baseline", [(i, False, 9000, 90.0, 2, 0) for i in ids], turns=10, loop=True)
    write(tmp_path, "baseline_k3", [(i, False, 3000, 5.0, 1, 0) for i in ids], loop=True)
    write(tmp_path, "full", [(i, False, 3200, 30.0, 0, 0) for i in ids])
    res = vc.evaluate(tmp_path, n_boot=200)
    v = {c["id"]: c["verdict"] for c in res["claims"]}
    assert v["C8"] == vc.INCONCLUSIVE and v["C9"] == vc.INCONCLUSIVE
    assert v["C5"] == vc.INCONCLUSIVE      # nobody invented names: nothing to prevent
