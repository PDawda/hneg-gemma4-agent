# Results: Gemma 4 E4B, 20 SymPy bugs, code `411d9fd39892a989`

These are the result files behind every number in the project [README](../../README.md) and the paper.

| | |
| :-- | :-- |
| Model | `gemma-4-E4B-it-Q4_K_M.gguf` (unsloth), SHA-256 `85a896a0…cd1fab87` |
| Hardware | Kaggle T4, llama-cpp-python 0.3.36, greedy decoding |
| Tasks | 20 validated SWE-bench Lite SymPy instances, task-set hash `4be1f94b78ccc176` |
| Code | `411d9fd39892a989`; the exact source is in `code_snapshot_411d9fd39892a989.zip` |

## Files

| File | Contents |
| :-- | :-- |
| `run_meta.json` | Model hash, code fingerprint, task-set hash and every setting |
| `attempts_<config>.jsonl` | Every attempt: tokens, timing, test result, patch hash, model reply (up to 2,000 characters in this run) |
| `per_bug_<config>.csv`, `summary.csv` | Per-bug and per-configuration aggregates |
| `savings_report.md`, `savings.json` | Ablation table, paired bootstrap CIs, tail cost, ROI at an assumed $0.35/GPU-hour |
| `claims_verification.md`, `claims.json` | Verdict for each claim C1–C10 |
| `traces/` | Statechart event traces per bug (state transitions, attempt counter, stack depth) |
| `tasks*.jsonl`, `tasks_ids.txt` | The task set and its validation log |

## Regenerating the reports

The reports can be rebuilt from the per-bug and per-attempt files without a GPU:

```bash
cp -R results/E4B_411d9fd39892a989 /tmp/regen
python experiments/savings_report.py --results-dir /tmp/regen --gpu-cost-per-hour 0.35
python experiments/verify_claims.py  --results-dir /tmp/regen --gpu-cost-per-hour 0.35
```

Running these with the current code gives the same confidence intervals and claim verdicts as the files here (fixed bootstrap seed).

## Relation to the released code

The released code (`918612c5e74163c1`) adds fixes found by tracing this run: column-0 edit markers, both replay-guard escape routes masked, and identical retry wording for every configuration. Those fixes have **not** been re-measured, so running the released code will give numbers different from these.
