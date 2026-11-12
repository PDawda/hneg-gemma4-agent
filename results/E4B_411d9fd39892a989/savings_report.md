# H-NEG ablation: gemma-4-E4B-it-Q4_K_M.gguf

Bugs completed by every configuration: **n = 20**  
Model SHA-256: `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`  
Code fingerprint: `411d9fd39892a989` · task-set: `4be1f94b78ccc176` · GPU: Tesla T4 · K_max = 3

| Configuration | PFR | Loop rate | Halt rate | Context overflow | Tokens / bug | Latency / bug | Pass@1 | Tokens / resolved | Latency / resolved |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Baseline (unconstrained, up to --max-turns) | 50.5% | 95.0% | 0.0% | 0.0% | 73,282 | 219.3 s | 5.0% (1/20) | 1,465,631 | 4,385.6 s |
| Baseline + K_max retry cap (no masking) | 51.7% | 60.0% | 0.0% | 0.0% | 12,747 | 44.0 s | 0.0% (0/20) | n/a | n/a |
| + Loop 1 reflex only | 48.0% | 100.0% | 0.0% | 0.0% | 44,485 | 126.6 s | 0.0% (0/20) | n/a | n/a |
| + Identifier-scope logit masking | 46.5% | 95.0% | 0.0% | 0.0% | 44,675 | 152.1 s | 0.0% (0/20) | n/a | n/a |
| Full H-NEG (CPG + PDA + mask + anti-replay) | 48.3% | 10.0% | 100.0% | 0.0% | 13,030 | 56.7 s | 0.0% (0/20) | n/a | n/a |

## Paired comparisons (95% bootstrap CI over bugs)

| Comparison | Token reduction | Speedup | Δ Pass@1 (pp) |
| :--- | :-: | :-: | :-: |
| full vs baseline | 82.2% [79.6, 84.5] | 3.87× [3.10, 4.96] | -5.0 [-15.0, 0.0] |
| full vs baseline_k3 | -2.2% [-5.8, 0.4] | 0.78× [0.64, 0.89] | 0.0 [0.0, 0.0] |
| baseline_k3 vs baseline | 82.6% [80.1, 84.8] | 4.98× [3.76, 6.86] | -5.0 [-15.0, 0.0] |

## Tail cost and guarantees

| Configuration | p95 tokens / bug | Max tokens / bug | p95 latency | Max latency | Episodes re-submitting a failed patch | Attempts introducing out-of-scope names |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| Baseline (unconstrained, up to --max-turns) | 121,346 | 137,311 | 330.8 s | 357.8 s | 95.0% (133 attempts) | 0.0% |
| Baseline + K_max retry cap (no masking) | 24,647 | 27,093 | 107.8 s | 173.0 s | 60.0% (20 attempts) | 0.0% |
| + Loop 1 reflex only | 83,841 | 95,519 | 245.0 s | 259.3 s | 100.0% (142 attempts) | 0.0% |
| + Identifier-scope logit masking | 83,840 | 95,519 | 263.8 s | 289.3 s | 95.0% (135 attempts) | 0.0% |
| Full H-NEG (CPG + PDA + mask + anti-replay) | 25,858 | 27,093 | 108.0 s | 178.2 s | 10.0% (3 attempts) | 0.0% |

## ROI (GPU $0.35/h; projection for 1,000 bugs/month)

| Configuration | Cost / bug | Cost / resolved bug | Monthly cost | Bugs fixed / month |
| :--- | :-: | :-: | :-: | :-: |
| Baseline (unconstrained, up to --max-turns) | $0.0213 | $0.4264 | $21.32 | 50 |
| Baseline + K_max retry cap (no masking) | $0.0043 | n/a (none resolved) | $4.28 | 0 |
| + Loop 1 reflex only | $0.0123 | n/a (none resolved) | $12.31 | 0 |
| + Identifier-scope logit masking | $0.0148 | n/a (none resolved) | $14.79 | 0 |
| Full H-NEG (CPG + PDA + mask + anti-replay) | $0.0055 | n/a (none resolved) | $5.51 | 0 |

| Comparison | Saving in cost per resolved bug (95% CI) |
| :--- | :-: |
| full vs baseline | n/a |
| full vs baseline_k3 | n/a |
| baseline_k3 vs baseline | n/a |

_Monthly figures extrapolate this sample's per-bug cost and fix rate; prices are inputs, not measurements._

_Note: n = 20 is small; read the confidence intervals, not just the point estimates._
