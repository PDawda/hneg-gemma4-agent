# Claim verification: gemma-4-E4B-it-Q4_K_M.gguf

n = **20** bugs completed by every configuration · model SHA-256 `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87` · code `411d9fd39892a989` · task-set `4be1f94b78ccc176`

✅ VERIFIED: 3 · ❌ NOT SUPPORTED: 3 · ⚠️ INCONCLUSIVE: 5

## Guarantees (mechanism; checkable at any n)

| # | Claim | Paper | Measured | Verdict | Why |
| :-- | :-- | :-: | :-- | :-: | :-- |
| C1 | Infinite-loop elimination (identical failed patch re-submitted until retries ran out) | 42.1% → 0.0% | full 10.0% · baseline_k3 60.0% · baseline 95.0% · reflex_only 100.0% · static_mask 95.0% | ❌ NOT SUPPORTED | full H-NEG still re-submitted an identical failed patch until retries ran out |
| C2 | Bounded convergence loop: at most K_max = 3 attempts, then GRACEFUL_HALT | k ≤ 3 | max attempts 3; halt rate 100.0%; max tokens/bug 27,093 vs baseline 137,311 | ✅ VERIFIED | hard cap held on every bug |
| C3 | Failed-patch exclusion: a failed patch is never re-submitted | trie pruning | full 3 repeated attempts · baseline_k3 20 · baseline 133 · reflex_only 142 · static_mask 135 | ❌ NOT SUPPORTED | 3 attempts re-submitted an earlier patch |
| C4 | 0.0% parse failure rate (syntactically valid patches) | 0.0% | syntax failures 11.4% of applied patches; edits that could not be applied 41.7% of attempts (not prevented by the mask; same for every config) | ❌ NOT SUPPORTED | some applied patches did not parse |
| C5 | No out-of-scope identifiers (CPG-scoped logit mask) | masked to −∞ | full 0.0% · baseline_k3 0.0% · baseline 0.0% · reflex_only 0.0% · static_mask 0.0% | ⚠️ INCONCLUSIVE | 0% for full, but unconstrained configurations did not invent names either |

## Effects (statistical; measured for this model, not the 31B in Table 6.3)

| # | Claim | Paper | Measured | Verdict | Why |
| :-- | :-- | :-: | :-- | :-: | :-- |
| C6 | Token reduction per bug vs unconstrained baseline | −64.2% | 82.2% [95% CI 79.6, 84.5] | ✅ VERIFIED | mostly the retry cap: see C8 |
| C7 | End-to-end speedup per bug vs unconstrained baseline | 3.93× | 3.87× [95% CI 3.10, 4.96] | ✅ VERIFIED | CI excludes 1× |
| C8 | Governance adds value beyond the retry cap (full vs baseline_k3: same K_max and context) | implied by Table 6.3 | Pass@1 0.0% vs 0.0% · tokens/resolved n/a vs n/a · tokens/bug 2.2% change | ⚠️ INCONCLUSIVE | neither configuration resolved a bug: the model, not the governance, is the bottleneck |
| C9 | Higher issue-resolution rate (Pass@1) than the unconstrained baseline | 19.2% → 31.8% | 0.0% (0/20) vs 5.0% (1/20) | ⚠️ INCONCLUSIVE | baseline ahead but not significant (p = 1, 0 vs 1) |
| C10a | Lower cost per resolved bug: full vs baseline | −64.2% tokens, 3.93× per resolved patch | n/a saving | ⚠️ INCONCLUSIVE | a configuration resolved nothing, so cost per resolved bug is undefined |
| C10b | Lower cost per resolved bug: full vs baseline_k3 | −64.2% tokens, 3.93× per resolved patch | n/a saving | ⚠️ INCONCLUSIVE | a configuration resolved nothing, so cost per resolved bug is undefined |

_Verdicts use 95% paired bootstrap CIs over bugs and exact McNemar tests on resolved/unresolved pairs. INCONCLUSIVE means the data cannot decide, not that the claim is false._
