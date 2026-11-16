H-NEG benchmark: permanent provenance record
=============================================

Project:  H-NEG (Hierarchical Neuro-Symbolic Execution Governance)
Code:     https://github.com/PDawda/hneg-gemma4-agent
Run:      E4B_411d9fd39892a989  (created 2026-11-12 03:19:26 UTC)
Model:    gemma-4-E4B-it-Q4_K_M.gguf
          SHA-256 85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87
Tasks:    20 SWE-bench Lite SymPy tasks (task-set hash 4be1f94b78ccc176)
Hardware: 1 x Tesla T4 (Kaggle)

Files in this folder
--------------------
provenance_manifest.json  SHA-256 fingerprints of all 41 result files in the
                          repository folder results/E4B_411d9fd39892a989/.
                          This one file anchors the whole result bundle.
                          SHA-256 87c6c22d459c384e36d4f406ecabb81f634853f19a9893df73651bbe2d7b17cc
run_meta.json             Run configuration, environment and model hash.
                          SHA-256 85ccd9fc605fc9807e95131bc79fe080d43507f9a409af3e3ac51b688a62243d
claims_verification.md    Automatic verdict on every claim (verified /
                          not supported / inconclusive).
                          SHA-256 c274c84dd9c50374fc7587676fcbafaa691acd57226ed42b2863cd404fe00c0a

How to verify
-------------
  git clone https://github.com/PDawda/hneg-gemma4-agent.git
  cd hneg-gemma4-agent
  python experiments/provenance_manifest.py verify \
      --results-dir results/E4B_411d9fd39892a989 \
      --manifest https://arweave.net/<MANIFEST_DATA_TX_ID>
  # expected: 41 match, 0 mismatch, 0 missing

What this proves: these exact result files existed when this record was
published and have not changed since.
What it does not prove: which machine executed the run (that needs a signed
execution attestation; planned future work).
