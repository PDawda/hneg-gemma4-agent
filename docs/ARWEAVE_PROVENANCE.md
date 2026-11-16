# Arweave Provenance Report: H-NEG run `E4B_411d9fd39892a989`

This page records what was published permanently on Arweave for the H-NEG benchmark, how anyone can verify it, and what it does and does not prove.

## 1. What is anchored

One small file, **`provenance_manifest.json`** (7 KB), lists the SHA-256 fingerprint of **all 41 files** in the run's results folder. Publishing that one file permanently commits to the whole bundle: if any result file were later changed, its fingerprint would no longer match the permanent record.

| Identifier | Value |
| :-- | :-- |
| Run | `E4B_411d9fd39892a989` (created 2026-11-12 03:19:26 UTC) |
| Model | `gemma-4-E4B-it-Q4_K_M.gguf` (from `unsloth/gemma-4-E4B-it-GGUF`) |
| Model SHA-256 | `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87` |
| Code fingerprint | `411d9fd39892a989` |
| Task-set hash | `4be1f94b78ccc176` (20 SWE-bench Lite SymPy tasks) |
| Hardware | 1 × Tesla T4 (Kaggle) |
| Files covered | 41 files, 1,178,193 bytes |

### Key files and their fingerprints

| File | SHA-256 | Arweave TX ID |
| :-- | :-- | :-- |
| `provenance_manifest.json` | `87c6c22d459c384e36d4f406ecabb81f634853f19a9893df73651bbe2d7b17cc` | [`MFdCqNlYgqY4xuuC44-YjidfTtARapnfuOx-Nguc7H4`](https://arweave.net/MFdCqNlYgqY4xuuC44-YjidfTtARapnfuOx-Nguc7H4) |
| `run_meta.json` | `85ccd9fc605fc9807e95131bc79fe080d43507f9a409af3e3ac51b688a62243d` | [`hdrDkLpT5l1P8fVMoE5ug3cYEYmG-SlhPgT_RMwFsKs`](https://arweave.net/hdrDkLpT5l1P8fVMoE5ug3cYEYmG-SlhPgT_RMwFsKs) |
| `claims_verification.md` | `c274c84dd9c50374fc7587676fcbafaa691acd57226ed42b2863cd404fe00c0a` | [`CfJIW5M8IIOMudYls3Vn_6Rcr_7qsQOw3bCIP8WUSUs`](https://arweave.net/CfJIW5M8IIOMudYls3Vn_6Rcr_7qsQOw3bCIP8WUSUs) |
| `summary.csv` | `a0a77f6224c6c658b64671169bee9e4ef01b0ba9658dc9a9143af8154511315e` | covered by the manifest |
| `code_snapshot_411d9fd39892a989.zip` | `620e31bdb035a330b8065e9d608e10f8cad85cd8a3fd849545f4c7230acdc9d8` | covered by the manifest |

### Verification log

| Date | Check | Result |
| :-- | :-- | :-- |
| 2026-11-16 | SHA-256 of each of the 3 files as served by Arweave (Turbo gateway) vs. the repo | ✅ 3 / 3 match |
| 2026-11-16 | Live Arweave manifest vs. every file in `results/E4B_411d9fd39892a989/` | ✅ 41 match, 0 mismatch, 0 missing |

*Uploaded via ArDrive Turbo to a public drive on 2026-11-16. New uploads are served by `turbo-gateway.com` within minutes and by `arweave.net` once propagated. Both serve identical bytes.*

The full list of 41 fingerprints is in [`results/E4B_411d9fd39892a989/provenance_manifest.json`](../results/E4B_411d9fd39892a989/provenance_manifest.json).

> ⚠️ **Do not edit anything in `results/E4B_411d9fd39892a989/` after uploading.** Any change breaks the match with the permanent record. That is the point of the record, so make sure the folder is final first.

---

## 2. How to upload (about 10 minutes, author only)

Arweave uploads are **permanent and public**. Only upload files you're happy to have public forever. These files contain only public SymPy benchmark data, model and code hashes, and Kaggle environment details: no personal data and no private code.

1. **Commit and push** `experiments/provenance_manifest.py` and `results/E4B_411d9fd39892a989/provenance_manifest.json` to GitHub first, so the repo and the permanent record match.
2. Go to **[app.ardrive.io](https://app.ardrive.io)** and sign in. You can create a wallet there.
3. Create a **public** drive, e.g. `hneg-provenance`. It must be *public*: private drives are encrypted, and nobody could verify them.
4. Upload these three files from `results/E4B_411d9fd39892a989/`:
   - `provenance_manifest.json` (**required**; this alone anchors everything)
   - `run_meta.json`
   - `claims_verification.md`

   All three are under 8 KB. ArDrive's Turbo service has offered free uploads for small files; check the current limit on the upload screen. Otherwise the cost is a fraction of a cent.
5. For each file, open its details in ArDrive and copy the **Data TX ID** (not the File ID).
6. Wait until `https://arweave.net/<DATA_TX_ID>` shows the file. This can take from a few minutes to about an hour.
7. Paste the TX IDs into the table above and into the article (section 4).

---

## 3. How anyone can verify it

**Quick check: one file**

```bash
curl -sL https://arweave.net/MFdCqNlYgqY4xuuC44-YjidfTtARapnfuOx-Nguc7H4 | shasum -a 256
# expected: 87c6c22d459c384e36d4f406ecabb81f634853f19a9893df73651bbe2d7b17cc
```

**Full check: the permanent manifest against every result file in the repo**

```bash
git clone https://github.com/PDawda/hneg-gemma4-agent.git
cd hneg-gemma4-agent
python experiments/provenance_manifest.py verify \
  --results-dir results/E4B_411d9fd39892a989 \
  --manifest https://arweave.net/MFdCqNlYgqY4xuuC44-YjidfTtARapnfuOx-Nguc7H4
# expected: 41 match, 0 mismatch, 0 missing
```

**Recompute the results themselves (no GPU)**

```bash
cp -R results/E4B_411d9fd39892a989 /tmp/regen
python experiments/savings_report.py --results-dir /tmp/regen --gpu-cost-per-hour 0.35
python experiments/verify_claims.py  --results-dir /tmp/regen --gpu-cost-per-hour 0.35
```

Together these show three things:
1. the published results are the ones in the repo;
2. they haven't changed since the publication date on Arweave;
3. the tables and claim verdicts follow from the raw attempt logs.

---

## 4. Text for the article

> **Permanent provenance on Arweave.** The full result bundle (41 files) is anchored by a single SHA-256 manifest stored permanently on Arweave:
> - Provenance manifest: https://arweave.net/MFdCqNlYgqY4xuuC44-YjidfTtARapnfuOx-Nguc7H4 (SHA-256 `87c6c22d…7b17cc`)
> - Run metadata: https://arweave.net/hdrDkLpT5l1P8fVMoE5ug3cYEYmG-SlhPgT_RMwFsKs
> - Claim verification: https://arweave.net/CfJIW5M8IIOMudYls3Vn_6Rcr_7qsQOw3bCIP8WUSUs
>
> Anyone can check every result file in the repository against it with one command (`python experiments/provenance_manifest.py verify …`). This proves the results haven't changed since publication. It does not prove which machine executed the run; that needs signed runner attestations, which are next on the roadmap.

---

## 5. What this does and does not prove

| ✅ Proves | ❌ Does not prove |
| :-- | :-- |
| These exact 41 result files existed when the manifest was published | That a specific GPU actually executed the run |
| None of them has changed since | That the run was performed honestly; only that its outputs are fixed |
| Which exact model file was used (by SHA-256) | That the model behaves the same on other hardware |
| The tables and verdicts can be recomputed from the raw logs | Anything about runs not listed in the manifest |

Closing the right-hand column needs **execution attestation**: a signed receipt from the machine that ran the job (for example, a Nosana job ID plus a hardware-signed attestation, bound to the manifest's hash). That is planned future work.
