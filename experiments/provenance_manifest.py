"""Build or verify a SHA-256 provenance manifest for a results directory.

The manifest lists every file in the run's results directory with its size and
SHA-256, plus the run's key identifiers (model hash, code fingerprint, task-set
hash). Publishing this one small file permanently (e.g. on Arweave) commits to
the whole result bundle: changing any listed file changes its hash.

    # build  -> writes <results-dir>/provenance_manifest.json
    python experiments/provenance_manifest.py build  --results-dir results/E4B_411d9fd39892a989

    # verify a results directory against a manifest (local file or URL)
    python experiments/provenance_manifest.py verify --results-dir results/E4B_411d9fd39892a989 \
        --manifest https://arweave.net/<TX_ID>

    # print the SHA-256 of any file or URL (e.g. an Arweave download)
    python experiments/provenance_manifest.py hash https://arweave.net/<TX_ID>
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

MANIFEST_NAME = "provenance_manifest.json"
SCHEMA = "hneg-provenance/1"


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(source):
    """Bytes from a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "hneg-provenance"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    with open(source, "rb") as f:
        return f.read()


def _files(results_dir):
    out = []
    for root, _dirs, names in os.walk(results_dir):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, results_dir).replace(os.sep, "/")
            if rel == MANIFEST_NAME or n.startswith(".") or "__pycache__" in rel:
                continue
            out.append((rel, full))
    return sorted(out)


def build(results_dir):
    meta_path = os.path.join(results_dir, "run_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    files = [{"path": rel, "bytes": os.path.getsize(full), "sha256": _sha256_file(full)}
             for rel, full in _files(results_dir)]
    manifest = {
        "schema": SCHEMA,
        "project": "H-NEG (github.com/PDawda/hneg-gemma4-agent)",
        "run": os.path.basename(os.path.normpath(results_dir)),
        "run_created": meta.get("created"),
        "model": meta.get("model"),
        "model_sha256": meta.get("model_sha256"),
        "code_fingerprint": meta.get("code_sha256"),
        "task_set_hash": meta.get("task_ids_sha256"),
        "gpu": meta.get("gpu"),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "proves": "These exact result files existed when this manifest was published and have not changed since.",
        "does_not_prove": "That a particular machine executed the run; this is a record of published results, not an execution attestation.",
    }
    out = os.path.join(results_dir, MANIFEST_NAME)
    data = (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode()
    with open(out, "wb") as f:
        f.write(data)
    print("wrote %s  (%d files, %d bytes listed)" % (out, len(files), manifest["total_bytes"]))
    print("manifest sha256: %s" % _sha256_bytes(data))
    return 0


def verify(results_dir, manifest_src):
    manifest = json.loads(_read(manifest_src).decode())
    if manifest.get("schema") != SCHEMA:
        print("unexpected manifest schema: %r" % manifest.get("schema"))
        return 2
    ok = bad = missing = 0
    for entry in manifest["files"]:
        full = os.path.join(results_dir, entry["path"])
        if not os.path.exists(full):
            print("MISSING   %s" % entry["path"])
            missing += 1
        elif _sha256_file(full) != entry["sha256"]:
            print("MISMATCH  %s" % entry["path"])
            bad += 1
        else:
            ok += 1
    listed = {e["path"] for e in manifest["files"]}
    extra = [rel for rel, _ in _files(results_dir) if rel not in listed]
    for rel in extra:
        print("NOT LISTED %s" % rel)
    print("%d match, %d mismatch, %d missing, %d not listed" % (ok, bad, missing, len(extra)))
    print("model sha256 in manifest: %s" % manifest.get("model_sha256"))
    return 0 if (bad == 0 and missing == 0) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--results-dir", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--results-dir", required=True)
    v.add_argument("--manifest", required=True, help="local path or URL (e.g. https://arweave.net/<TX_ID>)")
    h = sub.add_parser("hash")
    h.add_argument("source", help="local path or URL")
    args = ap.parse_args()
    if args.cmd == "build":
        return build(args.results_dir)
    if args.cmd == "verify":
        return verify(args.results_dir, args.manifest)
    if args.cmd == "hash":
        print(_sha256_bytes(_read(args.source)))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
