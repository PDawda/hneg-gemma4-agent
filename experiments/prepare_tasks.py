#!/usr/bin/env python3
"""Build the benchmark task file (JSONL) from SWE-bench Lite.

The harness in ``run_benchmark.py`` asks the model for the complete corrected
contents of one target file, so only instances whose gold patch edits exactly
one ``.py`` file are eligible. From those, ``--n`` instances (default 120) are
drawn with a fixed seed; the selected ids are written next to the task file so
the subset is reproducible.

For every selected instance the script

1. clones the repository once into ``--repos-dir/_mirrors`` and makes a cheap
   ``git clone --shared`` working copy per instance,
2. checks out ``base_commit`` and applies the instance's ``test_patch`` (the
   FAIL_TO_PASS tests live there),
3. derives a ``test_cmd`` that runs exactly the FAIL_TO_PASS tests.

Each repository needs its own Python environment with its dependencies at the
right versions (the SWE-bench Docker images are the easiest source). Use
``--env-prefix`` to wrap every test command, e.g.
``--env-prefix "conda run -n swe_{repo_slug}_{version}"``.

Usage
-----
python experiments/prepare_tasks.py --repos-dir /data/swe --out experiments/tasks.jsonl
python experiments/prepare_tasks.py --dataset-file swe_lite.jsonl --n 120 --seed 0 ...
python experiments/prepare_tasks.py --select-only --out experiments/tasks.jsonl   # no cloning
python experiments/prepare_tasks.py --repos sympy/sympy django/django --validate --n 20   # Kaggle pilot
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import run_tests  # noqa: E402
DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)
DJANGO_TEST = re.compile(r"^(\w+) \(([\w.]+)\)")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_instances(dataset: str, split: str, dataset_file: Optional[str]) -> List[Dict]:
    if dataset_file:
        text = Path(dataset_file).read_text(encoding="utf-8")
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("pip install datasets  (or pass --dataset-file)")
        rows = [dict(r) for r in load_dataset(dataset, split=split)]
    for r in rows:
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if isinstance(r.get(key), str):
                r[key] = json.loads(r[key])
    return rows


def patched_files(diff: str) -> List[str]:
    return sorted({b for _, b in DIFF_FILE.findall(diff)})


def eligible(inst: Dict) -> bool:
    files = patched_files(inst["patch"])
    return len(files) == 1 and files[0].endswith(".py") and bool(inst.get("FAIL_TO_PASS"))


def candidate_order(instances: List[Dict], seed: int, repos: Optional[Sequence[str]] = None) -> List[Dict]:
    """Eligible instances (optionally restricted to ``repos``) in a seeded random order."""
    pool = sorted((i for i in instances if eligible(i) and (not repos or i["repo"] in repos)),
                  key=lambda i: i["instance_id"])
    return random.Random(seed).sample(pool, len(pool))


def select(instances: List[Dict], n: int, seed: int, repos: Optional[Sequence[str]] = None) -> List[Dict]:
    order = candidate_order(instances, seed, repos)
    if len(order) < n:
        print(f"warning: only {len(order)} eligible instances (< {n}); using all", file=sys.stderr)
    return sorted(order[:n], key=lambda i: i["instance_id"])


# --------------------------------------------------------------------------- #
# Test commands
# --------------------------------------------------------------------------- #
# Make the checked-out package importable without installing it (src/ or flat layout).
PYTHONPATH_PREFIX = 'PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}" '
PYTHON = "python"   # interpreter used in test commands (--python)
SHELL_ERRORS = {124, 125, 126, 127}   # timeout, patch failed, not executable, not found


def build_test_cmd(inst: Dict) -> str:
    return PYTHONPATH_PREFIX + _bare_test_cmd(inst)


def _bare_test_cmd(inst: Dict) -> str:
    tests: List[str] = inst["FAIL_TO_PASS"]
    test_files = [f for f in patched_files(inst.get("test_patch", "")) if f.endswith(".py")]

    if inst["repo"] == "django/django":
        labels = []
        for t in tests:
            m = DJANGO_TEST.match(t.strip())
            if m:
                labels.append(f"{m.group(2)}.{m.group(1)}")
        if not labels:  # fall back to whole test modules touched by the test patch
            labels = sorted({f[len("tests/"):-3].replace("/", ".") for f in test_files if f.startswith("tests/")})
        return f"{PYTHON} tests/runtests.py --verbosity 2 --parallel 1 " + " ".join(shlex.quote(l) for l in labels)

    if all("::" in t for t in tests):
        return f"{PYTHON} -m pytest -q -p no:cacheprovider -rA " + " ".join(shlex.quote(t) for t in tests)

    # Bare test names (e.g. sympy): run the touched test files, selecting by name.
    names = sorted({re.split(r"[\[ (]", t.strip())[0] for t in tests})
    expr = " or ".join(names)
    return f"{PYTHON} -m pytest -q -p no:cacheprovider -rA {' '.join(map(shlex.quote, test_files))} -k {shlex.quote(expr)}"


# --------------------------------------------------------------------------- #
# Repositories
# --------------------------------------------------------------------------- #
def git(*args: str, cwd: Optional[Path] = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def checkout_instance(inst: Dict, repos_dir: Path) -> Path:
    slug = inst["repo"].replace("/", "__")
    mirror = repos_dir / "_mirrors" / slug
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {inst['repo']} ...", flush=True)
        git("clone", "--quiet", f"https://github.com/{inst['repo']}.git", str(mirror))
    dest = repos_dir / inst["instance_id"]
    if not dest.exists():
        git("clone", "--quiet", "--shared", "--no-checkout", str(mirror), str(dest))
    git("checkout", "--quiet", "--force", inst["base_commit"], cwd=dest)
    git("clean", "-fdq", cwd=dest)
    if inst.get("test_patch"):
        subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=dest, check=True,
                       input=inst["test_patch"], text=True)
    return dest


def with_env(cmd: str, inst: Dict, env_prefix: str) -> str:
    if not env_prefix:
        return cmd
    prefix = env_prefix.format(repo_slug=inst["repo"].replace("/", "__"),
                               version=inst.get("version", ""), instance_id=inst["instance_id"])
    return f"{prefix} {cmd}"


def validate(inst: Dict, repo_path: Path, env_prefix: str, timeout: int) -> Dict:
    """SWE-bench style check that this environment can evaluate the instance:
    FAIL_TO_PASS tests must fail on the base code and pass with the gold patch."""
    cmd = with_env(build_test_cmd(inst), inst, env_prefix)
    target = patched_files(inst["patch"])[0]
    base_exit, base_out, _ = run_tests(repo_path, target, None, cmd, timeout)
    gold_exit, gold_out, _ = run_tests(repo_path, target, None, cmd, timeout, patch=inst["patch"])
    kept = base_exit != 0 and base_exit not in SHELL_ERRORS and gold_exit == 0
    return {"instance_id": inst["instance_id"], "repo": inst["repo"], "base_exit": base_exit,
            "gold_exit": gold_exit, "kept": kept, "gold_output_tail": "" if kept else gold_out[-1500:]}


def to_task(inst: Dict, repo_path: Optional[Path], env_prefix: str) -> Dict:
    cmd = build_test_cmd(inst)
    if env_prefix:
        prefix = env_prefix.format(repo_slug=inst["repo"].replace("/", "__"),
                                   version=inst.get("version", ""), instance_id=inst["instance_id"])
        cmd = f"{prefix} {cmd}"
    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "base_commit": inst["base_commit"],
        "repo_path": str(repo_path) if repo_path else None,
        "target_file": patched_files(inst["patch"])[0],
        "issue": inst["problem_statement"],
        "test_cmd": cmd,
        "fail_to_pass": inst["FAIL_TO_PASS"],
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset-file", default=None, help="local JSON/JSONL export instead of the HF hub")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repos", nargs="+", default=None, help="only these repos, e.g. sympy/sympy django/django")
    ap.add_argument("--repos-dir", default=str(ROOT / "experiments" / "repos"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "tasks.jsonl"))
    ap.add_argument("--env-prefix", default="", help="template prefixed to every test_cmd; "
                    "fields: {repo_slug} {version} {instance_id}")
    ap.add_argument("--select-only", action="store_true", help="write tasks without cloning (repo_path=null)")
    ap.add_argument("--validate", action="store_true",
                    help="keep only instances whose tests fail on base and pass with the gold patch here; "
                         "draws further candidates until --n are kept")
    ap.add_argument("--test-timeout", type=int, default=300)
    ap.add_argument("--python", default="python", help="interpreter for test commands")
    ap.add_argument("--max-candidates", type=int, default=None, help="with --validate: stop after this many tries")
    args = ap.parse_args(list(argv) if argv is not None else None)
    global PYTHON
    PYTHON = args.python
    if args.select_only and args.validate:
        ap.error("--validate needs checkouts; drop --select-only")

    instances = load_instances(args.dataset, args.split, args.dataset_file)
    order = candidate_order(instances, args.seed, args.repos)
    if not args.validate:
        order = order[: args.n]
    elif args.max_candidates:
        order = order[: args.max_candidates]
    print(f"{len(order)} candidate instance(s){' from ' + ', '.join(args.repos) if args.repos else ''}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repos_dir = Path(args.repos_dir)
    val_log = out.with_name(out.stem + "_validation.jsonl")
    if args.validate:
        val_log.write_text("")

    kept: List[Dict] = []
    failures: List[str] = []
    for i, inst in enumerate(order, 1):
        if len(kept) >= args.n:
            break
        print(f"[{i}/{len(order)}] {inst['instance_id']}", flush=True)
        path = None
        if not args.select_only:
            try:
                path = checkout_instance(inst, repos_dir).resolve()
            except subprocess.CalledProcessError as exc:
                failures.append(inst["instance_id"])
                print(f"  skipped (checkout failed): {exc}", file=sys.stderr)
                continue
        if args.validate:
            v = validate(inst, path, args.env_prefix, args.test_timeout)
            with open(val_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(v) + "\n")
            print(f"  base_exit={v['base_exit']} gold_exit={v['gold_exit']} -> {'KEEP' if v['kept'] else 'drop'}", flush=True)
            if not v["kept"]:
                shutil.rmtree(path, ignore_errors=True)
                continue
        kept.append(to_task(inst, path, args.env_prefix))

    kept.sort(key=lambda t: t["instance_id"])
    with open(out, "w", encoding="utf-8") as fh:
        for t in kept:
            fh.write(json.dumps(t) + "\n")
    ids_file = out.with_name(out.stem + "_ids.txt")
    ids_file.write_text("".join(t["instance_id"] + "\n" for t in kept))
    print(f"\nwrote {len(kept)} task(s) to {out} (ids: {ids_file})")
    if args.validate:
        print(f"validation log: {val_log}")
    if len(kept) < args.n:
        print(f"warning: only {len(kept)} of the requested {args.n} tasks", file=sys.stderr)
    if failures:
        print(f"{len(failures)} instance(s) failed to check out: {failures}", file=sys.stderr)

if __name__ == "__main__":
    main()
