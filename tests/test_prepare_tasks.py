import importlib.util
import json
import sys
import subprocess

from conftest import ROOT

spec = importlib.util.spec_from_file_location("prepare_tasks", ROOT / "experiments" / "prepare_tasks.py")
pt = importlib.util.module_from_spec(spec)
sys.modules["prepare_tasks"] = pt
spec.loader.exec_module(pt)


def diff(path):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"


def inst(iid, repo="psf/requests", patch_file="requests/models.py", f2p=("tests/test_x.py::test_a",), test_file="tests/test_x.py"):
    return {"instance_id": iid, "repo": repo, "base_commit": "abc", "patch": diff(patch_file),
            "test_patch": diff(test_file), "problem_statement": "bug", "FAIL_TO_PASS": list(f2p), "version": "2.0"}


def test_eligibility_and_selection():
    rows = [inst(f"r-{i}") for i in range(10)] + [inst("multi", patch_file="a.py") | {"patch": diff("a.py") + diff("b.py")},
                                                  inst("nonpy", patch_file="README.rst")]
    assert not pt.eligible(rows[-1]) and not pt.eligible(rows[-2])
    a = [i["instance_id"] for i in pt.select(rows, 5, seed=0)]
    assert [i["repo"] for i in pt.candidate_order(rows, 0, repos=["other/repo"])] == []
    assert a == [i["instance_id"] for i in pt.select(list(reversed(rows)), 5, seed=0)]   # order independent
    assert len(a) == 5 and "multi" not in a


def test_test_commands():
    cmd = pt.build_test_cmd(inst("p"))
    assert cmd.startswith(pt.PYTHONPATH_PREFIX)
    assert cmd.endswith("python -m pytest -q -p no:cacheprovider -rA tests/test_x.py::test_a")
    dj = inst("d", repo="django/django", f2p=["test_trailing_newline (auth_tests.test_validators.UsernameValidatorsTests)"],
              test_file="tests/auth_tests/test_validators.py")
    assert pt.build_test_cmd(dj).endswith("auth_tests.test_validators.UsernameValidatorsTests.test_trailing_newline")
    sy = inst("s", repo="sympy/sympy", f2p=["test_foo", "test_bar"], test_file="sympy/core/tests/test_basic.py")
    assert pt.build_test_cmd(sy).endswith("sympy/core/tests/test_basic.py -k 'test_bar or test_foo'")


def test_to_task_with_env_prefix(tmp_path):
    t = pt.to_task(inst("p"), None, "conda run -n swe_{repo_slug}_{version}")
    assert t["test_cmd"].startswith("conda run -n swe_psf__requests_2.0 PYTHONPATH=")
    assert t["target_file"] == "requests/models.py" and t["repo_path"] is None


def test_main_select_only(tmp_path):
    data = tmp_path / "lite.jsonl"
    data.write_text("\n".join(json.dumps(inst(f"r-{i}") | {"FAIL_TO_PASS": json.dumps(["t::a"])}) for i in range(4)))
    out = tmp_path / "tasks.jsonl"
    pt.main(["--dataset-file", str(data), "--n", "3", "--select-only", "--out", str(out)])
    tasks = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(tasks) == 3 and tasks[0]["fail_to_pass"] == ["t::a"]
    assert (tmp_path / "tasks_ids.txt").read_text().count("\n") == 3


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def test_validate_with_real_repo(tmp_path):
    """Base code fails the FAIL_TO_PASS test, gold patch passes -> kept; a bogus gold patch -> dropped."""
    repo = tmp_path / "upstream"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "mathy.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_mathy.py").write_text("from pkg.mathy import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    gold = ("diff --git a/pkg/mathy.py b/pkg/mathy.py\n--- a/pkg/mathy.py\n+++ b/pkg/mathy.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n")
    rec = {"instance_id": "toy-1", "repo": "toy/toy", "patch": gold, "test_patch": "",
           "FAIL_TO_PASS": ["tests/test_mathy.py::test_add"]}
    pt.PYTHON = sys.executable
    v = pt.validate(rec, repo, "", timeout=120)
    assert v["kept"] and v["base_exit"] == 1 and v["gold_exit"] == 0, v

    bogus = gold.replace("a + b", "a * b")
    v2 = pt.validate({**rec, "patch": bogus}, repo, "", timeout=120)
    assert not v2["kept"] and v2["gold_exit"] == 1
    pt.PYTHON = "definitely-not-a-python"
    assert not pt.validate(rec, repo, "", timeout=60)["kept"]      # 127 is not a test failure
    pt.PYTHON = "python"
