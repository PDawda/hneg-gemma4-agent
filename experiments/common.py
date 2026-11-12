"""Helpers shared by the experiment scripts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def run_tests(repo: Path, target: str, new_source: Optional[str], test_cmd: str, timeout: int,
              patch: Optional[str] = None) -> Tuple[int, str, Optional[str]]:
    """Run ``test_cmd`` in a throwaway copy of ``repo``.

    Either ``new_source`` replaces ``target`` or, if ``patch`` is given, the
    unified diff is applied with ``git apply``. Returns
    ``(exit_code, tail_of_output, final_target_source)``; exit code 124 means
    timeout and 125 means the patch did not apply.
    """
    with tempfile.TemporaryDirectory(prefix="hneg_") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(repo, work, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        if patch is not None:
            proc = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=work, input=patch,
                                  text=True, capture_output=True)
            if proc.returncode != 0:
                return 125, proc.stderr[-2000:], None
        elif new_source is not None:
            (work / target).write_text(new_source, encoding="utf-8")
        final = (work / target).read_text(encoding="utf-8") if (work / target).exists() else None
        try:
            proc = subprocess.run(test_cmd, shell=True, cwd=work, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s", final
        return proc.returncode, (proc.stdout + proc.stderr)[-4000:], final
