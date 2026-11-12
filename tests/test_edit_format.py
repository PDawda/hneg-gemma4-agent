import importlib.util
import sys

from conftest import ROOT

spec = importlib.util.spec_from_file_location("edit_format", ROOT / "experiments" / "edit_format.py")
ef = importlib.util.module_from_spec(spec)
sys.modules["edit_format"] = ef
spec.loader.exec_module(ef)

SRC = "import os\n\n\nclass A:\n    def f(self, x):\n        return x - 1\n\n    def g(self):\n        return 0\n"


def block(s, r):
    return f"```python\n<<<<<<< SEARCH\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n```"


def test_parse_and_apply_exact():
    blocks = ef.parse_edit_blocks("Sure!\n" + block("        return x - 1", "        return x + 1"))
    assert blocks == [("        return x - 1", "        return x + 1")]
    out, err = ef.apply_edits(SRC, blocks)
    assert err is None and "return x + 1" in out and out.count("\n") == SRC.count("\n")


def test_apply_tolerates_uniform_indent_offset():
    blocks = ef.parse_edit_blocks(block("def f(self, x):\n    return x - 1", "def f(self, x):\n    return x + 1"))
    out, err = ef.apply_edits(SRC, blocks)
    assert err is None
    assert "    def f(self, x):\n        return x + 1" in out


def test_apply_errors():
    assert ef.apply_edits(SRC, [])[1].startswith("no SEARCH/REPLACE")
    assert "not found" in ef.apply_edits(SRC, [("return y", "return z")])[1]
    assert "matches" in ef.apply_edits(SRC, [("return", "yield")])[1]
    assert "do not change" in ef.apply_edits(SRC, [("return 0", "return 0")])[1]
    assert "empty SEARCH" in ef.apply_edits(SRC, [("  ", "x")])[1]


def test_multiple_blocks_in_order():
    blocks = ef.parse_edit_blocks(block("return x - 1", "return x + 1") + "\n" + block("return 0", "return 42"))
    out, err = ef.apply_edits(SRC, blocks)
    assert err is None and "x + 1" in out and "return 42" in out


def test_context_small_file_is_full():
    shown, full = ef.build_code_context(SRC, "f is broken")
    assert full and shown == SRC


def test_context_large_file_selects_mentioned_definitions():
    filler = "".join(f"def unrelated_{i}():\n    return {i}\n\n\n" for i in range(400))
    src = ("import math\n\n\n" + filler + "class Matrix:\n    def other(self):\n        return 1\n\n"
           "    def det_bareiss(self):\n        return 0  # bug\n\n\n" + filler.replace("unrelated", "more"))
    shown, full = ef.build_code_context(src, "Matrix.det_bareiss returns 0 for det_bareiss of ...", max_chars=3000)
    assert not full and len(shown) < len(src)
    assert "import math" in shown and "def det_bareiss" in shown and "class Matrix:" in shown
    assert "unrelated_5" not in shown and "lines" in shown and "omitted" in shown
    # A SEARCH copied from the excerpt applies to the real file.
    out, err = ef.apply_edits(src, [("        return 0  # bug", "        return 1")])
    assert err is None and "return 1" in out


def test_parse_lenient_marker_lengths():
    text = "<<< SEARCH\n        return x - 1\n===\n        return x + 1\n>>> REPLACE"
    assert ef.parse_edit_blocks(text) == [("        return x - 1", "        return x + 1")]


def test_parse_unified_diff_fallback():
    diff = "```diff\n--- a/m.py\n+++ b/m.py\n@@ -5,2 +5,2 @@\n     def f(self, x):\n-        return x - 1\n+        return x + 1\n```"
    blocks = ef.parse_edit_blocks(diff)
    assert blocks == [("    def f(self, x):\n        return x - 1", "    def f(self, x):\n        return x + 1")]
    out, err = ef.apply_edits(SRC, blocks)
    assert err is None and "return x + 1" in out


def test_fuzzy_match_unique_region():
    src = SRC + "\n\ndef compute_total(items, tax_rate):\n    subtotal = sum(i.price for i in items)\n    return subtotal * tax_rate\n"
    search = "def compute_total(items, tax_rate):\n    subtotal = sum(i.price for i in item)\n    return subtotal * tax_rate"
    replace = "def compute_total(items, tax_rate):\n    subtotal = sum(i.price for i in items)\n    return subtotal * (1 + tax_rate)"
    out, err = ef.apply_edits(src, [(search, replace)])
    assert err is None and "(1 + tax_rate)" in out and "for i in item)" not in out


def test_fuzzy_match_refuses_short_or_unrelated_search():
    _, err = ef.apply_edits(SRC, [("return y", "return z")])
    assert "not found" in err
    _, err = ef.apply_edits(SRC, [("totally unrelated text that is long enough", "x = 1")])
    assert "not found" in err


def test_noop_block_skipped_but_all_noop_is_error():
    noop = ("        return 0", "        return 0")
    real = ("        return x - 1", "        return x + 1")
    out, err = ef.apply_edits(SRC, [noop, real])
    assert err is None and "return x + 1" in out
    _, err = ef.apply_edits(SRC, [noop])
    assert err == "the edits do not change the file"


def test_omission_markers_are_ignored_in_search():
    s = "    def f(self, x):\n# ... lines 7-9 omitted ...\n        return x - 1"
    out, err = ef.apply_edits(SRC, [(s, "    def f(self, x):\n        return x + 1")])
    assert err is None and "return x + 1" in out


def test_edit_feedback_shows_closest_lines():
    blocks = [("    def f(self, y):\n        return y - 100", "    def f(self, y):\n        return y")]
    _, err = ef.apply_edits(SRC, blocks)
    msg = ef.edit_feedback(SRC, blocks, err)
    assert "closest lines" in msg and "return x - 1" in msg
    assert ef.failed_block_index(err) == 0
    assert "SEARCH" in ef.edit_feedback(SRC, [], "no SEARCH/REPLACE edit blocks found in the reply")


def test_double_separator_block_is_rejected_not_applied():
    # Real Gemma 4 E4B reply (sympy-15308, full, turn 2): a second ======= inside REPLACE.
    reply = ("```python\n<<<<<<< SEARCH\n        return x - 1\n=======\n        return x + 1\n"
             "=======\n        return x - 1\n>>>>>>> REPLACE\n```")
    out, err = ef.apply_edits(SRC, ef.parse_edit_blocks(reply))
    assert out is None and "malformed" in err


DOC_SRC = ('def atoms(self):\n    """Return atoms.\n\n    Examples\n    ========\n\n    >>> 1\n    """\n'
           '    return set()\n')


def test_indented_docstring_underline_is_not_a_separator():
    # Real failure (sympy-18532): SEARCH text containing a docstring "========" underline.
    s = '    Examples\n    ========\n\n    >>> 1\n    """\n    return set()'
    r = '    Examples\n    ========\n\n    >>> 1\n    """\n    return {self}'
    reply = f"```python\n<<<<<<< SEARCH\n{s}\n=======\n{r}\n>>>>>>> REPLACE\n```"
    blocks = ef.parse_edit_blocks(reply)
    assert blocks == [(s, r)]
    out, err = ef.apply_edits(DOC_SRC, blocks)
    assert err is None and "return {self}" in out


def test_lenient_markers_still_need_column_zero():
    assert ef.parse_edit_blocks("  <<< SEARCH\nx\n  ===\ny\n  >>> REPLACE") == []
