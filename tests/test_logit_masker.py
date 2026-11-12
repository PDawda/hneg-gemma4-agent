import math

import pytest
import torch

from conftest import FakeTokenizer
from src.logit_masker import (
    FailedDiffTrie,
    HNEGLogitsProcessor,
    IdentifierScopeGate,
    _names_bound_in,
    filter_draft_tokens,
)

PIECES = [
    "```", "python", "\n", "▁", "▁▁▁▁", "(", ")", ":", ",", ".", "▁=", "▁==",
    "▁return", "▁def", "▁verify", "_token", "▁expected", "▁sign", "sig", "n",
    "▁secure", "_compare", "▁secure_compare", "(secure", "▁hmac", "compare_digest",
    "▁tmp", "▁newvar", "▁payload", "x", "▁sign(", "▁bogus(", "▁\"", "secret",
]
ALLOW = {"verify_token", "expected", "sign", "hmac", "payload", "signature"}


@pytest.fixture
def tok():
    return FakeTokenizer(PIECES)


@pytest.fixture
def gate(tok):
    return IdentifierScopeGate(tok, ALLOW)


def fenced(code):
    return "```python\n" + code


# --------------------------------------------------------------------------- #
# Trie
# --------------------------------------------------------------------------- #
def test_trie_blocks_continuation_after_min_prefix():
    t = FailedDiffTrie(min_prefix_len=2)
    t.insert([1, 2, 3, 4])
    t.insert([1, 2, 5])
    assert t.blocked_next([1]) == set()               # below min_prefix_len
    assert t.blocked_next([1, 2]) == {3, 5}
    assert t.blocked_next([1, 2, 3]) == {4}
    assert t.blocked_next([9, 9]) == set()
    assert t.contains([1, 2, 5]) and not t.contains([1, 2])
    assert len(t) == 2
    t.clear()
    assert len(t) == 0 and t.blocked_next([1, 2]) == set()


def test_trie_rejects_negative_prefix():
    with pytest.raises(ValueError):
        FailedDiffTrie(-1)


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
def test_processor_masks_failed_prefix(tok):
    trie = FailedDiffTrie(min_prefix_len=2)
    trie.insert([10, 11, 12])
    proc = HNEGLogitsProcessor(tok, failed_trie=trie)
    out = proc(torch.tensor([[10, 11]]), torch.zeros(1, 40))
    assert out[0, 12].item() == -math.inf
    assert out[0, 13].item() == 0.0
    assert proc.stats["trie_masked"] == 1


def test_processor_respects_patch_start(tok):
    trie = FailedDiffTrie(min_prefix_len=1)
    trie.insert([10, 11])
    proc = HNEGLogitsProcessor(tok, failed_trie=trie, patch_start=2)
    out = proc(torch.tensor([[5, 6, 10]]), torch.zeros(1, 40))   # prompt = [5, 6]
    assert out[0, 11].item() == -math.inf


def test_processor_never_returns_all_inf(tok):
    class BlockAll:
        def allowed_token_ids(self, text):
            return set()

    proc = HNEGLogitsProcessor(tok, grammar_gate=BlockAll())
    out = proc(torch.tensor([[3]]), torch.zeros(1, 40))
    assert torch.isfinite(out).sum().item() == 1
    assert out[0, tok.eos_token_id].item() == 0.0
    assert proc.stats["fallbacks"] == 1


def test_processor_prefers_grammar_over_trie_on_conflict(tok):
    class OnlyTwelve:
        def allowed_token_ids(self, text):
            return {12}

    trie = FailedDiffTrie(min_prefix_len=1)
    trie.insert([10, 12])
    proc = HNEGLogitsProcessor(tok, failed_trie=trie, grammar_gate=OnlyTwelve())
    out = proc(torch.tensor([[10]]), torch.zeros(1, 40))
    assert out[0, 12].item() == 0.0 and torch.isfinite(out).sum().item() == 1


def test_softcap(tok):
    proc = HNEGLogitsProcessor(tok, softcap=30.0)
    out = proc(torch.tensor([[3]]), torch.full((1, 40), 1000.0))
    assert out.max().item() <= 30.0


def test_filter_draft_tokens(tok):
    trie = FailedDiffTrie(min_prefix_len=1)
    trie.insert([10, 11, 12])
    proc = HNEGLogitsProcessor(tok, failed_trie=trie)
    assert filter_draft_tokens(proc, torch.tensor([10]), [13, 14], 40) == [13, 14]
    assert filter_draft_tokens(proc, torch.tensor([10]), [11, 12], 40) == []


# --------------------------------------------------------------------------- #
# Identifier scope gate
# --------------------------------------------------------------------------- #
def test_names_bound_in():
    code = "def f(a, b: int = 2, *args):\n    x, y = 1, 2\n    for i in r:\n        pass\n    import os\n    with o as h:\n"
    assert {"f", "a", "b", "args", "x", "y", "i", "os", "h"} <= _names_bound_in(code)


def test_gate_unconstrained_outside_fence(gate, tok):
    assert gate.allowed_token_ids("Here is the fix: secure") is None
    assert gate.blocked_token_ids("Here is the fix: ") is None


def test_gate_blocks_out_of_scope_identifier_start(gate, tok):
    blocked = set(gate.blocked_token_ids(fenced("def f(payload):\n    return")).tolist())
    assert tok.vocab["▁secure"] in blocked
    assert tok.vocab["▁secure_compare"] in blocked
    assert tok.vocab["▁bogus("] in blocked
    assert tok.vocab["▁sign"] not in blocked
    assert tok.vocab["▁sign("] not in blocked
    assert tok.vocab["▁expected"] not in blocked


def test_gate_blocks_after_open_paren(gate, tok):
    blocked = set(gate.blocked_token_ids(fenced("    return sign")).tolist())
    assert tok.vocab["(secure"] in blocked


def test_gate_inside_identifier(gate, tok):
    allowed = gate.allowed_token_ids(fenced("    return sig"))
    assert tok.vocab["n"] in allowed            # -> "sign"
    assert tok.vocab["x"] not in allowed        # -> "sigx"
    assert tok.vocab["("] not in allowed        # "sig" is not a complete name
    allowed = gate.allowed_token_ids(fenced("    return sign"))
    assert tok.vocab["("] in allowed


def test_gate_allows_attributes_of_external_modules(gate, tok):
    code = fenced("    return hmac.")
    assert gate.allowed_token_ids(code) is None
    blocked = gate.blocked_token_ids(code)
    assert blocked is None or tok.vocab["compare_digest"] not in set(blocked.tolist())
    assert gate.allowed_token_ids(fenced("    return hmac.compare_digest")) is None


def test_gate_allows_names_bound_by_patch(gate, tok):
    code = fenced("    tmp = 1\n    return")
    assert tok.vocab["▁tmp"] not in set(gate.blocked_token_ids(code).tolist())
    assert tok.vocab["▁newvar"] in set(gate.blocked_token_ids(code).tolist())


def test_gate_allows_new_names_where_binding_is_legal(gate, tok):
    for code in ("    ", "    for", "    def", "def f(payload,", "    import", "    x = [", "    except E as"):
        b = gate.blocked_token_ids(fenced(code))
        assert b is None or tok.vocab["▁newvar"] not in set(b.tolist()), code


def test_gate_ignores_strings_and_comments(gate, tok):
    assert gate.blocked_token_ids(fenced('    msg = "secure')) is None
    assert gate.allowed_token_ids(fenced("    # secure")) is None


def test_gate_runtime_allow(gate, tok):
    code = fenced("    return")
    assert tok.vocab["▁secure_compare"] in set(gate.blocked_token_ids(code).tolist())
    gate.allow("secure_compare")
    assert tok.vocab["▁secure_compare"] not in set(gate.blocked_token_ids(code).tolist())


def test_processor_with_gate_end_to_end(tok, gate):
    proc = HNEGLogitsProcessor(tok, grammar_gate=gate)
    prompt = tok.ids("```", "python", "\n", "▁▁▁▁", "▁return")
    out = proc(torch.tensor([prompt]), torch.zeros(1, len(tok.pieces)))
    assert out[0, tok.vocab["▁secure"]].item() == -math.inf
    assert out[0, tok.vocab["▁expected"]].item() == 0.0
    assert proc.stats["grammar_masked_steps"] == 1


def test_gate_does_not_carry_names_across_newlines(gate, tok):
    """Regression: "$" matched before a trailing newline, so the first token of a new
    line was treated as continuing the previous line's last word (seen on Kaggle)."""
    code = fenced("    expected = sign(payload)\n    # Use hmac.compare_digest for timing-safe comparison\n")
    assert gate.allowed_token_ids(code) is None
    code2 = fenced("    return expected\n")
    assert gate.allowed_token_ids(code2) is None
    b = gate.blocked_token_ids(code2)
    assert b is None or tok.vocab["▁newvar"] not in set(b.tolist())   # new statement: unconstrained


# --------------------------------------------------------------------------- #
# Edit-level anti-replay guard
# --------------------------------------------------------------------------- #
from src.logit_masker import EditReplayGuard  # noqa: E402

EDIT_PIECES = ["<<<<<<< SEARCH\n", "a = 1", "a = 2", "\n", "=======\n", ">>>>>>> REPLACE", "\n>>>>>>> REPLACE",
               "\n=======\n", "=", ">", "b"]


@pytest.fixture
def etok():
    return FakeTokenizer(EDIT_PIECES)


def _blocked(guard, text):
    b = guard.blocked_token_ids(text)
    return set() if b is None else set(b.tolist())


def test_guard_blocks_noop_edit_close(etok):
    g = EditReplayGuard(etok)
    close = set(etok.ids(">>>>>>> REPLACE", "\n>>>>>>> REPLACE", ">"))
    assert close <= _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 1\n")       # REPLACE == SEARCH
    assert set(etok.ids("\n>>>>>>> REPLACE", "\n=======\n")) == _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 1")
    close = set(etok.ids(">>>>>>> REPLACE", ">"))
    assert not close & _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 2\n")    # a real change may close
    assert not _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 1\n>>>>>>> REPLACE")  # already closed


def test_guard_blocks_replayed_failed_pair_only(etok):
    g = EditReplayGuard(etok)
    g.add_failed_pair("a = 1", "a  =  2")                                          # whitespace-normalised
    assert set(etok.ids(">>>>>>> REPLACE")) <= _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 2\n")
    close = set(etok.ids(">>>>>>> REPLACE", ">"))
    assert not close & _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 2\nb\n")  # different replacement
    assert not close & _blocked(g, "<<<<<<< SEARCH\nb\n=======\na = 2\n")         # different anchor
    assert not set(etok.ids("=======\n")) & _blocked(g, "```python\n<<<<<<< SEARCH\na = 1\n")  # anchor may proceed


def test_guard_blocks_separator_after_unfound_search(etok):
    g = EditReplayGuard(etok)
    g.add_failed_search("a = 2")
    assert set(etok.ids("=======\n", "\n=======\n", "=")) <= _blocked(g, "<<<<<<< SEARCH\na = 2\n")
    assert not set(etok.ids("=======\n")) & _blocked(g, "<<<<<<< SEARCH\na = 1\n")   # other SEARCH may proceed


def test_processor_applies_guard_and_counts(etok):
    g = EditReplayGuard(etok)
    proc = HNEGLogitsProcessor(etok, replay_guard=g)
    ids = etok.ids("<<<<<<< SEARCH\n", "a = 1", "\n", "=======\n", "a = 1", "\n")
    proc.reset(patch_start=0)
    out = proc(torch.tensor([ids]), torch.zeros(1, len(etok.pieces)))
    assert out[0, etok.ids(">>>>>>> REPLACE")[0]] == -math.inf
    assert out[0, etok.ids("b")[0]] == 0
    assert proc.stats["replay_blocked"] == 1


def test_guard_blocks_second_separator_inside_replace(etok):
    g = EditReplayGuard(etok)
    sep = set(etok.ids("=======\n", "\n=======\n", "="))
    # always: a REPLACE section may not contain another separator line
    assert sep <= _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 2\n")
    # replaying a failed pair: both the close marker and the separator escape are masked
    g.add_failed_pair("a = 1", "a = 2")
    blocked = _blocked(g, "<<<<<<< SEARCH\na = 1\n=======\na = 2\n")
    assert sep | set(etok.ids(">>>>>>> REPLACE", ">")) <= blocked
    assert etok.ids("b")[0] not in blocked             # a genuinely different line is still allowed
    assert g.blocked_events == 1


def test_guard_ignores_indented_docstring_underline(etok):
    g = EditReplayGuard(etok)
    g.add_failed_search("Examples")
    # an indented "========" inside SEARCH is a docstring underline: still inside SEARCH,
    # and the real column-0 separator must stay available
    text = "<<<<<<< SEARCH\n    Examples\n    ========\n"
    assert not set(etok.ids("=======\n")) & _blocked(g, text)


def test_guard_blocks_close_marker_inside_search(etok):
    g = EditReplayGuard(etok)
    close = set(etok.ids(">>>>>>> REPLACE", ">"))
    assert close <= _blocked(g, "<<<<<<< SEARCH\na = 1\n")             # never valid before the separator
    g.add_failed_search("a = 2")
    blocked = _blocked(g, "<<<<<<< SEARCH\na = 2\n")                    # failed SEARCH: neither escape works
    assert close | set(etok.ids("=======\n", "=")) <= blocked
    assert etok.ids("b")[0] not in blocked                              # extending the SEARCH is still allowed
