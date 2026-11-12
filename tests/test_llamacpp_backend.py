"""Adapter tests with a fake llama_cpp.Llama (no model or llama-cpp-python needed).

Set HNEG_TEST_GGUF=/path/to/model.gguf to also run a live generation test.
"""
import math
import os

import numpy as np
import pytest
import torch

from src.llamacpp_backend import LlamaCppTokenizer, NumpyLogitsAdapter
from src.logit_masker import FailedDiffTrie, HNEGLogitsProcessor, IdentifierScopeGate

PIECES = ["<s>", "</s>", " return", " secure", "_compare", " expected", "(", ")", "```", "python", "\n", " ", "\xe2"]


class FakeLlama:
    def n_vocab(self):
        return len(PIECES)

    def token_eos(self):
        return 1

    def token_bos(self):
        return 0

    def detokenize(self, ids, prev_tokens=None, special=False):
        out = b""
        for i in ids:
            p = PIECES[i]
            if i in (0, 1):
                out += p.encode() if special else b""
            elif p == "\xe2":
                out += b"\xe2"            # partial UTF-8 byte token
            else:
                out += p.encode()
        return out

    def tokenize(self, text, add_bos=True, special=False):
        raise NotImplementedError


def test_tokenizer_adapter():
    tok = LlamaCppTokenizer(FakeLlama())
    pieces = dict((i, p) for p, i in tok.iter_vocab())
    assert pieces[0] == "" and pieces[12] == ""          # control + partial byte -> ""
    assert pieces[3] == " secure"
    assert tok.decode([2, 5]) == " return expected"
    assert tok.special_text(1) == "</s>"
    assert tok.eos_token_id == 1 and tok.bos_token_id == 0


def test_numpy_adapter_applies_gate_and_trie():
    tok = LlamaCppTokenizer(FakeLlama())
    gate = IdentifierScopeGate(tok, {"expected"})
    proc = HNEGLogitsProcessor(tok, grammar_gate=gate, patch_start=1)
    adapter = NumpyLogitsAdapter(proc)
    ids = np.array([0, 8, 9, 10, 2], dtype=np.intc)           # <s> ```python\n return
    out = adapter(ids, np.zeros(len(PIECES), dtype=np.float32))
    assert out.dtype == np.float32
    assert out[3] == -np.inf and out[5] == 0.0              # " secure" blocked, " expected" kept
    assert proc.stats["steps"] == 1 and adapter.overhead_s > 0

    trie = FailedDiffTrie(min_prefix_len=1)
    trie.insert([8, 9, 10, 2, 5])
    proc2 = HNEGLogitsProcessor(tok, failed_trie=trie, patch_start=1)
    out2 = NumpyLogitsAdapter(proc2)(ids, np.zeros(len(PIECES), dtype=np.float32))
    assert out2[5] == -np.inf and np.isfinite(out2).sum() == len(PIECES) - 1


@pytest.mark.skipif(not os.environ.get("HNEG_TEST_GGUF"), reason="set HNEG_TEST_GGUF to a .gguf file")
def test_live_gguf_generation():
    from src.llamacpp_backend import GGUFRunner

    r = GGUFRunner(model_path=os.environ["HNEG_TEST_GGUF"], n_ctx=2048, n_gpu_layers=0, max_new_tokens=48)
    trie = FailedDiffTrie(4)
    proc = HNEGLogitsProcessor(r.tok, failed_trie=trie, grammar_gate=IdentifierScopeGate(r.tok, {"add"}))
    msgs = [{"role": "user", "content": "Write def add(a, b) in a ```python block."}]
    _, ids1, _, _ = r.generate(msgs, proc)
    trie.insert(ids1)
    _, ids2, _, _ = r.generate(msgs, proc)
    assert ids1 and ids2 and ids1 != ids2 and not trie.contains(ids2)
