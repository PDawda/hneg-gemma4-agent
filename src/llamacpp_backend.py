"""llama.cpp (GGUF) backend for H-NEG.

Runs the same micro/macro governance on quantized GGUF checkpoints through
``llama-cpp-python``. This is the sandbox path for laptops and 8-12 GB GPUs,
e.g. ``unsloth/gemma-4-E2B-it-GGUF`` / ``gemma-4-E2B-it-Q4_K_M.gguf``, before
moving to the 31B checkpoint on the Hugging Face ``transformers`` path.

Do not load GGUF files with ``transformers``: it de-quantizes them to full
precision (~10-15 GB for E2B/E4B), which defeats the purpose.

Default sandbox model: ``unsloth/gemma-4-E2B-it-GGUF`` / ``gemma-4-E2B-it-Q4_K_M.gguf``
(3.1 GB, one Kaggle T4); E4B Q4_K_M (5.0 GB) is the higher-quality alternative.

Components
----------
LlamaCppTokenizer   Adapter exposing the tokenizer surface the H-NEG gate and
                    processor use (vocabulary pieces, ``decode``, EOS).
NumpyLogitsAdapter  Wraps :class:`HNEGLogitsProcessor` as a llama-cpp-python
                    ``logits_processor`` (numpy in / numpy out).
GGUFRunner          Drop-in for ``run_benchmark.GemmaRunner``:
                    ``generate(messages, processor) -> (text, new_ids, n_prompt, seconds)``.

llama.cpp applies Gemma's final-logit soft-cap inside the graph, so keep
``symbolic_engine.logit_softcap: null``.
"""

from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .logit_masker import HNEGLogitsProcessor

DEFAULT_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_FILE = "gemma-4-E2B-it-Q4_K_M.gguf"
class ContextOverflow(ValueError):
    """The prompt does not leave room to generate within ``n_ctx``."""


_FALLBACK_EOG_TEXT = {"<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<turn|>"}


# --------------------------------------------------------------------------- #
# Tokenizer adapter
# --------------------------------------------------------------------------- #
class LlamaCppTokenizer:
    """Tokenizer surface over a ``llama_cpp.Llama`` instance.

    Vocabulary pieces are rendered with ``detokenize([id])`` so SentencePiece
    (``▁``) and byte-level BPE (``Ġ``) vocabularies both appear as plain text
    with real leading spaces. Control tokens render as ``""`` and partial UTF-8
    byte tokens as ``""`` too; the gate treats both as non-identifier tokens.
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.n_vocab = int(llm.n_vocab())
        self.eos_token_id = int(llm.token_eos())
        bos = llm.token_bos()
        self.bos_token_id = int(bos) if bos is not None and bos >= 0 else None
        self._pieces: Optional[List[str]] = None

    def _piece(self, i: int) -> str:
        try:
            raw = self.llm.detokenize([i])
        except Exception:  # pragma: no cover - defensive against odd vocab entries
            return ""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return ""  # partial multi-byte token

    @property
    def pieces(self) -> List[str]:
        if self._pieces is None:
            self._pieces = [self._piece(i) for i in range(self.n_vocab)]
        return self._pieces

    def iter_vocab(self) -> Iterator[Tuple[str, int]]:
        """(piece, id) for every id; unlike ``get_vocab`` this never collapses duplicates."""
        return ((p, i) for i, p in enumerate(self.pieces))

    def get_vocab(self) -> Dict[str, int]:
        return {p: i for i, p in enumerate(self.pieces)}

    def special_text(self, i: int) -> str:
        try:
            return self.llm.detokenize([i], special=True).decode("utf-8", errors="ignore")
        except TypeError:  # very old llama-cpp-python without ``special``
            return ""

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if not len(ids):
            return ""
        try:
            raw = self.llm.detokenize(list(map(int, ids)), special=not skip_special_tokens)
        except TypeError:
            raw = self.llm.detokenize(list(map(int, ids)))
        return raw.decode("utf-8", errors="ignore")

    def encode(self, text: str, add_bos: bool = False, special: bool = True) -> List[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=special))


# --------------------------------------------------------------------------- #
# Logits processor adapter
# --------------------------------------------------------------------------- #
class NumpyLogitsAdapter:
    """llama-cpp-python ``logits_processor``: ``(input_ids, scores) -> scores``.

    ``input_ids`` holds prompt + generated tokens, so the wrapped processor's
    ``patch_start`` must be the prompt length (``GGUFRunner`` sets it).
    ``overhead_s`` accumulates wall time spent in H-NEG masking.
    """

    def __init__(self, processor: HNEGLogitsProcessor) -> None:
        self.processor = processor
        self.overhead_s = 0.0

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        p = self.processor
        p.stats["steps"] += 1
        if p.softcap:
            scores = (p.softcap * np.tanh(scores / p.softcap)).astype(scores.dtype, copy=False)
        row = torch.from_numpy(np.asarray(input_ids, dtype=np.int64))
        mask = p.allowed_mask(row, int(scores.shape[-1])).numpy()
        out = np.where(mask, scores, np.float32(-np.inf)).astype(scores.dtype, copy=False)
        self.overhead_s += time.perf_counter() - t0
        return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def resolve_gguf(repo_id: Optional[str] = None, filename: Optional[str] = None, model_path: Optional[str] = None) -> str:
    """Local path to the GGUF file, downloading from the Hub (cached) if needed."""
    if model_path:
        return model_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id or DEFAULT_REPO, filename or DEFAULT_FILE)


class GGUFRunner:
    """GGUF generation with H-NEG masking; same interface as ``GemmaRunner``."""

    def __init__(
        self,
        repo_id: Optional[str] = DEFAULT_REPO,
        filename: Optional[str] = DEFAULT_FILE,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install llama-cpp-python  (see README: GGUF sandbox)") from exc

        self.model_path = resolve_gguf(repo_id, filename, model_path)
        try:
            self.llm = Llama(model_path=self.model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                             n_threads=n_threads, seed=seed, verbose=verbose)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"llama.cpp could not load {self.model_path}. If the error mentions an unknown model "
                "architecture, upgrade llama-cpp-python to a release whose bundled llama.cpp supports it."
            ) from exc
        self.tok = LlamaCppTokenizer(self.llm)
        self.n_ctx = n_ctx
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eog_ids = self._find_eog_ids()
        self._formatter = self._make_formatter()
        self.last_overhead_s = 0.0
        self.last_stop: Dict[str, object] = {}

    # ---- helpers -------------------------------------------------------- #
    def _find_eog_ids(self) -> Set[int]:
        import llama_cpp

        ids = {self.tok.eos_token_id}
        checks = []
        model = getattr(self.llm, "_model", None)
        vocab = getattr(model, "vocab", None)
        if vocab is not None and hasattr(llama_cpp, "llama_vocab_is_eog"):
            checks.append(lambda t: llama_cpp.llama_vocab_is_eog(vocab, t))
        if model is not None and hasattr(llama_cpp, "llama_token_is_eog"):
            checks.append(lambda t: llama_cpp.llama_token_is_eog(model.model, t))
        for check in checks:
            try:
                return ids | {t for t in range(self.tok.n_vocab) if check(t)}
            except Exception:
                continue
        # Fallback: match well-known end-of-turn control token texts.
        return ids | {t for t in range(self.tok.n_vocab) if self.tok.special_text(t) in _FALLBACK_EOG_TEXT}

    def _make_formatter(self):
        template = (self.llm.metadata or {}).get("tokenizer.chat_template")
        if not template:
            return None
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        bos = self.tok.special_text(self.tok.bos_token_id) if self.tok.bos_token_id is not None else ""
        eos = self.tok.special_text(self.tok.eos_token_id)
        return Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)

    def _render(self, messages: List[Dict[str, str]]) -> str:
        if self._formatter is None:
            body = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            return body + "\n\nASSISTANT:"
        return self._formatter(messages=messages).prompt

    def encode(self, messages: List[Dict[str, str]]) -> List[int]:
        """Chat-template the messages; fold ``system`` into the first user turn
        for templates that reject a system role."""
        try:
            prompt = self._render(messages)
        except Exception:
            if messages and messages[0]["role"] == "system" and len(messages) > 1:
                merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                prompt = self._render(merged + messages[2:])
            else:
                raise
        ids = self.tok.encode(prompt, add_bos=False, special=True)
        bos = self.tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos) and self.llm.metadata.get("tokenizer.ggml.add_bos_token", "true") != "false":
            ids = [bos] + ids
        return ids

    # ---- generation ----------------------------------------------------- #
    def generate(self, messages: List[Dict[str, str]], processor: Optional[HNEGLogitsProcessor] = None):
        from llama_cpp import LogitsProcessorList

        prompt_ids = self.encode(messages)
        budget = self.n_ctx - len(prompt_ids)
        if budget <= 16:
            raise ContextOverflow(f"prompt is {len(prompt_ids)} tokens; n_ctx={self.n_ctx} leaves no room to generate")
        max_new = min(self.max_new_tokens, budget)

        lp = None
        adapter = None
        if processor is not None:
            processor.reset(patch_start=len(prompt_ids))
            adapter = NumpyLogitsAdapter(processor)
            lp = LogitsProcessorList([adapter])

        new_ids: List[int] = []
        stop: Dict[str, object] = {"reason": "max_tokens", "token": None, "text": None}
        t0 = time.perf_counter()
        gen = self.llm.generate(prompt_ids, temp=self.temperature, top_k=0 if self.temperature == 0 else 40,
                                top_p=1.0, min_p=0.0, repeat_penalty=1.0, reset=True, logits_processor=lp)
        for tok in gen:
            if tok in self.eog_ids:
                stop = {"reason": "eog", "token": int(tok), "text": self.tok.special_text(int(tok))}
                break
            new_ids.append(int(tok))
            if len(new_ids) >= max_new:
                break
        dt = time.perf_counter() - t0
        self.last_overhead_s = adapter.overhead_s if adapter else 0.0
        stop["raw_reply"] = self.tok.decode(new_ids, skip_special_tokens=False)[:8000]
        self.last_stop = stop
        return self.tok.decode(new_ids), new_ids, len(prompt_ids), dt
