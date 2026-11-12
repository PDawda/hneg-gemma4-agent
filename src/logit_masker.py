"""Micro-governance layer: inline token-logit masking for Gemma 4.

Implements the constrained logit projection

    z~_t(v) = SoftCap(z_t)(v) + M_t(v),   M_t(v) in {0, -inf}

where a token survives only if it is in ValidTokens(S_t) (grammar / CPG scope
gate) and not in TriePrune(H_failed) (anti-loop exclusion of previously failed
patch prefixes).

Components
----------
FailedDiffTrie        Prefix trie of failed patch token sequences.
IdentifierScopeGate   Reference ValidTokens(S_t) gate restricting identifiers to
                      the CPG symbol allowlist of the active execution frame.
EditReplayGuard       Anti-replay exclusion for SEARCH/REPLACE edits: masks the
                      token that would complete a no-op or already-failed edit.
HNEGLogitsProcessor   Hugging Face ``LogitsProcessor`` combining the masks.
filter_draft_tokens   Prunes speculative (MTP / assistant-model) draft tokens
                      against the same mask before verification.

The grammar gate is pluggable: any object exposing
``allowed_token_ids(generated_text: str) -> Optional[Iterable[int]]`` can be
passed as ``grammar_gate`` (e.g. an adapter around SynCode, Outlines or
XGrammar for full CFG enforcement). Returning ``None`` means "unconstrained at
this step".
"""

from __future__ import annotations

import bisect
import builtins
import keyword
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import torch

try:  # transformers is optional at import time so the trie can be unit-tested alone
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover
    class LogitsProcessor:  # type: ignore[no-redef]
        """Minimal stand-in for ``transformers.LogitsProcessor``."""

        def __call__(self, input_ids, scores):
            raise NotImplementedError


# --------------------------------------------------------------------------- #
# Anti-loop failure trie
# --------------------------------------------------------------------------- #
@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class FailedDiffTrie:
    """In-memory prefix trie H_failed over failed patch token sequences.

    ``blocked_next(prefix)`` returns the continuation tokens that would keep the
    generator on the path of a previously failed patch. Masking only begins once
    the shared prefix reaches ``min_prefix_len`` tokens, so common boilerplate at
    the start of a patch (imports, fences) is not penalised immediately.
    """

    def __init__(self, min_prefix_len: int = 8) -> None:
        if min_prefix_len < 0:
            raise ValueError("min_prefix_len must be >= 0")
        self.root = _TrieNode()
        self.min_prefix_len = min_prefix_len
        self.num_sequences = 0

    def insert(self, token_ids: Sequence[int]) -> None:
        node = self.root
        for tok in token_ids:
            node = node.children.setdefault(int(tok), _TrieNode())
        node.terminal = True
        self.num_sequences += 1

    def _walk(self, prefix: Sequence[int]) -> Optional[_TrieNode]:
        node = self.root
        for tok in prefix:
            node = node.children.get(int(tok))
            if node is None:
                return None
        return node

    def blocked_next(self, prefix: Sequence[int]) -> Set[int]:
        if len(prefix) < self.min_prefix_len:
            return set()
        node = self._walk(prefix)
        return set(node.children) if node is not None else set()

    def contains(self, token_ids: Sequence[int]) -> bool:
        node = self._walk(token_ids)
        return bool(node and node.terminal)

    def __len__(self) -> int:
        return self.num_sequences

    def clear(self) -> None:
        self.root = _TrieNode()
        self.num_sequences = 0


# --------------------------------------------------------------------------- #
# Grammar / CPG scope gate
# --------------------------------------------------------------------------- #
class GrammarGate(Protocol):
    """``allowed_token_ids`` returns the allowed ids (``None`` = unconstrained).

    A gate may additionally implement ``blocked_token_ids(generated_text)``
    returning ids (iterable or ``LongTensor``) to remove; this is cheaper when
    almost the whole vocabulary is allowed.
    """

    def allowed_token_ids(self, generated_text: str) -> Optional[Iterable[int]]:
        ...


_IDENT_CHAR = re.compile(r"[A-Za-z0-9_]")
_TOKEN_SPLIT = re.compile(r"^([^A-Za-z0-9_]*)([A-Za-z0-9_]*)(.*)$", re.S)
# \Z, not $: in Python "$" also matches *before* a trailing newline, which made the
# gate treat the first token of every new line as continuing the previous line's name.
_TRAILING_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

PY_KEYWORDS: Set[str] = set(keyword.kwlist) | set(getattr(keyword, "softkwlist", []))
PY_BUILTINS: Set[str] = set(dir(builtins))

_BINDING_CONTEXT = re.compile(
    r"\b(?:def|class|as|for|global|nonlocal|lambda|except)\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*\Z"
)
_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\b")
_DEF_SIGNATURE = re.compile(r"^\s*(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
_NAME_LIST = r"[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*"
_NEW_BINDINGS = (
    re.compile(rf"^[ \t]*\(?({_NAME_LIST})\)?\s*(?::[^=\n]*)?=(?!=)", re.M),   # a, b = ... / x: int = ...
    re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(rf"\bfor\s+\(?({_NAME_LIST})\)?\s+in\b"),
    re.compile(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bimport\s+\(?\s*([A-Za-z_][A-Za-z0-9_.]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_.]*)*)"),
    re.compile(r"\blambda\s+([^:]*):"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:="),
    re.compile(r"\b(?:global|nonlocal)\s+(" + _NAME_LIST + ")"),
)
_DEF_PARAMS = re.compile(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)?", re.S)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _names_bound_in(code: str) -> Set[str]:
    """Identifiers the partial patch binds itself (assignments, defs, params, loops, imports)."""
    names: Set[str] = set()
    for pat in _NEW_BINDINGS:
        for m in pat.finditer(code):
            names.update(_IDENT.findall(m.group(1)))
    for m in _DEF_PARAMS.finditer(code):
        for part in m.group(1).split(","):
            ident = _IDENT.search(part.split(":")[0].split("=")[0])
            if ident:
                names.add(ident.group(0))
    return names


def _prefixes(names: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for n in names:
        for i in range(1, len(n) + 1):
            out.add(n[:i])
    return out


def _in_string_or_comment(text: str) -> bool:
    if (text.count('"""') + text.count("'''")) % 2 == 1:
        return True
    line = text.rsplit("\n", 1)[-1]
    in_single = in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and (in_single or in_double):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not (in_single or in_double):
            return True
        i += 1
    return in_single or in_double


class IdentifierScopeGate:
    """Reference ValidTokens(S_t): restricts *referenced* bare names to a CPG allowlist.

    Every token is split as ``<pre><head><rest>`` where ``head`` is its first
    run of identifier characters. A token is masked if it would make the code
    reference a bare name that is not

    * in the static CPG allowlist (plus Python keywords and builtins), nor
    * bound by the partial patch itself (assignments, ``def``/``class``,
      parameters, loop targets, imports, ``as`` aliases) — recomputed each step.

    Concretely, while the decoder is inside an identifier ``p``, a pure
    identifier token must extend ``p`` toward an allowed name and a token that
    ends the identifier is allowed only if it completes an allowed name. A token
    that *starts* a new identifier after some leading characters (``" foo"``,
    ``"(foo"``) is masked unless ``foo`` is a prefix of an allowed name (or
    exactly an allowed name if the token continues past it).

    The gate is deliberately *unconstrained* where a new name may legitimately
    be introduced, or where the CPG cannot know the namespace:

    * at the start of a statement (possible assignment target),
    * after binding keywords (``def``, ``class``, ``for``, ``as``, ``lambda`` ...),
    * inside a ``def`` parameter list and right after ``[`` / ``{``
      (comprehension variables are used before they are bound),
    * on ``import`` / ``from`` lines,
    * after ``.`` (attributes of external objects), unless ``gate_attributes``,
    * inside strings and comments, and outside the ```` ``` ```` code fence
      when ``require_code_fence`` is set.

    This mirrors :func:`cpg_parser.reflex_check`, which validates loaded bare
    names only. It is a scope gate, not a full CFG parser; plug a SynCode /
    XGrammar adapter in via ``HNEGLogitsProcessor(grammar_gate=...)`` for full
    grammar enforcement.
    """

    def __init__(
        self,
        tokenizer,
        allowed_identifiers: Iterable[str],
        include_python_keywords: bool = True,
        include_builtins: bool = True,
        space_marker: str = "▁",  # SentencePiece "▁"
        require_code_fence: bool = True,
        gate_attributes: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.require_code_fence = require_code_fence
        self.gate_attributes = gate_attributes
        self.allowed: Set[str] = set(allowed_identifiers)
        if include_python_keywords:
            self.allowed |= PY_KEYWORDS
        if include_builtins:
            self.allowed |= PY_BUILTINS
        self._sorted: List[str] = sorted(self.allowed)
        self._allowed_prefixes: Set[str] = _prefixes(self.allowed)

        self._full_ident: Dict[str, List[int]] = {}   # token is entirely identifier chars
        self._mixed: Dict[str, List[int]] = {}        # identifier head, then other chars
        self._non_ident: List[int] = []               # starts with a non-identifier char
        # Tokens "<pre><head>..." that start a new identifier, keyed by pre then head:
        # (token_id, head_is_whole_token_tail). Only tokens whose head is NOT valid
        # against the static allowlist are kept, as candidates for blocking.
        self._starters: Dict[str, Dict[str, List[Tuple[int, bool]]]] = {}
        self._starter_tensors: Dict[str, torch.Tensor] = {}

        vocab_items = tokenizer.iter_vocab() if hasattr(tokenizer, "iter_vocab") else tokenizer.get_vocab().items()
        for piece, tid in vocab_items:
            text = piece.replace(space_marker, " ")
            if len(text) > 2 and text.startswith("<") and text.endswith(">"):
                self._non_ident.append(tid)  # special / byte-fallback tokens
                continue
            pre, head, rest = _TOKEN_SPLIT.match(text).groups()
            if not pre:
                if not head:
                    self._non_ident.append(tid)
                elif not rest:
                    self._full_ident.setdefault(head, []).append(tid)
                else:
                    self._mixed.setdefault(head, []).append(tid)
                continue
            self._non_ident.append(tid)
            if not head or head[0].isdigit() or not self._pre_is_gateable(pre):
                continue
            ends_in_head = not rest
            valid = head in self._allowed_prefixes if ends_in_head else head in self.allowed
            if not valid:
                self._starters.setdefault(pre, {}).setdefault(head, []).append((tid, ends_in_head))

        for pre, heads in self._starters.items():
            ids = [tid for lst in heads.values() for tid, _ in lst]
            self._starter_tensors[pre] = torch.tensor(ids, dtype=torch.long)

    # ---- helpers ---------------------------------------------------------- #
    def _pre_is_gateable(self, pre: str) -> bool:
        if any(c in pre for c in "'\"#\\\n"):
            return False                      # string/comment/new line: never gate
        if pre.endswith(".") and not self.gate_attributes:
            return False
        return True

    def allow(self, *names: str) -> None:
        """Add identifiers at runtime (rebuilds the starter index lazily)."""
        new = set(names) - self.allowed
        if not new:
            return
        self.allowed |= new
        self._sorted = sorted(self.allowed)
        new_prefixes = _prefixes(new)
        self._allowed_prefixes |= new_prefixes
        for pre in list(self._starters):
            heads = self._starters[pre]
            for h in list(heads):
                keep = [(t, e) for t, e in heads[h] if not ((e and h in new_prefixes) or (not e and h in new))]
                if keep:
                    heads[h] = keep
                else:
                    del heads[h]
            ids = [tid for lst in heads.values() for tid, _ in lst]
            self._starter_tensors[pre] = torch.tensor(ids, dtype=torch.long)

    def _names_with_prefix(self, p: str) -> List[str]:
        i = bisect.bisect_left(self._sorted, p)
        out = []
        while i < len(self._sorted) and self._sorted[i].startswith(p):
            out.append(self._sorted[i])
            i += 1
        return out

    def _unconstrained_at_end(self, before: str) -> bool:
        """True if an identifier starting right after ``before`` may be any name."""
        line = before.rsplit("\n", 1)[-1]
        if not line.strip():
            return True                                  # statement start
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            return True
        if stripped.endswith(("[", "{")):
            return True                                  # comprehension variable
        if before.endswith(".") and not self.gate_attributes:
            return True                                  # attribute access
        if _IMPORT_LINE.match(line):
            return True
        if _BINDING_CONTEXT.search(line):
            return True
        if _DEF_SIGNATURE.match(line) and line.count("(") > line.count(")"):
            return True                                  # parameter list
        return False

    def _code_segment(self, generated_text: str) -> Optional[str]:
        if not self.require_code_fence:
            return generated_text
        if generated_text.count("```") % 2 == 0:
            return None                                  # outside a fenced block
        code = generated_text.rsplit("```", 1)[-1]
        if "\n" not in code:
            return None                                  # still on "```python"
        return code.split("\n", 1)[1]

    # Kept for backwards compatibility with earlier callers.
    _in_string_or_comment = staticmethod(_in_string_or_comment)

    # ---- gate API ----------------------------------------------------------- #
    def allowed_token_ids(self, generated_text: str) -> Optional[Set[int]]:
        """Constraint while the decoder is *inside* an identifier."""
        code = self._code_segment(generated_text)
        if code is None:
            return None
        m = _TRAILING_IDENT.search(code)
        if m is None or _in_string_or_comment(code):
            return None
        p, start = m.group(0), m.start()
        if start > 0 and _IDENT_CHAR.match(code[start - 1]):
            return None  # e.g. a number suffix like "0x1F"
        if self._unconstrained_at_end(code[:start]):
            return None

        local = {n for n in _names_bound_in(code[:start]) if n.startswith(p)}
        allowed: Set[int] = set()
        for name in set(self._names_with_prefix(p)) | local:
            rem = name[len(p):]
            for i in range(1, len(rem) + 1):
                allowed.update(self._full_ident.get(rem[:i], ()))
            if rem:
                allowed.update(self._mixed.get(rem, ()))
        if p in self.allowed or p in local:
            allowed.update(self._non_ident)  # further restricted by blocked_token_ids
        return allowed

    def blocked_token_ids(self, generated_text: str) -> Optional[torch.Tensor]:
        """Tokens that would *start* an out-of-scope identifier at this position."""
        code = self._code_segment(generated_text)
        if code is None or _in_string_or_comment(code) or not self._starters:
            return None
        line = code.rsplit("\n", 1)[-1]
        local: Optional[Set[str]] = None
        local_prefixes: Set[str] = set()
        chunks: List[torch.Tensor] = []
        unblock: Set[int] = set()
        for pre, heads in self._starters.items():
            if not heads or self._unconstrained_at_end(line + pre):
                continue
            if local is None:
                local = _names_bound_in(code)
                local_prefixes = _prefixes(local)
            chunks.append(self._starter_tensors[pre])
            for lp in local_prefixes:
                for tid, ends_in_head in heads.get(lp, ()):
                    if ends_in_head or lp in local:
                        unblock.add(tid)
        if not chunks:
            return None
        blocked = torch.cat(chunks)
        if unblock:
            blocked = blocked[~torch.isin(blocked, torch.tensor(sorted(unblock), dtype=torch.long))]
        return blocked


# --------------------------------------------------------------------------- #
# Edit-level anti-replay guard (TriePrune for SEARCH/REPLACE patches)
# --------------------------------------------------------------------------- #
# Column-0 markers only: indented "========" lines are docstring underlines, not separators.
_EDIT_OPEN = re.compile(r"^<{3,12}[ \t]*SEARCH[^\n]*\n", re.MULTILINE)
_EDIT_SEP = re.compile(r"\n?^={3,12}[ \t]*\n", re.MULTILINE)
_EDIT_CLOSE = re.compile(r"^>{3,12}[ \t]*REPLACE", re.MULTILINE)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class EditReplayGuard:
    """Failed-patch exclusion at the level of SEARCH/REPLACE edits.

    A raw token trie over the whole reply is the wrong unit for edit blocks:
    every reply starts with the same fence, marker and anchor lines copied from
    the file, so masking after a shared prefix pushes the model *off* correct
    anchors. This guard instead masks only the token that would *complete* a
    known-bad edit, and leaves the rest of the reply untouched:

    * closing a REPLACE section whose text equals its SEARCH text (a no-op edit),
    * closing a REPLACE section whose (SEARCH, REPLACE) pair already failed,
    * closing a SEARCH section whose text was already not found / ambiguous,
    * a second ``=======`` separator inside a REPLACE section (never valid code;
      otherwise the model can dodge the closing mask by splitting the block).

    Comparison is whitespace-normalised, so a failed edit cannot be replayed
    with cosmetic changes. Use as ``HNEGLogitsProcessor(replay_guard=...)``.
    """

    def __init__(self, tokenizer) -> None:
        self.failed_pairs: Dict[str, Set[str]] = {}
        self.failed_search: Set[str] = set()
        self.blocked_events = 0
        nl_close, close, nl_sep, sep = [], [], [], []
        items = tokenizer.iter_vocab() if hasattr(tokenizer, "iter_vocab") else tokenizer.get_vocab().items()
        for piece, tid in items:
            text = piece.replace("\u2581", " ").replace("\u0120", " ").replace("\u010a", "\n")
            if not text:
                continue
            body = text.lstrip("\n")
            if text.startswith("\n") and body.startswith(">"):
                nl_close.append(tid)
            elif text.startswith(">"):
                close.append(tid)
            if text.startswith("\n") and body.startswith("="):
                nl_sep.append(tid)
            elif text.startswith("="):
                sep.append(tid)
        t = lambda ids: torch.tensor(sorted(set(ids)), dtype=torch.long)  # noqa: E731
        self._nl_close, self._close = t(nl_close), t(nl_close + close)
        self._nl_sep, self._sep = t(nl_sep), t(nl_sep + sep)

    def __len__(self) -> int:
        return len(self.failed_search) + sum(len(v) for v in self.failed_pairs.values())

    def add_failed_pair(self, search: str, replace: str) -> None:
        self.failed_pairs.setdefault(_norm_ws(search), set()).add(_norm_ws(replace))

    def add_failed_search(self, search: str) -> None:
        if search.strip():
            self.failed_search.add(_norm_ws(search))

    def blocked_token_ids(self, generated_text: str) -> Optional[torch.Tensor]:
        opens = list(_EDIT_OPEN.finditer(generated_text))
        if not opens:
            return None
        body = generated_text[opens[-1].end():]
        if _EDIT_CLOSE.search(body):
            return None                         # last block already closed
        at_line_start = generated_text.endswith("\n")
        close = self._close if at_line_start else self._nl_close
        sep = _EDIT_SEP.search(body)
        if sep is None:                         # inside SEARCH
            # A close marker before the separator is never valid (observed on Gemma 4
            # E4B as the escape route when the separator after a failed SEARCH was masked).
            if _norm_ws(body) in self.failed_search:
                self.blocked_events += 1
                return torch.cat([close, self._sep if at_line_start else self._nl_sep])
            return close
        search, replace = _norm_ws(body[:sep.start()]), _norm_ws(body[sep.end():])
        # Inside REPLACE a second "=======" line is never valid: it would split the
        # block and leave a separator line in the code (observed on Gemma 4 E4B
        # as an escape route when closing a replayed edit was masked).
        no_sep = self._sep if at_line_start else self._nl_sep
        if replace == search or replace in self.failed_pairs.get(search, ()):
            self.blocked_events += 1
            return torch.cat([close, no_sep])
        return no_sep


# --------------------------------------------------------------------------- #
# Logits processor
# --------------------------------------------------------------------------- #
class HNEGLogitsProcessor(LogitsProcessor):
    """Applies z~ = SoftCap(z) + M_t inside ``model.generate``.

    Parameters
    ----------
    tokenizer:
        Used to decode the generated suffix for the grammar gate.
    failed_trie:
        Anti-loop trie over raw token sequences (whole-file patches). ``None`` disables it.
    replay_guard:
        :class:`EditReplayGuard` for SEARCH/REPLACE patches: masks only the token
        that would complete a no-op or previously failed edit.
    grammar_gate:
        Object implementing ``allowed_token_ids``. ``None`` disables the gate.
    patch_start:
        Index into ``input_ids`` where the patch begins (typically the prompt
        length). Call :meth:`reset` before each generation.
    softcap:
        Final-logit soft-cap constant. Leave ``None`` for Hugging Face Gemma
        checkpoints, which already apply it inside the model.
    eos_token_id:
        Fallback token if the combined mask would eliminate every token.
    """

    def __init__(
        self,
        tokenizer,
        failed_trie: Optional[FailedDiffTrie] = None,
        grammar_gate: Optional[GrammarGate] = None,
        patch_start: int = 0,
        softcap: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        decode_window: Optional[int] = None,
        replay_guard: Optional["EditReplayGuard"] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.replay_guard = replay_guard
        self.failed_trie = failed_trie
        self.grammar_gate = grammar_gate
        self.patch_start = patch_start
        self.softcap = softcap
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        self.decode_window = decode_window
        self.stats = {"steps": 0, "trie_masked": 0, "grammar_masked_steps": 0, "fallbacks": 0, "replay_blocked": 0}
        self.fallback_log: List[str] = []   # decoded tail at each fallback (gate diagnostics)

    def reset(self, patch_start: int) -> None:
        self.patch_start = patch_start
        self.stats = {k: 0 for k in self.stats}
        self.fallback_log = []

    # ---- per-row mask ---------------------------------------------------- #
    def allowed_mask(self, row: torch.LongTensor, vocab_size: int) -> torch.BoolTensor:
        """Boolean mask of tokens allowed after ``row`` (True = allowed)."""
        device = row.device
        allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
        generated = row[self.patch_start:].tolist()
        text: Optional[str] = None

        if self.grammar_gate is not None:
            window = generated[-self.decode_window:] if self.decode_window else generated
            text = self.tokenizer.decode(window, skip_special_tokens=True)
            ids = self.grammar_gate.allowed_token_ids(text)
            masked = False
            if ids is not None:
                gate = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                idx = [i for i in ids if i < vocab_size]
                if idx:
                    gate[torch.tensor(idx, device=device)] = True
                allowed &= gate
                masked = True
            blocker = getattr(self.grammar_gate, "blocked_token_ids", None)
            if blocker is not None:
                blocked_ids = blocker(text)
                if blocked_ids is not None:
                    b = torch.as_tensor(blocked_ids, dtype=torch.long).to(device)
                    b = b[b < vocab_size]
                    if b.numel():
                        allowed[b] = False
                        masked = True
            self.stats["grammar_masked_steps"] += int(masked)

        grammar_only = allowed.clone()
        if self.replay_guard is not None:
            full = text if (text is not None and not self.decode_window) else \
                self.tokenizer.decode(generated, skip_special_tokens=True)
            before = self.replay_guard.blocked_events
            g = self.replay_guard.blocked_token_ids(full)
            if g is not None and g.numel():
                g = g.to(device)
                allowed[g[g < vocab_size]] = False
            self.stats["replay_blocked"] += self.replay_guard.blocked_events - before
        if self.failed_trie is not None and len(self.failed_trie):
            blocked = [b for b in self.failed_trie.blocked_next(generated) if b < vocab_size]
            if blocked:
                allowed[torch.tensor(blocked, device=device)] = False
                self.stats["trie_masked"] += len(blocked)

        if not allowed.any():
            # Never return an all -inf row (softmax -> NaN). Prefer grammar validity
            # over loop avoidance, then fall back to EOS.
            self.stats["fallbacks"] += 1
            if len(self.fallback_log) < 20:
                tail = text if text is not None else self.tokenizer.decode(generated[-40:], skip_special_tokens=True)
                self.fallback_log.append(tail[-160:])
            allowed = grammar_only
            if not allowed.any() and self.eos_token_id is not None:
                allowed[self.eos_token_id] = True
        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.stats["steps"] += 1
        if self.softcap:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        vocab_size = scores.shape[-1]
        mask = torch.stack([self.allowed_mask(row, vocab_size) for row in input_ids])
        return scores.masked_fill(~mask, -math.inf)


# --------------------------------------------------------------------------- #
# Speculative draft filtering
# --------------------------------------------------------------------------- #
def filter_draft_tokens(
    processor: HNEGLogitsProcessor,
    prefix_ids: torch.LongTensor,
    draft_ids: Sequence[int],
    vocab_size: int,
) -> List[int]:
    """Return the longest prefix of ``draft_ids`` that survives the H-NEG mask.

    Used to prune MTP / assistant-model speculative tokens before the target
    model's verification pass, so grammatically invalid or loop-replicating
    branches never consume verification compute.
    """
    accepted: List[int] = []
    row = prefix_ids.clone()
    for tok in draft_ids:
        if not processor.allowed_mask(row, vocab_size)[tok]:
            break
        accepted.append(int(tok))
        row = torch.cat([row, row.new_tensor([tok])])
    return accepted
