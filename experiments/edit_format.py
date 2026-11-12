"""SEARCH/REPLACE edit format and deterministic code-context construction.

The model returns only the code it changes::

    ```python
    <<<<<<< SEARCH
    exact lines copied from the file
    =======
    replacement lines
    >>>>>>> REPLACE
    ```

Edits are applied to the original file; the resulting full file then goes
through the same reflex check and test suite as before. Compared with asking
for the whole corrected file, output stays at a few hundred tokens and large
files no longer get truncated.

Everything here is used identically by **every** configuration, so it cannot
bias the ablation:

* ``parse_edit_blocks`` accepts 3-12 marker characters and, if the model
  answers with a unified diff instead, converts its hunks to edit blocks.
* ``apply_edits`` matches exactly, then ignoring trailing whitespace, then
  ignoring indentation, then by a unique fuzzy match (difflib ratio >= 0.9).
  Blocks whose REPLACE equals their SEARCH are skipped.
* ``closest_region`` gives the retry prompt the file lines nearest to a SEARCH
  text that could not be found.
* ``build_code_context`` decides what the model sees: small files in full; for
  large files, the import header plus the definitions whose names appear in
  the issue text, up to a character budget, with explicit omission markers.
"""

from __future__ import annotations

import ast
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

# Markers must start at column 0. Python code is indented, but docstring section
# underlines ("Examples" / "========") are lines of "=" too: an indented one must
# never be taken for the separator. The canonical 7-character markers are tried
# first; 3-12 characters (still at column 0) only if that finds nothing.
def _block_re(n: str) -> "re.Pattern[str]":
    return re.compile(rf"^<{n}[ \t]*SEARCH[^\n]*\n(.*?)\n?^={n}[ \t]*\n(.*?)\n?^>{n}[ \t]*REPLACE",
                      re.DOTALL | re.MULTILINE)


BLOCK = _block_re("{7}")
BLOCK_LENIENT = _block_re("{3,12}")
SEP_LINE = re.compile(r"^={7}[ \t]*$")       # a second separator inside one block
OMISSION = re.compile(r"^\s*# \.\.\. (?:lines \d+-\d+|rest of file) omitted \.\.\.\s*$")
FUZZY_THRESHOLD = 0.9

EDIT_INSTRUCTIONS = (
    "You are a software engineering agent. Fix the bug described in the issue by editing the target file.\n"
    "Reply with edit blocks only, inside one ```python code block, in exactly this format:\n\n"
    "```python\n"
    "<<<<<<< SEARCH\n"
    "    def area(self):\n"
    "        return self.width * self.width\n"
    "=======\n"
    "    def area(self):\n"
    "        return self.width * self.height\n"
    ">>>>>>> REPLACE\n"
    "```\n\n"
    "Rules:\n"
    "- Copy the SEARCH lines character for character from the shown file, with the same indentation.\n"
    "- Include enough surrounding lines that the SEARCH text occurs exactly once in the file.\n"
    "- REPLACE must differ from SEARCH.\n"
    "- Never copy the '# ... omitted ...' marker lines.\n"
    "- Use several blocks for several changes. Do not output the whole file and do not add explanations."
)

FORMAT_REMINDER = (
    "Reply only with edit blocks:\n<<<<<<< SEARCH\n(lines copied exactly from the file)\n=======\n"
    "(replacement lines)\n>>>>>>> REPLACE"
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _diff_hunks(text: str) -> List[Tuple[str, str]]:
    """Convert unified-diff hunks (``@@`` hunks or a ```diff block) to edit blocks."""
    bodies = re.findall(r"```diff[^\n]*\n(.*?)```", text, re.DOTALL)
    if not bodies and "\n@@" in "\n" + text:
        bodies = [text]
    blocks: List[Tuple[str, str]] = []
    for body in bodies:
        hunks: List[List[str]] = [[]]
        for line in body.split("\n"):
            if line.startswith(("---", "+++", "diff ", "index ")):
                continue
            if line.startswith("@@"):
                hunks.append([])
            elif line.startswith("```"):
                break
            else:
                hunks[-1].append(line)
        for h in hunks:
            while h and not h[-1].strip():
                h.pop()
            if not any(l.startswith(("-", "+")) for l in h):
                continue
            if not all(l == "" or l[0] in " +-" for l in h):
                continue                       # not diff-shaped
            search = [l[1:] for l in h if l == "" or l[0] in " -"]
            replace = [l[1:] for l in h if l == "" or l[0] in " +"]
            blocks.append(("\n".join(search), "\n".join(replace)))
    return blocks


def parse_edit_blocks(text: str) -> List[Tuple[str, str]]:
    blocks = BLOCK.findall(text) or BLOCK_LENIENT.findall(text)
    return [(s, r) for s, r in blocks] or _diff_hunks(text)


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_unique(haystack: List[str], needle: List[str], key) -> Optional[int]:
    n = len(needle)
    want = [key(l) for l in needle]
    hits = [i for i in range(len(haystack) - n + 1) if [key(l) for l in haystack[i:i + n]] == want]
    return hits[0] if len(hits) == 1 else (-1 if hits else None)


def _windows(lines: List[str], s_lines: List[str], threshold: float) -> List[Tuple[float, int, int]]:
    """(ratio, start, size) of file windows similar to ``s_lines`` (whitespace-insensitive)."""
    target = "\n".join(l.strip() for l in s_lines if l.strip())
    if not target:
        return []
    stripped = [l.strip() for l in lines]
    n = len(s_lines)
    out = []
    for size in sorted({max(1, n - 1), n, n + 1}):
        for i in range(0, len(lines) - size + 1):
            cand = "\n".join(x for x in stripped[i:i + size] if x)
            sm = SequenceMatcher(None, cand, target, autojunk=False)
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            r = sm.ratio()
            if r >= threshold:
                out.append((r, i, size))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def _fuzzy_find(lines: List[str], s_lines: List[str]) -> Optional[Tuple[int, int]]:
    """Unique best fuzzy window ``(start, size)``; ``(-1, 0)`` if two distinct regions tie."""
    if len("".join(s_lines).strip()) < 20:
        return None                      # too short to match fuzzily without false hits
    hits = _windows(lines, s_lines, FUZZY_THRESHOLD)
    if not hits:
        return None
    r0, i0, n0 = hits[0]
    for r, i, n in hits[1:]:
        if r < r0 - 0.01:
            break
        if i >= i0 + n0 or i + n <= i0:  # a different, non-overlapping region scores the same
            return -1, 0
    return i0, n0


def _reindent(window: List[str], s_lines: List[str], r_lines: List[str]) -> List[str]:
    """Shift ``r_lines`` by the indentation offset between the file window and SEARCH."""
    for fl, sl in zip([l for l in window if l.strip()], [l for l in s_lines if l.strip()]):
        fi, si = len(fl) - len(fl.lstrip()), len(sl) - len(sl.lstrip())
        if fi > si:
            shift = fl[: fi - si]
            return [shift + l if l.strip() else l for l in r_lines]
        if fi < si:
            d = si - fi
            return [l[d:] if l[:d].strip() == "" else l.lstrip() for l in r_lines]
        return r_lines
    return r_lines


def apply_edits(source: str, blocks: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Apply edit blocks in order. Returns ``(new_source, None)`` or ``(None, error)``."""
    if not blocks:
        return None, "no SEARCH/REPLACE edit blocks found in the reply"
    out = source
    applied = 0
    for k, (search, replace) in enumerate(blocks, 1):
        s_lines = [l for l in search.split("\n") if not OMISSION.match(l)]
        r_lines = [l for l in replace.split("\n") if not OMISSION.match(l)]
        search, replace = "\n".join(s_lines), "\n".join(r_lines)
        if not search.strip():
            return None, f"edit block {k}: empty SEARCH section"
        if any(SEP_LINE.match(l) for l in s_lines + r_lines):
            return None, f"edit block {k}: malformed block (more than one ======= separator)"
        if _norm(search) == _norm(replace):
            continue                      # no-op block; error below if nothing else changes
        count = out.count(search)
        if count == 1:
            out = out.replace(search, replace, 1)
            applied += 1
            continue
        if count > 1:
            return None, f"edit block {k}: SEARCH text matches {count} places; add more context lines"
        lines = out.split("\n")
        i = _find_unique(lines, s_lines, lambda l: l.rstrip())
        size = len(s_lines)
        if i is None:
            i = _find_unique(lines, s_lines, lambda l: l.strip())
        if i is None:
            hit = _fuzzy_find(lines, s_lines)
            if hit is not None:
                i, size = hit
        if i is None:
            return None, f"edit block {k}: SEARCH text not found in the file"
        if i == -1:
            return None, f"edit block {k}: SEARCH text matches several places; add more context lines"
        window = lines[i:i + size]
        lines[i:i + size] = _reindent(window, s_lines, r_lines)
        out = "\n".join(lines)
        applied += 1
    if not applied or out == source:
        return None, "the edits do not change the file"
    return out, None


def failed_block_index(error: Optional[str]) -> Optional[int]:
    """0-based index of the block named in an ``apply_edits`` error, if any."""
    m = re.match(r"edit block (\d+):", error or "")
    return int(m.group(1)) - 1 if m else None


def closest_region(source: str, search: str, context: int = 2, max_lines: int = 30) -> Optional[str]:
    """The file lines most similar to ``search`` (plus ``context`` lines), for retry feedback."""
    lines = source.split("\n")
    s_lines = [l for l in search.split("\n") if not OMISSION.match(l)]
    if not "".join(s_lines).strip():
        return None
    hits = _windows(lines, s_lines, 0.5)
    if not hits:
        return None
    _, i, n = hits[0]
    a, b = max(0, i - context), min(len(lines), i + n + context)
    return "\n".join(lines[a:min(b, a + max_lines)])


def edit_feedback(source: str, blocks: List[Tuple[str, str]], error: str) -> str:
    """Retry message for an edit that could not be applied (same for every config)."""
    msg = f"Your edit could not be applied: {error}"
    k = failed_block_index(error)
    if k is not None and k < len(blocks) and ("not found" in error or "several places" in error):
        near = closest_region(source, blocks[k][0])
        if near:
            msg += f"\nThe closest lines in the file are:\n```python\n{near}\n```\nCopy SEARCH lines exactly from the file."
    elif "no SEARCH/REPLACE" in error or "do not change" in error:
        msg += "\n" + FORMAT_REMINDER
    return msg


# --------------------------------------------------------------------------- #
# Code context
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def build_code_context(source: str, issue: str, max_chars: int = 24000) -> Tuple[str, bool]:
    """Return ``(code_shown, is_full_file)``.

    Files up to ``max_chars`` are shown whole. Larger files are reduced to the
    import header plus the top-level/class-level definitions whose names occur
    in the issue (ranked by mention count, then file order), with
    ``# ... lines A-B omitted ...`` markers so SEARCH blocks stay exact.
    """
    if len(source) <= max_chars:
        return source, True
    lines = source.split("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(lines)[:max_chars], False

    mentions = {}
    for w in _IDENT.findall(issue):
        mentions[w] = mentions.get(w, 0) + 1

    # (start, end, name, class_header) with 1-based inclusive lines; methods carry
    # their class header span so it can be shown alongside them.
    spans: List[Tuple[int, int, str, Optional[Tuple[int, int]]]] = []

    def collect(nodes, owner: Optional[Tuple[int, int]] = None) -> None:
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in n.decorator_list] + [n.lineno])
                end = getattr(n, "end_lineno", n.lineno)
                if isinstance(n, ast.ClassDef) and owner is None:
                    header = (start, n.lineno)
                    spans.append((start, n.lineno, n.name, None))   # class header line(s) only
                    collect(n.body, header)
                else:
                    spans.append((start, end, n.name, owner))
    collect(tree.body)

    header_end = 0
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Expr)) or (isinstance(n, ast.Assign) and n.lineno < 60):
            header_end = getattr(n, "end_lineno", n.lineno)
        else:
            break

    ranked = sorted((s for s in spans if mentions.get(s[2])), key=lambda s: (-mentions[s[2]], s[0]))
    chosen = {(1, max(header_end, 1))}
    budget = max_chars - len("\n".join(lines[:header_end]))
    for start, end, _, owner in ranked:
        extra = [(start, end)] + ([owner] if owner and owner not in chosen else [])
        size = sum(len("\n".join(lines[a - 1:b])) for a, b in extra)
        if size > budget:
            continue
        chosen.update(extra)
        budget -= size
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(chosen):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    if len(merged) == 1 and not ranked:
        # Nothing in the issue matched a definition: show the head of the file.
        return "\n".join(lines)[:max_chars] + "\n# ... rest of file omitted ...", False

    parts, prev = [], 0
    for s, e in merged:
        if s > prev + 1:
            parts.append(f"# ... lines {prev + 1}-{s - 1} omitted ...")
        parts.append("\n".join(lines[s - 1:e]))
        prev = e
    if prev < len(lines):
        parts.append(f"# ... lines {prev + 1}-{len(lines)} omitted ...")
    return "\n".join(parts), False
