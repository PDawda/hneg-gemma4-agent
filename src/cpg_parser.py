"""Code Property Graph (CPG) extraction and Loop-1 reflex validation.

Builds a lightweight CPG over a repository:

* Python files are parsed with the standard-library ``ast`` module (exact,
  no extra dependencies).
* Other languages (Rust, C++, JS, Go, ...) use tree-sitter via
  ``tree_sitter_language_pack`` when it is installed.

Nodes are definitions, imports and call sites; edges are ``CONTAINS``
(syntactic nesting), ``CALLS`` and ``IMPORTS``. The graph feeds the
micro-layer with a per-frame symbol allowlist and produces the Context
Integrity Contract identifier ``SR_id`` (a hash binding the worktree snapshot
to the graph that was built from it).
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

TS_LANGS = {
    ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".h": "cpp",
    ".c": "c",
    ".js": "javascript", ".ts": "typescript",
    ".go": "go",
    ".java": "java",
}
TS_DEF_TYPES = {
    "function_item", "function_definition", "function_declaration", "method_declaration",
    "struct_item", "enum_item", "trait_item", "impl_item", "class_specifier",
    "struct_specifier", "class_declaration", "type_declaration",
}


@dataclass
class CPGNode:
    node_id: str
    kind: str              # module | class | function | import | call | ts_def
    name: str
    file: str
    lineno: int
    end_lineno: int
    parent: Optional[str] = None
    symbols: Set[str] = field(default_factory=set)  # names bound in this scope


@dataclass
class ReflexResult:
    ok: bool
    latency_ms: float
    error: Optional[str] = None
    lineno: Optional[int] = None
    unknown_names: Tuple[str, ...] = ()
    over_budget: bool = False


class CodePropertyGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, CPGNode] = {}
        self.edges: List[Tuple[str, str, str]] = []           # (src, dst, label)
        self.defs_by_name: Dict[str, List[str]] = defaultdict(list)
        self.file_hashes: Dict[str, str] = {}
        # Parameter and keyword-argument names seen anywhere in the repo, so that
        # keyword arguments to repo (and commonly used library) callables are legal.
        self.param_names: Set[str] = set()
        # Module-level names each module binds (e.g. ``S = SingletonRegistry()``): these are
        # what ``from pkg import *`` re-exports, so they are legal references repo-wide.
        self.module_exports: Set[str] = set()
        # Names each file already reads: anything the original file uses is legal in a patch.
        self.used_names: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_repo(cls, root: Union[str, Path], include: Iterable[str] = ("**/*.py",), exclude_dirs: Iterable[str] = (".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__")) -> "CodePropertyGraph":
        root = Path(root)
        excl = set(exclude_dirs)
        g = cls()
        for pattern in include:
            for path in sorted(root.glob(pattern)):
                if path.is_file() and not (excl & set(path.relative_to(root).parts)):
                    g.add_file(path, rel=str(path.relative_to(root)))
        g.link_calls()
        return g

    def add_file(self, path: Union[str, Path], rel: Optional[str] = None, source: Optional[str] = None) -> None:
        path = Path(path)
        rel = rel or str(path)
        if source is None:
            source = path.read_text(encoding="utf-8", errors="replace")
        self.file_hashes[rel] = hashlib.sha256(source.encode()).hexdigest()
        if path.suffix == ".py":
            self._add_python(rel, source)
        elif path.suffix in TS_LANGS:
            self._add_tree_sitter(rel, source, TS_LANGS[path.suffix])

    def _nid(self, file: str, name: str, lineno: int) -> str:
        return f"{file}::{name}::Node_0x{lineno:X}"

    def _add_node(self, node: CPGNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent:
            self.edges.append((node.parent, node.node_id, "CONTAINS"))
        if node.kind in {"class", "function", "ts_def"}:
            self.defs_by_name[node.name].append(node.node_id)

    def _add_python(self, file: str, source: str) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError:
            return
        mod_id = self._nid(file, "<module>", 0)
        n_lines = source.count("\n") + 1
        used: Set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.arg):
                self.param_names.add(n.arg)
            elif isinstance(n, ast.keyword) and n.arg:
                self.param_names.add(n.arg)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)
        self.used_names[file] = used
        for stmt in tree.body:
            for t in (stmt.targets if isinstance(stmt, ast.Assign) else
                      [stmt.target] if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) else []):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        self.module_exports.add(n.id)
        self._add_node(CPGNode(mod_id, "module", file, file, 0, n_lines, symbols=_bound_names(tree)))

        def visit(node: ast.AST, parent_id: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(child, ast.ClassDef) else "function"
                    nid = self._nid(file, child.name, child.lineno)
                    self._add_node(CPGNode(nid, kind, child.name, file, child.lineno,
                                           getattr(child, "end_lineno", child.lineno), parent_id,
                                           symbols=_bound_names(child)))
                    self.nodes[parent_id].symbols.add(child.name)
                    visit(child, nid)
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        target = alias.name if isinstance(child, ast.Import) else f"{child.module or ''}.{alias.name}"
                        nid = self._nid(file, f"import:{target}", child.lineno)
                        self._add_node(CPGNode(nid, "import", target, file, child.lineno, child.lineno, parent_id))
                        self.edges.append((parent_id, nid, "IMPORTS"))
                elif isinstance(child, ast.Call):
                    name = _call_name(child.func)
                    if name:
                        nid = self._nid(file, f"call:{name}", child.lineno) + f":{child.col_offset}"
                        self._add_node(CPGNode(nid, "call", name, file, child.lineno, child.lineno, parent_id))
                    visit(child, parent_id)
                else:
                    visit(child, parent_id)

        visit(tree, mod_id)

    def _add_tree_sitter(self, file: str, source: str, lang: str) -> None:
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            return
        parser = get_parser(lang)
        tree = parser.parse(source.encode())
        mod_id = self._nid(file, "<module>", 0)
        self._add_node(CPGNode(mod_id, "module", file, file, 0, source.count("\n") + 1))
        src = source.encode()

        def visit(node, parent_id: str) -> None:
            for child in node.children:
                pid = parent_id
                if child.type in TS_DEF_TYPES:
                    name_node = child.child_by_field_name("name")
                    if name_node is not None:
                        name = src[name_node.start_byte:name_node.end_byte].decode()
                        pid = self._nid(file, name, child.start_point[0] + 1)
                        self._add_node(CPGNode(pid, "ts_def", name, file, child.start_point[0] + 1,
                                               child.end_point[0] + 1, parent_id))
                elif child.type in {"identifier", "type_identifier", "field_identifier"}:
                    self.nodes[parent_id].symbols.add(src[child.start_byte:child.end_byte].decode())
                visit(child, pid)

        visit(tree.root_node, mod_id)

    def link_calls(self) -> None:
        """Add CALLS edges from call sites to every definition with that name."""
        for nid, node in list(self.nodes.items()):
            if node.kind == "call":
                short = node.name.rsplit(".", 1)[-1]
                for target in self.defs_by_name.get(short, ()):
                    self.edges.append((nid, target, "CALLS"))

    # ------------------------------------------------------------------ #
    # Queries (the `query_cpg_graph` tool)
    # ------------------------------------------------------------------ #
    @property
    def sr_id(self) -> str:
        """Context Integrity Contract id: hash over (file, content-hash) pairs."""
        h = hashlib.sha256()
        for f in sorted(self.file_hashes):
            h.update(f.encode())
            h.update(self.file_hashes[f].encode())
        return h.hexdigest()[:16]

    def find_definitions(self, name: str) -> List[CPGNode]:
        return [self.nodes[n] for n in self.defs_by_name.get(name, ())]

    def node_at(self, file: str, lineno: int) -> Optional[CPGNode]:
        """Innermost definition in ``file`` enclosing ``lineno``."""
        best = None
        for n in self.nodes.values():
            if n.file == file and n.kind in {"module", "class", "function", "ts_def"} and n.lineno <= lineno <= n.end_lineno:
                if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                    best = n
        return best

    def ancestors(self, node_id: str) -> List[CPGNode]:
        out, cur = [], self.nodes.get(node_id)
        while cur is not None:
            out.append(cur)
            cur = self.nodes.get(cur.parent) if cur.parent else None
        return out

    def callees(self, node_id: str) -> List[CPGNode]:
        calls = {d for s, d, l in self.edges if s == node_id and l == "CONTAINS" and self.nodes[d].kind == "call"}
        return [self.nodes[d] for s, d, l in self.edges if s in calls and l == "CALLS"]

    def symbol_allowlist(self, file: str, node_name: Optional[str] = None) -> Set[str]:
        """Identifiers legally referenceable from the active frame.

        Union of: names bound in the enclosing scope chain, imported names,
        every definition name in the repository (for cross-file references),
        attribute names observed in the scope chain, and parameter / keyword
        argument names used anywhere in the repository, module-level names
        bound anywhere in the repository (re-exported by ``import *``), and every
        name the file itself already reads.
        """
        names: Set[str] = set(self.defs_by_name) | self.param_names | self.module_exports
        names |= self.used_names.get(file, set())
        scope = None
        if node_name:
            scope = next((self.nodes[n] for n in self.defs_by_name.get(node_name, ()) if self.nodes[n].file == file), None)
        if scope is None:
            scope = self.nodes.get(self._nid(file, "<module>", 0))
        if scope is not None:
            for anc in self.ancestors(scope.node_id):
                names |= anc.symbols
        for n in self.nodes.values():
            if n.file == file and n.kind in {"import", "call"}:
                names.update(part for part in n.name.split(".") if part)
        return names

    def summary(self) -> Dict[str, int]:
        kinds: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            kinds[n.kind] += 1
        return {"files": len(self.file_hashes), "nodes": len(self.nodes), "edges": len(self.edges), **kinds}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _call_name(func: ast.AST) -> Optional[str]:
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return ".".join(reversed(parts)) or None


def _bound_names(scope: ast.AST) -> Set[str]:
    """Names bound directly in ``scope`` (params, assignments, imports, defs, attrs)."""
    names: Set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = scope.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if arg is not None:
                names.add(arg.arg)
    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not scope:
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


_BUILTINS = set(dir(builtins))


def reflex_check(source: str, allowlist: Optional[Set[str]] = None, latency_cap_ms: float = 50.0) -> ReflexResult:
    """Loop 1 (Reflex): in-memory syntax, indentation and name-scope validation.

    Runs without launching a container. If ``allowlist`` is given, loaded names
    that are neither bound in the patch, built in, nor in the allowlist are
    reported as ``unknown_names`` and fail the check.
    """
    t0 = time.perf_counter()

    def done(**kw) -> ReflexResult:
        ms = (time.perf_counter() - t0) * 1000
        return ReflexResult(latency_ms=ms, over_budget=ms > latency_cap_ms, **kw)

    for i, line in enumerate(source.splitlines(), 1):
        indent = line[: len(line) - len(line.lstrip())]
        if " " in indent and "\t" in indent:
            return done(ok=False, error="mixed tabs and spaces in indentation", lineno=i)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)   # e.g. invalid escapes in old code
            tree = ast.parse(source)
            compile(tree, "<patch>", "exec")
    except SyntaxError as exc:
        return done(ok=False, error=f"{type(exc).__name__}: {exc.msg}", lineno=exc.lineno)

    if allowlist is not None:
        bound = _bound_names(tree)
        loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        unknown = sorted(loaded - bound - _BUILTINS - allowlist)
        if unknown:
            return done(ok=False, error=f"out-of-scope identifiers: {', '.join(unknown)}", unknown_names=tuple(unknown))
    return done(ok=True)
