import textwrap

import pytest

from src.cpg_parser import CodePropertyGraph, reflex_check


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "auth").mkdir()
    (tmp_path / "auth" / "jwt.py").write_text(textwrap.dedent('''
        import hmac
        import hashlib
        from auth.keys import load_secret

        def sign(payload: bytes) -> str:
            return hmac.new(load_secret(), payload, hashlib.sha256).hexdigest()

        def verify_token(payload: bytes, signature: str) -> bool:
            expected = sign(payload)
            return expected == signature
    '''))
    (tmp_path / "auth" / "keys.py").write_text("def load_secret(encoding='utf-8') -> bytes:\n    return b'dev'\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("def ignored(): pass\n")
    return tmp_path


def test_graph_structure(repo):
    g = CodePropertyGraph.from_repo(repo)
    s = g.summary()
    assert s["files"] == 2 and s["function"] == 3
    assert not g.find_definitions("ignored")
    node = g.find_definitions("verify_token")[0]
    assert node.node_id == "auth/jwt.py::verify_token::Node_0x9"
    assert [c.name for c in g.callees(node.node_id)] == ["sign"]
    assert g.node_at("auth/jwt.py", 10).name == "verify_token"


def test_sr_id_tracks_content(repo):
    a = CodePropertyGraph.from_repo(repo).sr_id
    assert a == CodePropertyGraph.from_repo(repo).sr_id
    (repo / "auth" / "keys.py").write_text("def load_secret():\n    return b'prod'\n")
    assert CodePropertyGraph.from_repo(repo).sr_id != a


def test_symbol_allowlist(repo):
    allow = CodePropertyGraph.from_repo(repo).symbol_allowlist("auth/jwt.py", "verify_token")
    assert {"sign", "load_secret", "hmac", "hashlib", "payload", "signature", "expected", "encoding"} <= allow
    assert "secure_compare" not in allow


def test_reflex_check():
    allow = {"sign", "hmac"}
    good = "def verify_token(payload, signature):\n    expected = sign(payload)\n    return hmac.compare_digest(expected, signature)\n"
    assert reflex_check(good, allow).ok
    bad = reflex_check("def f(x)\n    return x\n", allow)
    assert not bad.ok and bad.lineno == 1 and "SyntaxError" in bad.error
    oos = reflex_check(good.replace("hmac.compare_digest", "secure_compare"), allow)
    assert not oos.ok and oos.unknown_names == ("secure_compare",)
    mixed = reflex_check("if True:\n \tpass\n")
    assert not mixed.ok and "tabs" in mixed.error
    assert reflex_check(good).latency_ms < 50
