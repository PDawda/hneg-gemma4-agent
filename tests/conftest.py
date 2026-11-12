import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeTokenizer:
    """SentencePiece-style toy tokenizer ("▁" marks a leading space)."""

    eos_token_id = 1

    def __init__(self, pieces):
        self.pieces = ["<pad>", "<eos>"] + list(pieces)
        self.vocab = {p: i for i, p in enumerate(self.pieces)}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            p = self.pieces[i]
            if skip_special_tokens and p.startswith("<") and p.endswith(">"):
                continue
            out.append(p.replace("▁", " "))
        return "".join(out)

    def ids(self, *pieces):
        return [self.vocab[p] for p in pieces]
