import csv, pickle, bz2
from pathlib import Path

import hashlib

def csv_to_lists(filename: Path) -> list:
    frequencies = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for key,value in reader:
            frequencies.append([key, float(value)])
    return frequencies


# Only locally: Generate word ngram counts from NLTK corpora
def corpus_sents():  # CHANGE
    # Only import NLTK when this function is called (so not when deployed)

    from nltk.corpus import brown, reuters, gutenberg, webtext
    # Each yields lists of tokens already sentence-segmented
    for s in brown.sents():      yield s
    for s in reuters.sents():    yield s
    for s in gutenberg.sents():  yield s
    for s in webtext.sents():    yield s

def shard_for(ctx, n_shards): # Deterministic shard assignment
    h = hashlib.sha1(str(ctx).encode("utf8")).hexdigest()
    return int(h, 16) % n_shards

_NeuralLM_cls = None


def _neural_lm_cls():
    """Build the torch ``NeuralLM`` class on first use.

    Deferred so that importing this module (e.g. for ``csv_to_lists`` or
    ``shard_for``) does not pull in torch.
    """
    global _NeuralLM_cls
    if _NeuralLM_cls is None:
        import torch.nn as nn

        class NeuralLM(nn.Module):
            def __init__(self, vocab_size, n_ctx, embed_dim, hidden, dropout_p):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, embed_dim)
                self.fc1 = nn.Linear(n_ctx * embed_dim, hidden)
                self.act = nn.Tanh()
                self.drop = nn.Dropout(dropout_p)
                self.fc2 = nn.Linear(hidden, vocab_size)
            def forward(self, ctx_idx):  # ctx_idx: (B, n)
                e = self.emb(ctx_idx)               # (B, n, d)
                x = e.reshape(e.size(0), -1)        # (B, n*d)
                h = self.act(self.fc1(x))           # (B, H)
                logits = self.fc2(h)                # (B, V)
                return logits

        _NeuralLM_cls = NeuralLM
    return _NeuralLM_cls


def NeuralLM(*args, **kwargs):
    """Instantiate the (lazily built) torch ``NeuralLM`` model."""
    return _neural_lm_cls()(*args, **kwargs)


def encode(seq):
    import sentencepiece as spm
    from pathlib import Path
    import os
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BPE_PATH = MODEL_DIR / "bpe.model"
    if not BPE_PATH.exists():
        raise FileNotFoundError(f"Missing SentencePiece model at {BPE_PATH}")
    sp = spm.SentencePieceProcessor(model_file=str(BPE_PATH))
    return sp.encode(" ".join(seq), out_type=int)