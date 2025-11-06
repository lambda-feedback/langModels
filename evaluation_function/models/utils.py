import csv, pickle, bz2
import torch.nn as nn
import hashlib

def csv_to_lists(filename: str) -> list:
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
    import nltk
    from nltk.corpus import brown, reuters, gutenberg, webtext
    # Each yields lists of tokens already sentence-segmented
    for s in brown.sents():      yield s
    for s in reuters.sents():    yield s
    for s in gutenberg.sents():  yield s
    for s in webtext.sents():    yield s

def shard_for(ctx, n_shards): # Deterministic shard assignment
    h = hashlib.sha1(str(ctx).encode("utf8")).hexdigest()
    return int(h, 16) % n_shards

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
    
def encode(seq):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file="bpe.model")
    return sp.encode(" ".join(seq), out_type=int)