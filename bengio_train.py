# Minimal Bengio-style Neural N-gram Language Model
# Training only
from lf_toolkit.evaluation import Result, Params

def run(response, answer, params: Params) -> Result:

    import math, re, json, os
    import numpy as np
    import sentencepiece as spm
    from collections import Counter
    from pathlib import Path

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from evaluation_function.models.utils import NeuralLM, encode

    # Setup paths for saving/loading model and data
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODEL_DIR / "bengio_model.pt"
    MODEL_CONFIG_PATH = MODEL_DIR / "bengio_model_config.json"

    import nltk
    #nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    print('Starting ...')

    # -----------------------
    # 0) Hyperparameters
    # -----------------------
    N = 7                # context length (previous n words)
    EMBED_DIM = 64        # word embedding size
    HIDDEN = 256          # hidden layer width
    LR = 1e-4
    EPOCHS = 20
    BATCH = 64
    MIN_FREQ = 2          # keep all tokens initially; raise for larger corpora
    MAX_VOCAB = 25000     # truncate if needed
    DROPOUT_P = 0.3
    WEIGHT_DECAY = 1e-5

    # -----------------------
    # 1) Data 
    # -----------------------
    if 0:
        TEXT = (
    "the cat sat on the mat \
    the dog sat on the rug \
    the man ate an apple \
    the woman ate a pear \
    the cat chased the mouse \
    the dog chased the cat \
    the man saw the woman \
    the woman saw the man"
        )
    elif 0:
        with open("corpus_generated_large.txt", "r", encoding="utf-8") as f:
            TEXT = f.read()
    else:
        def load_brown_subset(max_tokens=100_000, categories=None):
            """Return a string of up to `max_tokens` tokens from selected Brown categories."""
            tokens = []
            cats = categories or brown.categories()
            for cat in cats:
                for sent in brown.sents(categories=cat):
                    clean = [w.lower() for w in sent if any(c.isalpha() for c in w)]
                    if clean:
                        tokens.extend(clean + ["<eos>"])
                    if len(tokens) >= max_tokens:
                        tokens = tokens[:max_tokens]
                        return " ".join(tokens)
            return " ".join(tokens)
        def make_hybrid_corpus(synthetic_path="corpus_generated_large.txt", brown_tokens=100_000):
            synthetic = Path(synthetic_path).read_text()
            brown_part = load_brown_subset(max_tokens=brown_tokens, categories=["fiction", "romance", "mystery"])
            return synthetic + "\n" + brown_part
        TEXT = load_brown_subset(max_tokens=250_000, categories=["fiction", "romance", "mystery"])#make_hybrid_corpus()

    # --- Subword tokenisation (SentencePiece) ---

    # Save TEXT to file for SentencePiece training
    # Ensure the corpus has one sentence per line for SentencePiece
    # Split on <eos> markers or periods to create sensible line breaks
    lines = re.split(r"<eos>|[.!?]", TEXT)
    clean_lines = [ln.strip() for ln in lines if ln.strip()]
    Path("corpus.txt").write_text("\n".join(clean_lines))

    # Train only if model not already present
    if not Path("bpe.model").exists():
        spm.SentencePieceTrainer.train(
            input="corpus.txt",
            model_prefix="bpe",
            vocab_size=8000,          # adjust as needed
            model_type="bpe",
            character_coverage=1.0
        )

    sp = spm.SentencePieceProcessor(model_file="bpe.model")

    def tokenize(text: str):
        return sp.encode(text, out_type=str)

    tokens = tokenize(TEXT)

    # -----------------------
    # 2) Vocab
    # -----------------------
    print(f"Creating vocab. Total tokens: {len(tokens)}")

    # SentencePiece handles vocab internally
    print(f"Creating vocab from SentencePiece model ...")
    stoi = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    itos = {i: s for s, i in stoi.items()}
    vocab_size = sp.get_piece_size()
    UNK = sp.unk_id()  # usually 0

    ids = encode(tokens)

    # -----------------------
    # 3) Make (context, next) dataset
    # -----------------------
    print("Making dataset ...")

    def make_dataset(id_seq, n):
        X, y = [], []
        for i in range(len(id_seq) - n):
            ctx = id_seq[i:i+n]
            tgt = id_seq[i+n]
            X.append(ctx)
            y.append(tgt)
        X = torch.tensor(np.array(X), dtype=torch.long)
        y = torch.tensor(np.array(y), dtype=torch.long)
        return X, y

    X, y = make_dataset(ids, N)

    # Train/val split (simple)
    num = len(X)
    cut = int(0.9 * num)
    X_tr, y_tr = X[:cut], y[:cut]
    X_va, y_va = X[cut:], y[cut:]

    # -----------------------
    # 4) Model (Bengio 2003 style)
    #    embeddings -> concat -> tanh hidden -> softmax
    # -----------------------
    print("Building model ...")

    vocab_size = sp.get_piece_size()# len(vocab_list)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralLM(vocab_size, N, EMBED_DIM, HIDDEN, DROPOUT_P).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR,weight_decay=WEIGHT_DECAY)

    # -----------------------
    # 5) Simple mini-batch training loop
    # -----------------------

    print("Training model ...")
    def batches(X, y, batch):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            j = idx[i:i+batch]
            yield X[j], y[j]

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_va, y_va = X_va.to(device), y_va.to(device)

    print(f"Training for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0.0
        for xb, yb in batches(X_tr, y_tr, BATCH):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item() if len(X_va) else float('nan')
        ppl = math.exp(val_loss) if val_loss == val_loss else float('nan')
        print(f"epoch {epoch:2d} | train_loss {train_loss:.3f} | val_loss {val_loss:.3f} | ppl {ppl:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    config = {
        "N": N,
        "EMBED_DIM": EMBED_DIM,
        "HIDDEN": HIDDEN,
        "DROPOUT_P": DROPOUT_P,
        "vocab_size": vocab_size,
        "model_path": "bengio_model.pt",
    }
    json.dump(config, open(MODEL_CONFIG_PATH, "w"), indent=2)
    print("✅ Saved model weights and config.")
    return Result(is_correct=True, feedback_items=[("general", "Model trained successfully.")])