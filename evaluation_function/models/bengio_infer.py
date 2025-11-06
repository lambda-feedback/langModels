# Inference code for Bengio-style Neural N-gram Language Model
import json, os
from evaluation_function.models.utils import NeuralLM
from pathlib import Path
from lf_toolkit.evaluation import Result, Params

def predict_next(context_words, topk=5, model=None,config=None, sp=None, device=None):
    from evaluation_function.models.utils import encode
    import torch
    N = config["N"]
    UNK = sp.unk_id()
    with torch.no_grad():
        ctx_ids = encode(context_words[-N:])
        if len(ctx_ids) < N:
            ctx_ids = [UNK] * (N - len(ctx_ids)) + ctx_ids
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze()
        topv, topi = torch.topk(probs, k=min(topk, len(probs)))
    return [(sp.id_to_piece(int(i)), float(v)) for v, i in zip(topv, topi)]

def complete(prompt, steps=10,model=None,config=None,sp=None,device=None):
    import random
    import torch
    with torch.no_grad():
        words = sp.encode(prompt, out_type=str)
        for _ in range(steps):
            dist = predict_next(words, topk=5, model=model, config=config, sp=sp, device=device)
            words_probs = [(word, prob) for word, prob in dist]
            words_list, probs = zip(*words_probs)
            next_word = random.choices(words_list, weights=probs, k=1)[0]
            words.append(next_word)
    return sp.decode(words)

def run(response, answer, params: Params) -> Result:
    print("Loading Bengio-style Neural N-gram Language Model for inference...")
    import torch
    import sentencepiece as spm

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODEL_DIR / "bengio_model.pt"
    MODEL_CONFIG_PATH = MODEL_DIR / "bengio_model_config.json"
    BPE_PATH = MODEL_DIR / "bpe.model"
    if not BPE_PATH.exists():
        raise FileNotFoundError(f"Missing SentencePiece model at {BPE_PATH}")
    sp = spm.SentencePieceProcessor(model_file=str(BPE_PATH))

    with open(MODEL_CONFIG_PATH) as f:
        config = json.load(f)

    model = NeuralLM(
        vocab_size=config["vocab_size"],
        n_ctx=config["N"],
        embed_dim=config["EMBED_DIM"],
        hidden=config["HIDDEN"],
        dropout_p=config["DROPOUT_P"]
    ).to(device)

    print(model)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    result=[]
    completion = response if isinstance(response, str) else "the general"
    result.append(complete(completion, steps=20, model=model, config=config, sp=sp, device=device))

    return Result(is_correct=True, feedback_items=[("general", ''.join(result))])    