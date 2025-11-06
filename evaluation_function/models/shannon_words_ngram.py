"""
A simple n-gram (word) Shannon-style language model with add-one smoothing.
"""
import lmdb, pickle, json, os, random
from pathlib import Path
from io import StringIO
from lf_toolkit.evaluation import Result, Params
from .utils import shard_for
from collections import defaultdict
import hashlib

# Local users run the following once (no need if using Docker):
#nltk.download("brown"); nltk.download("reuters"); nltk.download("gutenberg"); nltk.download("webtext")  # CHANGE (one-time)

START, END = "<s>", "</s>"

# Setup paths for saving/loading model and data
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"/"lmdb_sharded"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = "0"
    
def normalize_context(ctx, n):
    """Ensure context length == n-1 by truncating or padding."""
    target_len = n - 1
    if len(ctx) > target_len:
        ctx = ctx[-target_len:]  # keep most recent tokens
    elif len(ctx) < target_len:
        pad = (START,) * (target_len - len(ctx))
        ctx = pad + tuple(ctx)
    return tuple(ctx)

def query_sharded(n, context):
    """Query the sharded LMDB for the given n-gram context. Returns counts dict or None."""
    context = normalize_context(context, n)
    n_dir = BASE_DIR / f"ngrams_{n}"
    with open(n_dir / "index.json") as f:
        index = json.load(f)
    shard = shard_for(tuple(context), len(index))
    env = lmdb.open(index[str(shard)], readonly=True, lock=False)
    with env.begin() as txn:
        data = txn.get(pickle.dumps(tuple(context)))
        if not data:
            print(f"Context {context} not found in shard {shard}.")
            print(index)
        return pickle.loads(data) if data else None

def generate(start="", max_len=20, n=None, dev=False):
    start_tokens = start.lower().split()
    n = max(2, len(start_tokens) + 1) if n is None else n  # Note the requirement n>1, otherwise there's 'no context' and the model fails
    start_tokens = start.lower().split()
    need = n-1
    ctx = tuple((([START]*need) + start_tokens)[-need:]) if need else ()
    out = start_tokens[:]
    for _ in range(max_len):
        res = query_sharded(n, ctx)
        next_word = max(res, key=res.get) if res else None

        if next_word in (None, END):
            out.append('#')
            break
        out.append(next_word)
        if need:
            ctx = tuple((list(ctx)+[next_word])[-need:])
    return " ".join(out)

def run(response, answer, params:Params) -> Result:
    output=[]
    word_count = params.get("word_count", 10)
    if word_count == "random":
        word_count = random.randint(3,15)
    response_used = isinstance(response, str)
    context = response if response_used else "the general" # Default context
    context_window = params.get("context_window", 3) or 3
    try:
        output.append(generate(context,word_count,context_window,dev=params.get("dev", False)))
    except Exception as e:
        #return Result(is_correct=False,feedback_items=[("general", f"An error occured."),("error",str(e))])
        tb = traceback.format_exc()
        return {
            "status": "error",
            "is_correct": False,
            "feedback": "An error occurred.",
            "error_message": str(e),
            "traceback": tb,
        }
    preface = 'Context window: '+str(context_window)+', Word count: '+str(word_count)+'. Output: <br>'
    feedback_items = [("general", preface + ' '.join(output))]
    #feedback_items.append("| Answer not an integer; used default context window") if not response_used else None
    is_correct = True
    print(feedback_items)
    return Result(is_correct=is_correct,feedback_items=feedback_items)
