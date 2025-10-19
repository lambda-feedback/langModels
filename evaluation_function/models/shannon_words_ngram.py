"""
A simple n-gram (word) Shannon-style language model with add-one smoothing.
"""
import os, random, pickle, bz2, tempfile
from pathlib import Path
from io import StringIO
from lf_toolkit.evaluation import Result, Params
from .utils import csv_to_lists, build_counts


import sys, traceback
def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

log(f"[DEBUG] Starting shannon_words_ngram.py")

# Local users run the following once (no need if using Docker):
#nltk.download("brown"); nltk.download("reuters"); nltk.download("gutenberg"); nltk.download("webtext")  # CHANGE (one-time)

START, END = "<s>", "</s>"

# Setup paths for saving/loading model and data
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WORD_LENGTHS_PATH = MODEL_DIR / "norvig_word_length_frequencies.csv"
# If creating when deployed: 
#FILE = Path(tempfile.gettempdir()) / "ngram_counts.pkl"
# If creating locally, to be copied when deployed:
FILE = MODEL_DIR / "ngram_counts.pkl.bz2"

def get_counts(n=3):
    if os.path.exists(FILE):
        with bz2.BZ2File(FILE, "rb") as f:
            cache = pickle.load(f)
    else: # from here the deployed version will not work because the corpora are not bundled (to save space)
        cache = {}
    if n not in cache:
        print(f"Building counts for n={n} (this may take a while)...")
        cache[n] = build_counts(n, START, END) # similarly, only works if NLTK corpora are available
        try:
            with bz2.BZ2File(FILE, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: couldn't save n-gram cache to {FILE}: {e}")
    
    counts = cache[n]
    if n == 1:
        counts.setdefault((), {})                 # CHANGE: ensure unigram context exists
        counts[()].pop(END, None)                 # CHANGE: drop </s> if present (old caches)
    return counts

def sample_next(counts, ctx):
    options = counts.get(ctx) or counts.get((), {})
    if not options:
        return None
    words, freqs = zip(*options.items())
    return random.choices(words, freqs)[0]

def generate(start="", max_len=20, n=None):
    start_tokens = start.lower().split()
    n = max(2, len(start_tokens) + 1) if n is None else n  # Note the requirement n>1, otherwise there's 'no context' and the model fails
    counts = get_counts(n)
    start_tokens = start.lower().split()
    need = n-1
    ctx = tuple((([START]*need) + start_tokens)[-need:]) if need else ()
    out = start_tokens[:]
    for _ in range(max_len):
        w = sample_next(counts, ctx)
        if w in (None, END):
            out.append('#')
            break
        out.append(w)
        if need:
            ctx = tuple((list(ctx)+[w])[-need:])
    return " ".join(out)

def run(response, answer, params:Params) -> Result:
    output=[]
    data = csv_to_lists(WORD_LENGTHS_PATH)
    word_lengths = {}
    word_lengths["tokens"] = [row[0] for row in data]
    word_lengths["weights"] = [row[1] for row in data]
    word_count = params.get("word_count", 10)
    if word_count == "random":
        word_count = random.randint(3,15)
    response_used = isinstance(response, str)
    context = response if response_used else "the general" # Default context
    context_window = params.get("context_window", 3) or 3
    output.append(generate(context,word_count,context_window))
    preface = 'Context window: '+str(context_window)+', Word count: '+str(word_count)+'. Output: <br>'
    feedback_items = [("general", preface + ' '.join(output))]
    #feedback_items.append("| Answer not an integer; used default context window") if not response_used else None
    is_correct = True
    print(feedback_items)
    return Result(is_correct=is_correct,feedback_items=feedback_items)
