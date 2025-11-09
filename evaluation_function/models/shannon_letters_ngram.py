import random
import csv
import os
from pathlib import Path
from .utils import csv_to_lists

from lf_toolkit.evaluation import Result, Params

printing=0

# Setup paths for saving/loading model and data
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LETTERS_PATH = MODEL_DIR / "norvig_letter_ngrams.csv"
WORD_LENGTHS_PATH = MODEL_DIR / "norvig_word_length_frequencies.csv"

# Shannon's English lagnuage generator using letter frequency

# Relative Frequencies of Letters in General English Plain text From Cryptographical Mathematics, by Robert Edward Lewand
# https://web.archive.org/web/20080708193159/http://pages.central.edu/emp/LintonT/classes/spring01/cryptography/letterfreq.html

def read_multingram_csv(filename: Path):
    lookups = {}
    current_n = None

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            first = row[0].strip()
            if first.endswith("-gram"):
                current_n = int(first.split("-")[0])
                lookups[current_n] = {}
                continue

            key, freq = first, float(row[1])
            prefix = key[:current_n - 1] if current_n > 1 else ""

            if prefix not in lookups[current_n]:
                lookups[current_n][prefix] = {"keys": [], "freqs": []}

            lookups[current_n][prefix]["keys"].append(key)
            lookups[current_n][prefix]["freqs"].append(freq)

    return lookups

def sample_ngram(lookups, n, prefix="", k=1):
    data = lookups[n][prefix]
    return random.choices(data["keys"], weights=data["freqs"], k=k)


def generate_word(lookups, N,n, printing=0) -> str: # N = max letters, n = context window (as in, n-gram)
    """Generate a random word using n-gram model up to N letters."""
    #lookups = read_multingram_csv(LETTERS_PATH)
    N_max=N
    samples = {}
    samples[1] = sample_ngram(lookups, n=1, prefix="", k=1)[0]
    print("1-gram:", samples[1]) if printing == 1 else None
    for i in range(2, N+1):
        if len(lookups)<=min(n,i):             
            samples[i] = samples[i-1]+'#'       # ## no i-grams available → stop
            N_max=i
            break
        prefix = samples[i-1][-n+1:]  # previous (i-1)-gram, last n letters
        if prefix not in lookups[len(prefix)+1]: # $$ missing bucket → stop
            if i>2:
                samples[i] = samples[i-1]+"$"  
                N_max=i
            else:
                samples[i] = "$"
                N_max = 1
            break
        else:
            new = sample_ngram(lookups, n=min(i,n), prefix=prefix, k=1)[0]
            print(f"i = {i}, N = {N}, n = {n},new string = {new}") if printing == 1 else None
            samples[i] = samples[i-1][:-n+1]+new
        print(f"{i}-gram:", samples[i]) if printing == 1 else None

    return samples[N_max]

def generate_single_letter(lookups, n, prefix="") -> list:
    """Return top 5 most probable next letters for a given prefix."""
    # Auto-trim prefix if too long
    expected_prefix_len = max(0, n - 1)
    if len(prefix) > expected_prefix_len:
        prefix = prefix[-expected_prefix_len:]  # keep last n-1 chars
    print(prefix)
    print(prefix in lookups.get(n, {}))
    if prefix not in lookups.get(n, {}):
        return []

    data = lookups[n][prefix]
    freqs = data["freqs"]
    keys = data["keys"]
    total = sum(freqs)
    probs = [f / total for f in freqs]

    pairs = sorted(zip(keys, probs), key=lambda x: x[1], reverse=True)
    return pairs[:5]

def run(response, answer, params:Params) -> Result:
    mode = params.get("mode", "production")
    context_window = params.get("context_window", 3)
    printing = params.get("printing", 0)

    if printing:
        print("#### Reading n-gram data ####")
    lookups = read_multingram_csv(LETTERS_PATH)

    result = Result(True)

    # === SINGLE MODE ===
    if mode == "single":
        prefix = params.get("response", "he").upper()
        top5 = generate_single_letter(lookups, context_window, prefix)
        if not top5:
            feedback = f"No data found for prefix '{prefix}' and n={context_window}."
        else:
            feedback_lines = []
            for k, p in top5:
                feedback_lines.append(f"{k[:-1]} | {k[-1]} - {p:.0%}")
            feedback = "<br>".join(feedback_lines)

        result.add_feedback("general", feedback)
        return result
    
    # === PRODUCTION MODE ===
    print("#### Getting data ####")
    data = csv_to_lists(WORD_LENGTHS_PATH)

    print("#### Generating word lengths ####")
    word_lengths = {
        "tokens": [row[0] for row in data],
        "weights": [row[1] for row in data],
    }

    word_count = params.get("word_count", 10)
    response_used = isinstance(response, int) and response > 1

    if word_count == "random":
        word_count = random.randint(3,15)

    print("#### Getting output ####")
    output=[]
    for _ in range(word_count):
        k=int(random.choices(word_lengths["tokens"],weights=word_lengths["weights"],k=1)[0]) 
        output.append(generate_word(lookups,k,context_window))

    print("#### Generating Feedback ####")
    preface = 'Context window: '+str(context_window)+', Word count: '+str(word_count)+'. Output: <br>'
    result.add_feedback("general", preface + ' '.join(output))
    if response_used:
        result.add_feedback("general", "| Answer not an integer >1; used default context window") if not response_used else None

    return result
