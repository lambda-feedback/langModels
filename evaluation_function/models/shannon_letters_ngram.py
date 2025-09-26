import random
import csv
import os
from pathlib import Path
from io import StringIO
import re

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

import csv, re, random

def read_multingram_csv(filename: str):
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

NGRAM_LOOKUPS = read_multingram_csv(LETTERS_PATH)

def sample_ngram(lookups, n, prefix="", k=1):
    data = lookups[n][prefix]
    return random.choices(data["keys"], weights=data["freqs"], k=k)


def generate_word(n) -> str: # n is the number of letters in the word
    lookups = NGRAM_LOOKUPS
    n_max=n
    samples = {}
    samples[1] = sample_ngram(lookups, n=1, prefix="", k=1)[0]
    print("1-gram:", samples[1]) if printing == 1 else None
    for i in range(2, n+1):
        if len(lookups)<=i:             # no i-grams available → stop
            samples[i] = samples[i-1]+'#'
            n_max=i
            break
        prefix = samples[i-1]  # previous (i-1)-gram is the prefix
        if prefix not in lookups[i]:   # missing bucket → stop
            if i>2:
                samples[i] = samples[i-1]+"#"
                n_max=i
            else:
                samples[i] = "#"
            break
        else:
            samples[i] = sample_ngram(lookups, n=i, prefix=prefix, k=1)[0]
        print(f"{i}-gram:", samples[i]) if printing == 1 else None

    return samples[n_max]

def csv_to_lists(filename: str) -> list:
    frequencies = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for key,value in reader:
            frequencies.append([key, float(value)])
    return frequencies

def run(response, answer, params:Params) -> Result:
    output=[]
    data = csv_to_lists(WORD_LENGTHS_PATH)
    word_lengths = {}
    word_lengths["tokens"] = [row[0] for row in data]
    word_lengths["weights"] = [row[1] for row in data]
    word_count = params.get("word_count", 10)
    if word_count == "random":
        word_count = random.randint(3,15)
    for i in range(word_count):
        k=int(random.choices(word_lengths["tokens"],weights=word_lengths["weights"],k=1)[0]) 
        output.append(generate_word(k))
    output=' '.join(output)
    is_correct = True
    return Result(is_correct=is_correct,feedback_items=[("general",output)])
