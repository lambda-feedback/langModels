import random
import csv
import os
from pathlib import Path
from io import StringIO
import re

from lf_toolkit.evaluation import Result, Params

# Setup paths for saving/loading model and data
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LETTERS_PATH = MODEL_DIR / "norvig_letter_frequencies.csv"
WORD_LENGTHS_PATH = MODEL_DIR / "norvig_word_length_frequencies.csv"

# Relative Frequencies of Letters in General English Plain text From Cryptographical Mathematics, by Robert Edward Lewand
# https://web.archive.org/web/20080708193159/http://pages.central.edu/emp/LintonT/classes/spring01/cryptography/letterfreq.html

def csv_to_lists(filename: str) -> list:
    frequencies = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for key,value in reader:
            frequencies.append([key, float(value)])
    return frequencies

class FrequencyData:
    def __init__(self, filename: str = None):
        self.tokens = []
        self.weights = []
        if filename:
            data = csv_to_lists(filename)
            self.tokens = [row[0] for row in data]
            self.weights = [row[1] for row in data]

uniform_letters = FrequencyData()
uniform_letters.tokens  = [chr(65 + i) for i in range(26)]  # 'A' to 'Z'
uniform_letters.tokens.append(' ')  # Add space character   
uniform_letters.weights = [1] * 27  # Equal weights for uniform distribution    
letters = FrequencyData(LETTERS_PATH)
word_lengths = FrequencyData(WORD_LENGTHS_PATH)

def generate_string(uniform=False,word_count=5) -> str:
    output=[]
    for i in range(word_count):
        k=int(random.choices(word_lengths.tokens,weights=word_lengths.weights,k=1)[0]) 
        if uniform:
            output.append(''.join(random.choices(uniform_letters.tokens, weights=uniform_letters.weights,k=k)))
        else:   
            output.append(''.join(random.choices(letters.tokens, weights=letters.weights,k=k)))
    output=' '.join(output)
    return output

def run(response, answer, params: Params) -> Result:
    is_correct = True
    word_count = params.get("word_count", 10)
    if word_count == "random":
        word_count = random.randint(3,15)
    output = generate_string(uniform=params.get("uniform", False),word_count=word_count)
    return Result(is_correct=is_correct,feedback_items=[("general",output)])
