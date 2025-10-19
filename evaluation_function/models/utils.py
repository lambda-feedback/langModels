import csv

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

def build_counts(n=3, START="<s>", END="</s>"):
    counts = {}
    for sent in corpus_sents():
        tokens = [w.lower() for w in sent]
        s = ([START] * (n - 1)) + tokens + ([END] if n > 1 else [])  
        for i in range(len(s)-n+1):
            ctx = tuple(s[i:i+n-1])
            nxt = s[i+n-1]
            counts.setdefault(ctx, {})
            counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1
    return counts