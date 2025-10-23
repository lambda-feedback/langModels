import csv, pickle, bz2

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

def build_counts(FILE, n=[1,2,3,4], START="<s>", END="</s>"):
    for j in n:
        print(f"Building {j}-gram counts...")
        FILE_NAME = FILE.with_name(FILE.stem + f"_{j:02d}" + "".join(FILE.suffixes))
        counts = {}
        for sent in corpus_sents():
            tokens = [w.lower() for w in sent]
            s = ([START] * (j - 1)) + tokens + ([END] if j > 1 else [])  
            for i in range(len(s)-j+1):
                ctx = tuple(s[i:i+j-1])
                nxt = s[i+j-1]
                counts.setdefault(ctx, {})
                counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1
        with bz2.BZ2File(FILE_NAME, "wb") as f:
            pickle.dump(counts,f)
    return