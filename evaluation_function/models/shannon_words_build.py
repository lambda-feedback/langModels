import lmdb, pickle, nltk, json, os
from nltk.corpus import brown
from pathlib import Path
from collections import defaultdict
import hashlib
from .utils import shard_for
from lf_toolkit.evaluation import Result, Params

os.environ["PYTHONHASHSEED"] = "0"

START, END = "<s>", "</s>"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "storage"/"lmdb_sharded"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def corpus_sents(limit=None):
    for s in brown.sents()[:limit]:
        yield [w.lower() for w in s]

def build_sharded_lmdb(n=4, n_shards=64, map_size=2**28):
    n_dir = MODEL_DIR / f"ngrams_{n}"
    n_dir.mkdir(parents=True, exist_ok=True)

    # open environments (small, parallel)
    envs = [lmdb.open(str(n_dir / f"shard_{i:02d}.lmdb"), map_size=map_size) for i in range(n_shards)]
    index = {i: str(n_dir / f"shard_{i:02d}.lmdb") for i in range(n_shards)}

    print(f"Building {n}-grams into {n_shards} shards...")
    counts = [defaultdict(lambda: defaultdict(int)) for _ in range(n_shards)]

    for sent in corpus_sents(limit=None):
        s = ([START]*(n-1)) + sent + ([END] if n>1 else [])
        for i in range(len(s)-n+1):
            ctx = tuple(s[i:i+n-1])
            nxt = s[i+n-1]
            shard = shard_for(ctx, n_shards)
            counts[shard][ctx][nxt] += 1

    # write per-shard
    for shard, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for ctx, nexts in counts[shard].items():
                txn.put(pickle.dumps(ctx), pickle.dumps(dict(nexts)))
        env.close()
        print(f"✅ shard {shard:02d} written with {len(counts[shard])} contexts")

    with open(n_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"✅ index written to {n_dir/'index.json'}")


def run(response, answer, params:Params) -> Result:
    #nltk.download("brown", quiet=True)
    n_max = params.get("n_max",7)
    for n in range(2,n_max+1):
        print('Building for n=', n)
        build_sharded_lmdb(n=n, n_shards=64)
        print('Complete for n=', n)

    return Result(is_correct=True, feedback_items = [("general", "Complete.")])