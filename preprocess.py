"""Text preprocessing for Skip-gram Word2Vec.

Pipeline: raw text -> normalized tokens -> vocabulary (word <-> integer ID)
-> (target, context) ID pairs -> files that train.py reads.

Usage:
    python preprocess.py                      # defaults: data/raw/*.txt, window=2, min_count=5
    python preprocess.py --window 5 --min-count 3
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: load the corpus
# ---------------------------------------------------------------------------
def strip_gutenberg_boilerplate(text):
    """Project Gutenberg files wrap the book in a license header/footer.
    Keep only the text between the '*** START OF ...' and '*** END OF ...' lines.
    If the markers are missing, return the text unchanged."""
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", text)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()]
    return text


def load_corpus(raw_dir):
    """Read every .txt file in raw_dir (sorted, so the order is reproducible)."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files in {raw_dir}/. See README.md -> 'Get the data'."
        )
    texts = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            texts.append(strip_gutenberg_boilerplate(f.read()))
    return "\n".join(texts), [os.path.basename(p) for p in paths]


# ---------------------------------------------------------------------------
# Steps 2-3: normalize and tokenize
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase, then keep runs of letters (with inner apostrophes, e.g. "don't").

    - Lowercasing: "King" and "king" should be one word, not two.
    - Dropping punctuation and digits: they would otherwise become tokens
      (or glue onto words, e.g. "king," != "king").
    - Curly apostrophes are normalized to straight ones first.
    """
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


# ---------------------------------------------------------------------------
# Steps 4-5: vocabulary (+ rare-word removal)
# ---------------------------------------------------------------------------
def build_vocab(tokens, min_count=5):
    """Map each word with count >= min_count to an integer ID.

    IDs are assigned by descending frequency, ties broken alphabetically,
    so the same corpus always gives the same IDs.

    IMPORTANT: an ID is just a label (like a row number). "king" = 57 does not
    mean king is "close to" 56 or 58. Meaning comes later, from the embedding
    vector the network learns for row 57.
    """
    counts = Counter(tokens)
    kept = [(w, c) for w, c in counts.items() if c >= min_count]
    kept.sort(key=lambda wc: (-wc[1], wc[0]))
    # Saved format (matches evaluate.load_model): only the list is stored;
    # word2id is rebuilt from it, which is safe because position = ID.
    return {
        "id2word": [w for w, _ in kept],
        "counts": [c for _, c in kept],  # train.py uses these for negative sampling
    }


def word_to_id(vocab):
    return {w: i for i, w in enumerate(vocab["id2word"])}


def encode(tokens, w2i):
    """Convert tokens to IDs, dropping words not in the vocabulary (rare words).
    Note: dropping a word brings its neighbors closer together, which slightly
    widens the effective window. This is standard and we document it."""
    return [w2i[t] for t in tokens if t in w2i]


# ---------------------------------------------------------------------------
# Step 6: Skip-gram pairs
# ---------------------------------------------------------------------------
def make_skipgram_pairs(ids, window=2):
    """For each position i, pair ids[i] (target) with every word up to
    `window` positions to the left and right (context).

    Example, window=1: [the, king, ruled] ->
        (the, king), (king, the), (king, ruled), (ruled, king)
    Returns an int64 array of shape (num_pairs, 2).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    chunks = []
    # Vectorized: for each offset (-window..-1, 1..window), pair every
    # position with the position `offset` away, when it exists.
    for offset in range(-window, window + 1):
        if offset == 0 or abs(offset) >= n:
            continue
        if offset > 0:
            targets, contexts = ids[: n - offset], ids[offset:]
        else:
            targets, contexts = ids[-offset:], ids[: n + offset]
        chunks.append(np.stack([targets, contexts], axis=1))
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Step 7: save
# ---------------------------------------------------------------------------
def save_outputs(out_dir, vocab, pairs, meta):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    np.save(os.path.join(out_dir, "pairs.npy"), pairs)
    with open(os.path.join(out_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Step 8: sanity check
# ---------------------------------------------------------------------------
def sanity_check(tokens, vocab, pairs, n=10):
    id_to_word = vocab["id2word"]
    print(f"Tokens (before rare-word removal): {len(tokens):,}")
    print(f"First {n} tokens: {tokens[:n]}")
    print(f"Vocabulary size: {len(id_to_word):,}")
    print(f"Top {n} words: {list(zip(id_to_word[:n], vocab['counts'][:n]))}")
    print(f"Training pairs: {len(pairs):,}")
    print("Example pairs (target -> context):")
    rng = np.random.default_rng(0)
    for t, c in pairs[rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)]:
        print(f"  {id_to_word[t]:>12} ({t}) -> {id_to_word[c]} ({c})")
    # Hard checks: every ID must be a valid row index.
    assert pairs.min() >= 0 and pairs.max() < len(id_to_word), "pair ID out of range"
    assert len(vocab["counts"]) == len(id_to_word), "counts/id2word length mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=5)
    args = parser.parse_args()

    text, files = load_corpus(args.raw_dir)
    tokens = tokenize(text)
    vocab = build_vocab(tokens, args.min_count)
    ids = encode(tokens, word_to_id(vocab))
    pairs = make_skipgram_pairs(ids, args.window)

    meta = {
        "corpus_files": files,
        "min_count": args.min_count,
        "window": args.window,
        "num_tokens_raw": len(tokens),
        "num_tokens_kept": len(ids),
        "vocab_size": len(vocab["id2word"]),
        "num_pairs": int(len(pairs)),
    }
    sanity_check(tokens, vocab, pairs)
    save_outputs(args.out_dir, vocab, pairs, meta)
    print(f"\nSaved to {args.out_dir}/: vocab.json, pairs.npy, preprocess_meta.json")


if __name__ == "__main__":
    main()
