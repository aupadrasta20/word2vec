"""Write RANDOM embeddings in train.py's exact output format, so app.py can be
built and tested before real training finishes. Results are meaningless by design.

Usage: python make_placeholder.py   (after preprocess.py; overwritten by real training)
"""

import json
import os

import numpy as np

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/run1"   # same place train.py saves to


def make_placeholder(processed_dir=PROCESSED_DIR, model_dir=MODEL_DIR, dim=50):
    with open(os.path.join(processed_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    emb = np.random.default_rng(0).normal(size=(len(vocab["id2word"]), dim)).astype(np.float32)
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "embeddings.npy"), emb)
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"placeholder": True, "dim": dim}, f, indent=2)
    print(f"Wrote PLACEHOLDER (random) model to {model_dir}/")


if __name__ == "__main__":
    make_placeholder()
