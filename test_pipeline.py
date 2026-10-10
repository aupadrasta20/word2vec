"""Basic tests. Run with `pytest` or plain `python tests/test_pipeline.py`."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import (build_vocab, encode, make_skipgram_pairs,  # noqa: E402
                        save_outputs, strip_gutenberg_boilerplate, tokenize, word_to_id)
import evaluate  # noqa: E402  (friend's file)
from make_placeholder import make_placeholder  # noqa: E402


def test_tokenize_normalizes():
    assert tokenize("The King, ruled!  The KINGDOM's 1st day; don’t.") == \
        ["the", "king", "ruled", "the", "kingdom's", "st", "day", "don't"]


def test_gutenberg_strip():
    text = "license\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter"
    assert strip_gutenberg_boilerplate(text).strip() == "body"


def test_vocab_is_deterministic_and_filters_rare():
    tokens = "b a a c c c rare".split()
    vocab = build_vocab(tokens, min_count=1)
    assert vocab["id2word"] == ["c", "a", "b", "rare"]  # freq desc, ties alphabetical
    assert vocab["counts"] == [3, 2, 1, 1]
    assert "rare" not in build_vocab(tokens, min_count=2)["id2word"]


def test_skipgram_pairs_example():
    tokens = "the king ruled the kingdom".split()
    vocab = build_vocab(tokens, min_count=1)
    w2i, i2w = word_to_id(vocab), vocab["id2word"]
    pairs = make_skipgram_pairs(encode(tokens, w2i), window=2)
    words = sorted((i2w[t], i2w[c]) for t, c in pairs)
    king_contexts = sorted(c for t, c in words if t == "king")
    assert king_contexts == ["ruled", "the", "the"]       # "the" left, "ruled"+"the" right
    # Each of 5 positions pairs with all neighbors within 2: 2+3+4+3+2 = 14
    assert len(pairs) == 14
    assert pairs.dtype == np.int64 and pairs.shape[1] == 2


def test_skipgram_short_sequences():
    assert make_skipgram_pairs([0], window=3).shape == (0, 2)
    assert make_skipgram_pairs([0, 1, 2], window=5).shape == (6, 2)


def test_placeholder_matches_friends_loader():
    """Our outputs must load with evaluate.load_model and query with most_similar."""
    with tempfile.TemporaryDirectory() as d:
        proc, models = os.path.join(d, "processed"), os.path.join(d, "run1")
        vocab = build_vocab("x y y z z z".split(), min_count=1)
        save_outputs(proc, vocab, np.zeros((0, 2), dtype=np.int64), {})
        make_placeholder(proc, models, dim=4)
        emb, w2i, i2w = evaluate.load_model(models)
        assert emb.shape == (3, 4) and i2w == vocab["id2word"]
        assert [w for w, _ in evaluate.most_similar("z", emb, w2i, i2w, k=5)] != []
        try:
            evaluate.most_similar("zebra", emb, w2i, i2w)
            assert False, "expected KeyError"
        except KeyError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
