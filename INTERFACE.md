# Interface contract

All paths are relative to the repo root. Word IDs are the **same everywhere**:
row `i` of every array corresponds to `id2word[i]`. Nobody rebuilds the
vocabulary after `preprocess.py` runs; `word2id` is always derived from `id2word`.

## 1. `preprocess.py` -> `data/processed/`

| File | Format | Contents |
|---|---|---|
| `vocab.json` | JSON | `{"id2word": [...], "counts": [...]}` |
| `pairs.npy` | NumPy `int64`, shape `(N, 2)` | each row is `[center_id, context_id]` |
| `preprocess_meta.json` | JSON | corpus files, `min_count`, `window`, token counts, `vocab_size`, `num_pairs` |

- IDs are assigned by descending frequency (ties alphabetical), so output is deterministic.
- `counts[i]` feeds `make_noise_dist(counts)` in `train.py`.
- No `<unk>` token: words with `count < min_count` are dropped before pairing.

## 2. `train.py` -> `models/run1/`

| File | Format | Contents |
|---|---|---|
| `embeddings.npy` | NumPy `float32`, shape `(vocab_size, dim)` | `in_embed` weights; row `i` = word `i` |
| `vocab.json` | JSON | the `data/processed/vocab.json` dict, saved **unchanged** |
| `config.json` | JSON | `dim`, `neg_k`, `lr`, `batch_size`, `epochs`, `seed`, plus `preprocess_meta` copied in |
| `loss_history.json` | JSON list | average loss per epoch (or every N steps) |

## 3. `evaluate.py` (shared)

```python
load_model(model_dir) -> (emb, word2id, id2word)
most_similar(word, emb, word2id, id2word, k=10) -> list[(word, cosine)]   # KeyError if unknown
cosine_similarity(vec_a, vec_b) -> float
```

`app.py` only calls these; it lowercases user input before lookup.

## Placeholder model

`python make_placeholder.py` writes random vectors to `models/run1/` in the
format above with `"placeholder": true` in `config.json`; the app shows a
warning for it. Real training overwrites it.
