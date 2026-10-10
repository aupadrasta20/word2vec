# Word2Vec Skip-gram from Scratch (PyTorch)

An educational implementation of Skip-gram Word2Vec with negative sampling
(Mikolov et al., 2013), trained on a small public-domain corpus, with a
Streamlit demo for exploring nearest neighbors. This is a re-implementation of
an established algorithm for learning purposes, not novel research.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

Public-domain books from Project Gutenberg (~1M words total). Not committed to
the repo; download them into `data/raw/`:

```bash
mkdir -p data/raw
for id in 11 1342 1661 2701 2600; do
  curl -L -o data/raw/pg$id.txt https://www.gutenberg.org/cache/epub/$id/pg$id.txt
done
```

(Alice in Wonderland, Pride and Prejudice, Sherlock Holmes, Moby Dick, War and Peace.)
`preprocess.py` strips the Gutenberg license header/footer automatically.

## Run the pipeline

```bash
python preprocess.py --window 2 --min-count 5   # -> data/processed/
python train.py                                  # -> models/run1/
streamlit run app.py                             # interactive demo
pytest                                           # tests
```

Before training is done, `python make_placeholder.py` creates a random-vector
model in the same format so the app can be tested (the app shows a warning
banner for it).

File formats are defined in [INTERFACE.md](INTERFACE.md).

## Methodology

1. **Preprocessing** (`preprocess.py`): lowercase; keep alphabetic words
   (inner apostrophes kept, e.g. `don't`); drop digits/punctuation. Words
   appearing fewer than `min_count` times are removed (reduces noise and
   vocabulary size). IDs are assigned by descending frequency. The corpus is
   treated as one continuous token stream.
2. **Skip-gram pairs**: each word is paired with every word within `window`
   positions on either side, e.g. with window 2, "king" in
   "the king ruled the kingdom" gives (king, the), (king, ruled), (king, the).
3. **Training** (`train.py`): two embedding tables (input and output). For each
   (target, context) pair the model raises σ(v_target · u_context) and lowers it
   for `neg_samples` random "negative" words drawn from the unigram^0.75
   distribution. The input table is saved as the word embeddings.
4. **Evaluation** (`evaluate.py`): cosine similarity between embeddings; top-k
   nearest neighbors.

## Results

_Fill in with real `most_similar` output after training. Paste
it verbatim; do not edit or cherry-pick without saying so._

| Setting | Value |
|---|---|
| Corpus tokens / vocab size / pairs | _from `data/processed/preprocess_meta.json`_ |
| Embedding dim, window, negatives, epochs, seed | _from `models/run1/config.json`_ |

## Limitations

- Small corpus (~1M words vs. billions for published embeddings): neighbors of
  rare words are noisy; results reflect 19th-century fiction, not modern English.
- Similarity reflects shared *contexts*, not identical meaning (antonyms often score high).
- No frequent-word subsampling or phrase detection ("new york" is two tokens).
- No quantitative benchmark (e.g. analogy or WordSim) was run.

## Optional experiment

Re-run with `--window 2` vs `--window 5` and compare neighbors for the same
words. Smaller windows tend to capture more syntactic similarity; larger ones
more topical similarity.

## Contributions

- **[Your name]**: `preprocess.py`, `app.py`, `make_placeholder.py`, tests, interface doc, README.
- **[Friend's name]**: `train.py` (Skip-gram model, negative sampling, training loop), `evaluate.py` (loading, cosine similarity, nearest neighbors).
