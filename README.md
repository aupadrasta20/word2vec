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

Public-domain books from Project Gutenberg (~1M words total). They are already
committed in `data/raw/`, so this step is only needed to re-download them:

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
python train.py                                  # -> models/run1/ (~30 s on a laptop CPU)
python evaluate.py holmes king -k 5              # nearest neighbors in the terminal
streamlit run app.py                             # interactive demo
pytest                                           # tests
```

The trained model in `models/run1/` (~4 MB) is committed, so you can skip
straight to `evaluate.py` or the app. `data/processed/` is not committed
(`pairs.npy` is ~65 MB); recreate it with `preprocess.py`.

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
   distribution. The input table is saved as the word embeddings. Optimized
   with Adam on shuffled mini-batches; seeds are fixed for reproducibility.
4. **Evaluation** (`evaluate.py`): cosine similarity between embeddings; top-k
   nearest neighbors.

## Results

| Setting | Value |
|---|---|
| Corpus tokens / vocab size / pairs | 1,053,368 / 10,822 / 4,085,506 |
| Embedding dim, window, negatives, epochs, seed | 100, 2, 5, 3, 0 |
| Batch size, learning rate, device | 1024, 0.003 (Adam), CPU |
| Average loss per epoch | 2.4122 → 2.2513 → 2.1929 |

Training loss decreased every epoch (the starting loss is 6 × ln 2 ≈ 4.159,
since all predictions begin at 0.5). Output of
`python evaluate.py holmes king ship said good father -k 5`, verbatim
(the words were chosen before seeing results):

```
holmes:   sherlock 0.766, lestrade 0.680, wilson 0.645, bradstreet 0.606, watson 0.589
king:     queen 0.640, duchess 0.570, hatter 0.547, holder 0.547, landlord 0.506
ship:     boat 0.729, pequod 0.624, craft 0.603, junk 0.590, cruise 0.590
said:     answered 0.723, remarked 0.706, asked 0.695, replied 0.689, exclaimed 0.685
good:     bad 0.611, fine 0.557, humoured 0.515, natured 0.513, pleasant 0.512
father:   son 0.748, mother 0.735, sister 0.714, aunt 0.702, daughter 0.702
```

For reference, 10,000 random word pairs have a mean cosine similarity of 0.165
(99th percentile 0.465), so these neighbors are well above chance. Observations:
`said` and `father` give clean groups (speech verbs, family members); `holmes`
gives other characters from the same stories; `king`'s neighbors are mostly
*Alice in Wonderland* characters, reflecting this corpus rather than the general
meaning of "king"; `good`'s top neighbor is its antonym `bad` (shared contexts).
This is a qualitative check, not a benchmark.

## Limitations

- Small corpus (~1M words vs. billions for published embeddings): neighbors of
  rare words are noisy; results reflect 19th-century fiction, not modern English.
- Similarity reflects shared *contexts*, not identical meaning (antonyms often score high).
- No frequent-word subsampling or phrase detection ("new york" is two tokens).
- No quantitative benchmark (e.g. analogy or WordSim) was run.
- No lemmatization (i.e. whale, whales, whale's are all separate words)
- Words such as 'the' fail, because they have no meaningful neigbors (top: japan 0.52, swandam 0.49), since stopwords co-occur with everything and frequent-word subsampling isn’t applied.” Showing a failure makes the good results more believable.

## Optional experiment

Re-run with `--window 2` vs `--window 5` and compare neighbors for the same
words. Smaller windows tend to capture more syntactic similarity; larger ones
more topical similarity.

## Contributions

- **Ryan Liu**: `preprocess.py`, `app.py`, `make_placeholder.py`, tests, interface doc, README.
- **Akhil Upadrasta**: `train.py` (Skip-gram model, negative sampling, training loop [written with AI assistance, reviewed]), `evaluate.py` (loading, cosine similarity, nearest neighbors).
