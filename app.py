"""Streamlit demo: explore nearest neighbors in our trained Word2Vec embeddings.

Run:  streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

from evaluate import cosine_similarity, load_model, most_similar

MODEL_DIR = "models/run1"   # where train.py saves (see INTERFACE.md)

st.set_page_config(page_title="Word2Vec Explorer")
st.title("Word2Vec Skip-gram Explorer")


@st.cache_resource  # load once, not on every interaction
def get_model():
    emb, word2id, id2word = load_model(MODEL_DIR)
    config = {}
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return emb, word2id, id2word, config


try:
    emb, word2id, id2word, config = get_model()
except FileNotFoundError:
    st.error(f"No trained model found in `{MODEL_DIR}/`. Run training first.")
    st.stop()

if config.get("placeholder"):
    st.warning("Placeholder model: these are RANDOM vectors for testing the UI. "
               "Results are meaningless.")

st.caption(f"Vocabulary: {len(id2word):,} words · embedding dimension: {emb.shape[1]}")


def clean(word):
    """Match preprocess.py's normalization (vocabulary is lowercase)."""
    return word.strip().lower()


# --- Nearest neighbors -------------------------------------------------------
word = clean(st.text_input("Enter a word", value=""))
k = st.slider("Number of neighbors (k)", min_value=1, max_value=30, value=10)

if word:
    try:
        results = most_similar(word, emb, word2id, id2word, k)
    except KeyError:
        st.info(f"'{word}' is not in the vocabulary (it may be too rare in the corpus, "
                f"or never appear). Try a more common word.")
    else:
        df = pd.DataFrame(results, columns=["word", "cosine similarity"])
        df.index = range(1, len(df) + 1)
        st.table(df.style.format({"cosine similarity": "{:.3f}"}))

# --- Compare two words -------------------------------------------------------
with st.expander("Compare two words"):
    col1, col2 = st.columns(2)
    a, b = clean(col1.text_input("Word A")), clean(col2.text_input("Word B"))
    if a and b:
        missing = [w for w in (a, b) if w not in word2id]
        if missing:
            st.info(f"Not in the vocabulary: {', '.join(missing)}")
        else:
            score = cosine_similarity(emb[word2id[a]], emb[word2id[b]])
            st.metric("Cosine similarity", f"{score:.3f}")

# --- Explanation ---------------------------------------------------------------
with st.expander("What do these scores mean?"):
    st.markdown("""
- **Cosine similarity** measures the angle between two word vectors: 1 = same
  direction, 0 = unrelated (orthogonal), negative = opposite directions.
- Words get similar vectors when they appear in **similar contexts** in the
  training text. That often reflects related meaning, but it is *not* proof
  of synonymy: antonyms ("hot"/"cold") share contexts too.
- The model was trained on a small corpus, so neighbors of rare words are
  noisy and some results reflect the specific books rather than English in general.
""")

with st.expander("Model configuration"):
    st.json(config)
