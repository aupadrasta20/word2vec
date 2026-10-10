import numpy as np
import json
import os

def cosine_similarity(a,b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))

def load_model(model_dir):
    emb = np.load(os.path.join(model_dir, "embeddings.npy"))

    with open(os.path.join(model_dir, "vocab.json")) as f:
        vocab = json.load(f)

    id2word = vocab["id2word"]
    word2id = {w: i for i, w in enumerate(id2word)}

    if len(id2word) != emb.shape[0]:
        raise ValueError(f"vocab has {len(id2word)} words but embeddings have {emb.shape[0]} rows")
    return emb, word2id, id2word


def most_similar(word, emb, word2id:dict, id2word:list, k=10):
    if word not in word2id:
        raise KeyError(f"'{word}' is not in the vocabulary")
    
    norms = np.linalg.norm(emb, axis=1, keepdims=True)

    emb_norm = emb/np.maximum(norms, 1e-8)

    q = emb_norm[word2id[word]]
    scores = emb_norm @ q

    order = np.argsort(-scores)   #word ids sorted by decreasing similarity
    results = []
    for idx in order:
        if idx == word2id[word]:
            continue              #skip word
        results.append((id2word[idx], float(scores[idx])))
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    id2word = ["king", "queen", "apple", "banana"]
    word2id = {w: i for i, w in enumerate(id2word)}
    emb = np.array([[1.0, 0.1],
                    [0.9, 0.2],
                    [0.1, 1.0],
                    [0.2, 0.9]])

    print(most_similar("king", emb, word2id, id2word, k=3))
    print(most_similar("apple", emb, word2id, id2word, k=3))

    emb, word2id, id2word = load_model("models/run1")
    print(emb.shape)
    print(most_similar("w1", emb, word2id, id2word, k=3))

    print(most_similar("dog", emb, word2id, id2word))