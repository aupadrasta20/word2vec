import numpy as np


def cosine_similarity(a,b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))


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
