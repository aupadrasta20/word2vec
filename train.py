import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import time
import numpy as np


def make_noise_dist(counts):
    counts = torch.tensor(counts, dtype = torch.float)
    weights = counts**0.75
    probs = weights/weights.sum()
    return probs
def sample_negatives(probs, batch_size, k):
    flat = torch.multinomial(probs, batch_size * k, replacement = True)
    return flat.view(batch_size, k)

def save_artifacts(model, vocab, config, losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)   # create models/run1/ if it doesn't exist

    emb = model.in_embed.weight.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "embeddings.npy"), emb.astype(np.float32))

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent = 2)
    with open(os.path.join(out_dir, "loss_history.json"), "w") as f:
        json.dump(losses, f, indent = 2)
    

    print(f"Saved to {out_dir}: embeddings {emb.shape}")

def load_data(data_dir):
    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab = json.load(f)                         

    pairs = np.load(os.path.join(data_dir, "pairs.npy"))
    V = len(vocab["id2word"])

    # Sanity checks: fail loudly if the format isn't what we agreed
    assert len(vocab["counts"]) == V, "counts and id2word have different lengths"
    assert pairs.ndim == 2 and pairs.shape[1] == 2, f"pairs should be (N, 2), got {pairs.shape}"
    assert pairs.min() >= 0, "a pair has a negative ID"          
    assert pairs.max() < V, "a pair has an ID >= vocab size"    

    print(f"Loaded vocab of {V:,} words and {len(pairs):,} pairs")
    return vocab, pairs

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim)
        self.out_embed = nn.Embedding(vocab_size, dim)

        nn.init.uniform_(self.in_embed.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        # center: (B,)   context: (B,)   negatives: (B, K)
        v = self.in_embed(center)    
        u_pos = self.out_embed(context)
        u_neg = self.out_embed(negatives)

        pos_score = (v * u_pos).sum(dim = 1)  
        neg_score = (v.unsqueeze(1) * u_neg).sum(dim = 2)  

        pos_loss = F.logsigmoid(pos_score)               
        neg_loss = F.logsigmoid(-1*neg_score).sum(dim=1)  

        loss = -(pos_loss + neg_loss).mean()
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train Skip-gram Word2Vec with negative sampling.")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--neg-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="models/run1")
    args = parser.parse_args()

    # Same seed -> same initial weights, shuffles and negatives -> same model.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vocab, pairs = load_data(args.data_dir)
    with open(os.path.join(args.data_dir, "preprocess_meta.json")) as f:
        preprocess_meta = json.load(f)
    V = len(vocab["id2word"])

    # Move everything to the device once, not on every batch.
    pairs = torch.from_numpy(pairs).to(device)
    probs = make_noise_dist(vocab["counts"]).to(device)

    model = SkipGramNS(vocab_size=V, dim=args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    N, B = len(pairs), args.batch_size
    num_batches = (N + B - 1) // B   # last batch may be smaller
    losses = []
    N, B = len(pairs), args.batch_size
    num_batches = (N + B - 1) // B   # last batch may be smaller
    losses = []
    train_start = time.time()                       # NEW
    for epoch in range(1, args.epochs + 1):
        start = time.time()
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        perm = torch.randperm(N, device=device)   # new random order each epoch
        total_loss = 0.0
        for step in range(num_batches):
            batch = pairs[perm[step * B:(step + 1) * B]]
            center, context = batch[:, 0], batch[:, 1]
            negatives = sample_negatives(probs, len(batch), args.neg_k)

            loss = model(center, context, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss is a per-pair mean, so weight by batch size for an exact epoch average
            total_loss += loss.item() * len(batch)
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step:>5}/{num_batches}  loss {loss.item():.4f}")

        avg = total_loss / N
        losses.append(avg)
        print(f"Epoch {epoch}/{args.epochs}: avg loss {avg:.4f} ({time.time() - start:.0f}s)")

    config = {
        "dim": args.dim,
        "neg_k": args.neg_k,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": device,
        "train_seconds": round(time.time() - train_start, 1),   # NEW
        "preprocess_meta": preprocess_meta,
    }
    # Row i of the embeddings must be word i of the vocab, or every lookup is wrong.
    assert model.in_embed.weight.shape[0] == len(vocab["id2word"])
    save_artifacts(model, vocab, config, losses, args.out_dir)


if __name__ == "__main__":
    main()
