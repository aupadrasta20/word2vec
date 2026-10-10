import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
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


if __name__ == "__main__":
    torch.manual_seed(0)
    probs = make_noise_dist([100, 10, 1])
    print(probs)
    print(sample_negatives(probs, 2, 5))
    V, D, K = 10, 4, 5
    model = SkipGramNS(vocab_size=V, dim=D)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    center = torch.tensor([1, 2])
    context = torch.tensor([3, 4])
    losses = []
    for step in range(200):
        negatives = torch.randint(0, V, (2, K))   # fresh random negatives every step

        loss = model(center, context, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(step, loss.item())
            losses.append(loss.item())
            

    vocab = {"id2word": [f"w{i}" for i in range(V)], "counts": [1] * V}
    config = {"dim": D, "neg_k": K, "lr": 0.05, "steps": 200, "seed": 0}
    save_artifacts(model, vocab, config, losses, "models/run1")

    
