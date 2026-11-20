"""Stage 10 - party-symmetric heads: is the concatenation head the right reference?

Concatenating 16 x d features gives a first layer with 16*d*1000 weights for 2.7k training samples.  A head that pools
the parties (mean, or attention with a learned query: a minimal DeepSets / set-attention head) shares its weights
across parties, has 16x fewer parameters, and handles missing parties natively (pool over the present ones).
Reports, on the official VertiBench split (5 seeds): concat vs mean vs attention pooling for several encoders, with
and without per-party PCA-64, and accuracy with m parties missing at inference.   Output: $PE_OUT/vb_pooling.json
"""

import os

import numpy as np
import torch
import torch.nn as nn

import pe_common
from pe_common import DEV, evaluate, fmt, save_json, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
NCLS = int(ytr.max()) + 1


class PoolHead(nn.Module):
    """x: [B, 16*d] (flattened by train_head) -> view [B,16,d]; parties whose vector is all-zero are treated as missing."""

    def __init__(self, d, n_cls, mode, hidden=1000):
        super().__init__()
        self.d, self.mode = d, mode
        self.norm = nn.LayerNorm(d)
        if mode == "attention":
            self.q = nn.Parameter(torch.randn(d) / d**0.5)
            self.k = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, n_cls))

    def forward(self, x):
        z = x.view(len(x), -1, self.d)
        present = (z.abs().sum(-1) > 0).float()  # [B,P]
        z = self.norm(z)
        if self.mode == "mean":
            w = present / present.sum(1, keepdim=True).clamp_min(1)
        else:
            s = (self.k(z) @ self.q) / self.d**0.5
            s = s.masked_fill(present == 0, -1e9)
            w = torch.softmax(s, 1)
        return self.mlp((w[..., None] * z).sum(1))


def feats(name):
    return [torch.from_numpy(np.load(f"{CACHE}/f7_{name}_{s}.npy")).to(DEV).float() for s in ("train", "test")]


def run(Xtr, Xte, arch, test_mask=None, seeds=range(5)):
    runs = []
    for s in seeds:
        tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
        tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
        m = train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], NCLS, arch=arch, seed=s)
        T = Xte if test_mask is None else Xte * test_mask
        acc, auc = evaluate(m, T, yte)
        runs.append({"acc": acc, "auc": auc})
    return summarize(runs)


def main():
    import s2_vertibench_heads as s2

    orig = pe_common.make_head

    def make_head(arch, in_dim, n_cls, hidden=1000):
        if arch in ("mean", "attention"):
            return PoolHead(in_dim // 16, n_cls, arch, hidden)
        return orig(arch, in_dim, n_cls, hidden)

    pe_common.make_head = make_head
    res = {"official": {}, "missing": []}
    for name in ("dinov2_g_168", "dinov2_g_224", "clip_l14_224", "dinov2_s_168"):
        if not os.path.exists(f"{CACHE}/f7_{name}_test.npy"):
            continue
        Xtr, Xte = feats(name)
        for arch in ("mlp2", "mean", "attention"):
            r = run(Xtr, Xte, arch)
            res["official"][f"{name}/{arch}"] = r
            print(f"{name:14s} {arch:9s} {fmt(r)}", flush=True)
        save_json("vb_pooling.json", res)
    # PCA-64 then pooling
    Xtr, Xte = feats("dinov2_g_168")
    comp = s2.make_compressor(64, 32)
    for arch in ("mlp2", "mean"):
        runs = []
        for s in range(5):
            tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
            tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
            A, V, T = comp(Xtr[tr], Xtr[va], Xte, s)
            m = train_head(A, ytr[tr], V, ytr[va], NCLS, arch=arch, seed=s)
            acc, auc = evaluate(m, T, yte)
            runs.append({"acc": acc, "auc": auc})
        res["official"][f"dinov2_g_168_pca64/{arch}"] = summarize(runs)
        print("pca64", arch, fmt(res["official"][f"dinov2_g_168_pca64/{arch}"]), flush=True)
    # missing parties at inference (no imputation needed for the pooling heads)
    gen = torch.Generator(device="cpu").manual_seed(7)
    for m in (0, 4, 8, 12, 15):
        M = torch.ones(len(Xte), 16)
        for i in range(len(Xte)):
            M[i, torch.randperm(16, generator=gen)[:m]] = 0
        M = M.to(DEV)[:, :, None]
        for arch in ("mean", "attention"):
            r = run(Xtr, Xte, arch, test_mask=M)
            r.update({"arch": arch, "missing_at_test": m})
            res["missing"].append(r)
            print(f"missing {m:2d} {arch:9s} {fmt(r)}", flush=True)
        save_json("vb_pooling.json", res)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
