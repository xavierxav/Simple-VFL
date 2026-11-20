"""Stage 18 - VertiBench (224 px): parties that upload the features of two frozen encoders side by side.

Reuses the cache of s12 (f7_{name}_{split}.npy, [N, 16, d]).  Every feature is standardised with training statistics
(single encoders are re-run the same way for a fair comparison); MLP and linear heads, 5 seeds as s12.
  python s18_vb_combo.py   -> vb224_combo.json
"""

import json
import os

import numpy as np
import torch

from pe_common import DEV, evaluate_all, fmt, save_json, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
SETS = [["dinov2_g_224"], ["clip_l14_224"], ["clip_b16_224"], ["dinov2_b_224"],
        ["dinov2_g_224", "clip_l14_224"], ["clip_b16_224", "dinov2_b_224"], ["clip_l14_224", "clip_b16_224"]]


def main():
    ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    C = int(ytr.max()) + 1
    path = os.path.join(os.environ.get("PE_OUT", "/content/results"), "vb224_combo.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for names in SETS:
        key = "+".join(names)
        if key in res or not all(os.path.exists(f"{CACHE}/f7_{n}_test.npy") for n in names):
            continue
        A = {s: np.concatenate([np.load(f"{CACHE}/f7_{n}_{s}.npy").astype(np.float32) for n in names], -1) for s in ("train", "test")}
        m, sd = A["train"].mean(0, keepdims=True), A["train"].std(0, keepdims=True) + 1e-6  # per party and feature
        Xtr, Xte = [torch.from_numpy((A[s] - m) / sd).to(DEV) for s in ("train", "test")]
        res[key] = {"dim": int(Xtr.shape[-1])}
        for arch in ("mlp2", "linear"):
            runs = []
            for s in range(5):
                tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
                h = train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], C, arch=arch, seed=s)
                runs.append(evaluate_all(h, Xte, yte))
            res[key][arch] = summarize(runs)
            print(f"{key:28s} {arch:6s} {fmt(res[key][arch])}", flush=True)
        save_json("vb224_combo.json", res)
    save_json("vb224_combo.json", res)


if __name__ == "__main__":
    main()
