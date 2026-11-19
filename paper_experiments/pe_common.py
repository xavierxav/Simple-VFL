"""Shared harness for the paper experiments (frozen-feature heads on cached features).

Protocol copied from the repository's main pipeline so numbers stay comparable:
Adam lr=1e-3, batch 512, <=50 epochs, ReduceLROnPlateau(val loss, factor 0.3, patience 4),
early stopping on val loss (patience 10) with best-checkpoint restore, test at the best
validation-loss checkpoint. Head = Linear(K*d, 1000)-BN-ReLU-Linear(1000, C) ("mlp2") or
a single Linear ("linear"). Everything lives on the GPU: features are tiny.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.environ.get("PE_OUT", "/content/results")
os.makedirs(OUT, exist_ok=True)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_head(arch, in_dim, n_cls, hidden=1000):
    if arch == "linear":
        return nn.Linear(in_dim, n_cls)
    if arch == "mlp2":
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, n_cls))
    raise ValueError(arch)


def auc_macro_ovr(y, proba):
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < proba.shape[1]:  # class absent from the evaluated split
        keep = classes
        p = proba[:, keep]
        p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
        if len(keep) == 2:
            return float(roc_auc_score((y == keep[1]).astype(int), p[:, 1]))
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=keep))
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def _fwd(model, x, mask_fn=None):
    return model(x if mask_fn is None else mask_fn(x))


@torch.no_grad()
def predict_proba(model, X, bs=2048, transform=None):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i : i + bs]
        if transform is not None:
            xb = transform(xb)
        out.append(F.softmax(model(xb.flatten(1)), 1))
    return torch.cat(out)


def evaluate(model, X, y, transform=None):
    p = predict_proba(model, X, transform=transform)
    acc = (p.argmax(1) == y).float().mean().item()
    auc = auc_macro_ovr(y.cpu().numpy(), p.cpu().numpy())
    return acc, auc


def train_head(
    Xtr,
    ytr,
    Xva,
    yva,
    n_cls,
    arch="mlp2",
    hidden=1000,
    seed=0,
    epochs=50,
    lr=1e-3,
    bs=512,
    weight_decay=0.0,
    train_transform=None,
    class_weight=None,
    patience=10,
):
    """Train a head on [N, K, d] (or [N, D]) tensors already on DEV. Returns the best-val-loss model.

    ``train_transform`` is applied to each training batch (e.g. party dropout, view sampling,
    noise); the validation set is left untouched unless the caller pre-transforms it.
    """
    seed_all(seed)
    in_dim = Xtr[0].numel() if train_transform is None else train_transform(Xtr[:2]).flatten(1).shape[1]
    model = make_head(arch, in_dim, n_cls, hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEV)
    best, best_state, bad = float("inf"), None, 0
    n = len(Xtr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            xb = Xtr[idx]
            if train_transform is not None:
                xb = train_transform(xb)
            loss = F.cross_entropy(model(xb.flatten(1)), ytr[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(Xva), 2048):
                vl += F.cross_entropy(model(Xva[i : i + 2048].flatten(1)), yva[i : i + 2048], reduction="sum").item()
            vl /= len(Xva)
        sched.step(vl)
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def summarize(runs):
    """runs: list of dict(acc, auc) -> dict with mean/std in percent + raw lists."""
    acc = [r["acc"] * 100 for r in runs]
    auc = [r["auc"] * 100 for r in runs]
    a, sa = mean_std(acc)
    u, su = mean_std(auc)
    return {"acc": a, "acc_std": sa, "auc": u, "auc_std": su, "accs": acc, "aucs": auc, "n": len(runs)}


def save_json(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("saved", path, flush=True)


def fmt(s):
    return f"{s['acc']:.2f}±{s['acc_std']:.2f} / {s['auc']:.2f}±{s['auc_std']:.2f}"


def split_val(n, seed, frac=0.15, y=None):
    """Stratified train/val split indices (same recipe as the repository: 15% val)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    tr, va = train_test_split(idx, test_size=frac, random_state=seed, stratify=y)
    return tr, va


def robust_download(url, path, min_bytes=1):
    """HF's CDN often cuts long streams: use aria2c (multi-connection, resumable) and fall back to a curl resume loop."""
    import shutil
    import subprocess
    import time

    if os.path.exists(path) and os.path.getsize(path) >= min_bytes and not os.path.exists(path + ".aria2"):
        return path
    d, n = os.path.dirname(path), os.path.basename(path)
    os.makedirs(d, exist_ok=True)
    if shutil.which("aria2c") is None:
        subprocess.run("apt-get install -y -q aria2 > /dev/null 2>&1", shell=True)
    for attempt in range(30):
        if shutil.which("aria2c"):
            subprocess.run(f"aria2c -q -x 8 -s 8 -k 8M -c --max-tries=20 --retry-wait=2 --timeout=30 --file-allocation=none -d {d} -o {n} '{url}'", shell=True)
        else:
            subprocess.run(f"curl --http1.1 -L -C - -s -o {path} '{url}'", shell=True)
        if os.path.exists(path) and os.path.getsize(path) >= min_bytes and not os.path.exists(path + ".aria2"):
            return path
        print("download retry", attempt, url, flush=True)
        time.sleep(3)
    raise RuntimeError(f"could not download {url}")
