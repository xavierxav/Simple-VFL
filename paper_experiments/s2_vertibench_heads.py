"""Stage 2 - head-only experiments on cached VertiBench features (fast, GPU).

usage: python s2_vertibench_heads.py backbones pareto parties noise      (any subset, default: all)
Each experiment writes $PE_OUT/vb_<name>.json. All numbers: mean/std over seeds, test = official VertiBench test.
"""

import os
import sys
import time

import numpy as np
import torch
from scipy.stats import norm

from pe_common import DEV, evaluate, fmt, save_json, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
SEEDS = [0, 1, 2, 3, 4]
BACKBONES = ["dinov2_g", "dinov2_s", "dinov2_b", "dinov2_l", "clip_l14", "resnet50_in1k", "dinov2_b_random", "dinov2_g_4comp", "ssl4eo_vits16"]
ytr_all = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
NCLS = int(ytr_all.max().item()) + 1
_cache = {}


def feats(b):
    if b not in _cache:
        _cache.clear()
        _cache[b] = tuple(
            torch.from_numpy(np.load(f"{CACHE}/feat_{b}_{s}.npy")).to(DEV).float() for s in ("train", "test")
        )
    return _cache[b]


def splits(seed):
    tr, va = split_val(len(ytr_all), seed, 0.15, ytr_all.cpu().numpy())
    return torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)


def run_seeds(Xall, Xte, arch="mlp2", seeds=SEEDS, tr_tf=None, prep=None, hidden=1000):
    """Train/evaluate one head per seed. ``prep(Xtr, Xva, Xte, seed)`` may transform the three sets
    using statistics of the training rows only (PCA, quantisation, noise...)."""
    runs = []
    for s in seeds:
        tr, va = splits(s)
        Xtr, Xva, Xt = Xall[tr], Xall[va], Xte
        if prep is not None:
            Xtr, Xva, Xt = prep(Xtr, Xva, Xt, s)
        m = train_head(Xtr, ytr_all[tr], Xva, ytr_all[va], NCLS, arch=arch, hidden=hidden, seed=s, train_transform=tr_tf)
        acc, auc = evaluate(m, Xt, yte)
        runs.append({"acc": acc, "auc": auc})
    return summarize(runs)


# ------------------------------------------------------------------ E1: backbones
def exp_backbones():
    res = {}
    for b in BACKBONES:
        if not os.path.exists(f"{CACHE}/feat_{b}_test.npy"):
            print("skip", b, flush=True)
            continue
        Xall, Xte = feats(b)
        D = Xall.shape[-1]
        t = time.time()
        r = {"dim": D, "bytes_per_party_float32": D * 4}
        r["vertical_mlp2"] = run_seeds(Xall, Xte, "mlp2")
        r["vertical_linear"] = run_seeds(Xall, Xte, "linear")
        # horizontal proxy: one random revisit per sample (training: fresh draw each batch; eval: fixed draw)
        g = torch.Generator(device="cpu").manual_seed(123)
        pick = lambda X: X[torch.arange(len(X), device=DEV), torch.randint(0, 16, (len(X),), generator=g).to(DEV)]  # noqa: E731
        rnd = lambda xb: xb[torch.arange(len(xb), device=DEV), torch.randint(0, 16, (len(xb),), device=DEV)]  # noqa: E731
        r["horizontal_1view_mlp2"] = run_seeds(
            Xall, pick(Xte), "mlp2", tr_tf=rnd, prep=lambda a, v, t_, s: (a, pick(v), t_)
        )
        r["secs"] = round(time.time() - t)
        res[b] = r
        print(b, "D", D, "| vertical mlp2", fmt(r["vertical_mlp2"]), "| linear", fmt(r["vertical_linear"]),
              "| 1view", fmt(r["horizontal_1view_mlp2"]), flush=True)
        save_json("vb_backbones.json", res)
    # solo parties for the reference backbone
    if "dinov2_g" in res:
        Xall, Xte = feats("dinov2_g")
        solo = []
        for p in range(16):
            solo.append(run_seeds(Xall[:, p : p + 1], Xte[:, p : p + 1], "mlp2", seeds=[0, 1, 2]))
            print("solo party", p, fmt(solo[-1]), flush=True)
        res["dinov2_g_solo_parties"] = solo
        save_json("vb_backbones.json", res)
    res["n_test"] = len(yte)
    save_json("vb_backbones.json", res)


# ------------------------------------------------------------------ E2: communication / accuracy Pareto
def party_quantizer(Ztr, bits):
    lo = torch.quantile(Ztr, 0.01, dim=0)
    hi = torch.quantile(Ztr, 0.99, dim=0)
    L = 2**bits
    step = (hi - lo).clamp_min(1e-6) / L

    def q(Z):
        i = ((Z - lo) / step).floor().clamp(0, L - 1)
        return lo + (i + 0.5) * step

    return q


def make_compressor(d, bits):
    def prep(Xtr, Xva, Xt, seed):
        D = Xtr.shape[-1]
        outs = [[], [], []]
        for p in range(16):
            tr_p = Xtr[:, p]
            if d < D:
                mu = tr_p.mean(0)
                _, _, Vh = torch.linalg.svd(tr_p - mu, full_matrices=False)
                proj = lambda X, mu=mu, Vh=Vh: (X - mu) @ Vh[:d].T  # noqa: E731
            else:
                proj = lambda X: X  # noqa: E731
            Ztr = proj(tr_p)
            if bits == 16:
                qf = lambda Z: Z.half().float()  # noqa: E731
            elif bits < 16:
                qf = party_quantizer(Ztr, bits)
            else:
                qf = lambda Z: Z  # noqa: E731
            for o, X in zip(outs, (Xtr, Xva, Xt)):
                o.append(qf(proj(X[:, p])))
        return tuple(torch.stack(o, 1) for o in outs)

    return prep


def exp_pareto():
    Xall, Xte = feats("dinov2_g")
    res = []
    for d in [1536, 512, 256, 128, 64, 32, 16]:
        for bits in [32, 16, 8, 4, 2, 1]:
            if d != 1536 and bits == 16:
                continue
            r = run_seeds(Xall, Xte, "mlp2", prep=make_compressor(d, bits))
            r.update({"d": d, "bits": bits, "bytes_per_sample": 16 * d * bits / 8})
            res.append(r)
            print(f"d={d:5d} bits={bits:2d} bytes/sample={r['bytes_per_sample']:8.0f}  {fmt(r)}", flush=True)
            save_json("vb_pareto.json", res)


# ------------------------------------------------------------------ E3: number of parties / missing parties
def exp_parties():
    Xall, Xte = feats("dinov2_g")
    out = {"subsets": [], "missing": []}
    rng = np.random.RandomState(0)
    for k in [1, 2, 4, 8, 16]:
        draws = [list(range(16))] if k == 16 else [sorted(rng.choice(16, k, replace=False).tolist()) for _ in range(3)]
        runs = []
        for sub in draws:
            r = run_seeds(Xall[:, sub], Xte[:, sub], "mlp2", seeds=[0, 1, 2])
            runs.append(r)
        accs = [a for r in runs for a in r["accs"]]
        aucs = [a for r in runs for a in r["aucs"]]
        s = {"k": k, "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "n": len(accs)}
        out["subsets"].append(s)
        print("k =", k, fmt(s), flush=True)
    for p_drop in [0.0, 0.25, 0.5]:
        tf = (lambda xb, p=p_drop: xb * (torch.rand(xb.shape[0], 16, 1, device=DEV) >= p).float() / (1 - p)) if p_drop > 0 else None
        for missing in [0, 1, 2, 4, 8, 12]:
            gen = torch.Generator(device="cpu").manual_seed(7)
            M = torch.ones(len(Xte), 16)
            for i in range(len(Xte)):
                M[i, torch.randperm(16, generator=gen)[:missing]] = 0
            Xm = Xte * M.to(DEV)[:, :, None]
            r = run_seeds(Xall, Xm, "mlp2", seeds=[0, 1, 2], tr_tf=tf)
            r.update({"train_party_dropout": p_drop, "missing_at_test": missing})
            out["missing"].append(r)
            print(f"train p_drop={p_drop} missing={missing:2d}  {fmt(r)}", flush=True)
        save_json("vb_parties.json", out)
    save_json("vb_parties.json", out)


# ------------------------------------------------------------------ E4: noise on the uploaded features
def gdp_epsilon(mu, delta=1e-5):
    """Exact (eps, delta) of a mu-Gaussian-DP mechanism (Dong, Roth, Su 2019)."""
    f = lambda e: norm.cdf(-e / mu + mu / 2) - np.exp(e) * norm.cdf(-e / mu - mu / 2) - delta  # noqa: E731
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def exp_noise():
    Xall, Xte = feats("dinov2_g")
    res = []
    for r_noise in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]:

        def prep(Xtr, Xva, Xt, seed, r=r_noise):
            if r == 0:
                return Xtr, Xva, Xt
            sd = Xtr.std(0, keepdim=True)  # per party / per dim, from the training rows only
            g = torch.Generator(device=DEV).manual_seed(1000 + seed)
            return tuple(X + r * sd * torch.randn(X.shape, device=DEV, generator=g) for X in (Xtr, Xva, Xt))

        s = run_seeds(Xall, Xte, "mlp2", prep=prep)
        # indicative local-DP reading: clip each 1536-d party vector to the median L2 norm C, isotropic noise of
        # the same average per-coordinate std -> mu-GDP with mu = 2C / sigma (replace-one sensitivity 2C)
        norms = Xall.flatten(0, 1).norm(dim=1)
        C = norms.median().item()
        sigma = r_noise * Xall.std(0).mean().item()
        s.update({"noise_ratio": r_noise, "C": C, "sigma_per_coord": sigma})
        if r_noise > 0:
            mu = 2 * C / sigma
            s["gdp_mu"], s["eps_delta1e-5"] = mu, gdp_epsilon(mu)
        res.append(s)
        print("noise ratio", r_noise, fmt(s), "eps" , s.get("eps_delta1e-5"), flush=True)
        save_json("vb_noise.json", res)


EXPS = {"backbones": exp_backbones, "pareto": exp_pareto, "parties": exp_parties, "noise": exp_noise}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for n in names:
        print("\n=====", n, "=====", flush=True)
        EXPS[n]()
