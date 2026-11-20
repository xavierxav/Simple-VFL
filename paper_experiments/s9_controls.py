"""Stage 9 - control experiments requested by the reviews (VertiBench, frozen features, heads only).

E1 regularisation: is the PCA gain only a regularisation effect?  Full 1536-d DINOv2-g features with weight decay /
   dropout in the head, against PCA-256.
E2 missing parties at inference: zero-filling, zero-filling + rescaling by 16/(16-m), training-mean imputation;
   heads trained with and without party dropout.
E3 number of parties and single parties with 5 seeds.
E4 the actual DP mechanism: clip every per-party vector to the median norm c and add isotropic N(0, sigma^2 I),
   sigma = 2c/mu, mu set for a per-upload eps in {1, 8, 64} at delta = 1e-5 (noise drawn once: a release);
   on the full 1536-d vector and after per-party PCA-64 / PCA-16. eps over the 16 uploads of the same location
   composes to mu * sqrt(16).
E5 paired comparison of CLIP ViT-L/14 and DINOv2 ViT-g/14 on the official test set: seed-averaged probabilities,
   paired bootstrap of the accuracy difference and exact McNemar test; same against the majority class.
E6 a Sentinel-2 foundation model on all 13 bands (SSL4EO-S12 ViT-S/16, DINO weights, torchgeo), best effort.
Features: stage-7 cache (f7_*), extracted on demand.  Output: $PE_OUT/vb_controls.json
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest

from pe_common import DEV, evaluate, evaluate_all, fmt, predict_proba, save_json, seed_all, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
SEEDS = [0, 1, 2, 3, 4]
ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
NCLS = int(ytr.max()) + 1


def feats(name):
    return [torch.from_numpy(np.load(f"{CACHE}/f7_{name}_{s}.npy")).to(DEV).float() for s in ("train", "test")]


def splits(s):
    tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
    return torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)


def run(Xtr_all, Xte, seeds=SEEDS, prep=None, test_tf=None, keep_probs=False, **kw):
    runs, probs, vals = [], [], []
    for s in seeds:
        tr, va = splits(s)
        A, V, T = Xtr_all[tr], Xtr_all[va], Xte
        if prep is not None:
            A, V, T = prep(A, V, T, s)
        m = train_head(A, ytr[tr], V, ytr[va], NCLS, seed=s, **kw)
        Tt = T if test_tf is None else test_tf(T, A)
        runs.append(evaluate_all(m, Tt, yte))
        vals.append(evaluate(m, V, ytr[va])[0] * 100)
        if keep_probs:
            probs.append(predict_proba(m, Tt).cpu().numpy())
    out = summarize(runs)
    out["val_acc"], out["val_accs"] = float(np.mean(vals)), vals
    if keep_probs:
        out["probs"] = np.mean(probs, 0)
    return out


# ------------------------------------------------------------------ E1
def make_head_dropout(in_dim, n_cls, hidden=1000, p=0.5):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_cls))


def e1(G):
    import pe_common
    import s2_vertibench_heads as s2

    Xtr, Xte = G
    res = {}
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        res[f"full_wd{wd:g}"] = run(Xtr, Xte, weight_decay=wd)
        print("E1 full wd", wd, fmt(res[f"full_wd{wd:g}"]), flush=True)
    orig = pe_common.make_head
    pe_common.make_head = lambda arch, i, c, h=1000: make_head_dropout(i, c, h) if arch == "mlp2drop" else orig(arch, i, c, h)
    res["full_dropout0.5"] = run(Xtr, Xte, arch="mlp2drop")
    res["full_dropout0.5_wd1e-3"] = run(Xtr, Xte, arch="mlp2drop", weight_decay=1e-3)
    pe_common.make_head = orig
    def rotation(A, V, T, seed):  # per-party PCA keeping all 1536 components: a rotation, no truncation
        out = [[], [], []]
        for p in range(16):
            mu = A[:, p].mean(0)
            _, _, Vh = torch.linalg.svd(A[:, p] - mu, full_matrices=False)
            for o, X in zip(out, (A, V, T)):
                o.append((X[:, p] - mu) @ Vh.T)
        return tuple(torch.stack(o, 1) for o in out)

    def randproj(d):  # per-party Gaussian random projection to d dimensions (no data-dependent basis)
        def prep(A, V, T, seed):
            g = torch.Generator(device=DEV).manual_seed(500)
            R = torch.randn(16, A.shape[-1], d, device=DEV, generator=g) / d**0.5
            return tuple(torch.einsum("npd,pde->npe", X, R) for X in (A, V, T))

        return prep

    res["pca1536_rotation"] = run(Xtr, Xte, prep=rotation)
    res["randproj256"] = run(Xtr, Xte, prep=randproj(256))
    res["randproj64"] = run(Xtr, Xte, prep=randproj(64))
    for d in (256, 64):
        res[f"pca{d}"] = run(Xtr, Xte, prep=s2.make_compressor(d, 32))
        res[f"pca{d}_wd1e-3"] = run(Xtr, Xte, prep=s2.make_compressor(d, 32), weight_decay=1e-3)
    for k, v in res.items():
        print("E1", k, fmt(v), flush=True)
    return res


# ------------------------------------------------------------------ E2
def e2(G):
    Xtr, Xte = G
    res = []
    gen = torch.Generator(device="cpu").manual_seed(7)
    masks = {}
    for m in (0, 1, 2, 4, 8, 12):
        M = torch.ones(len(Xte), 16)
        for i in range(len(Xte)):
            M[i, torch.randperm(16, generator=gen)[:m]] = 0
        masks[m] = M.to(DEV)[:, :, None]
    for p_drop in (0.0, 0.25, 0.5):
        tf = (lambda xb, p=p_drop: xb * (torch.rand(xb.shape[0], 16, 1, device=DEV) >= p).float() / (1 - p)) if p_drop > 0 else None
        for m, M in masks.items():
            for imp in ("zero", "zero_rescale", "train_mean"):
                if m == 0 and imp != "zero":
                    continue

                def test_tf(T, A, M=M, imp=imp, m=m):
                    if imp == "zero":
                        return T * M
                    if imp == "zero_rescale":
                        return T * M * 16 / (16 - m)
                    return T * M + (1 - M) * A.mean(0, keepdim=True)  # per-party training mean

                r = run(Xtr, Xte, test_tf=test_tf, train_transform=tf)
                r.update({"train_party_dropout": p_drop, "missing_at_test": m, "imputation": imp})
                res.append(r)
                print(f"E2 p_drop={p_drop} missing={m:2d} {imp:12s} {fmt(r)}", flush=True)
    return res


# ------------------------------------------------------------------ E3
def e3(G):
    Xtr, Xte = G
    out = {"subsets": [], "solo": []}
    rng = np.random.RandomState(0)
    for k in (1, 2, 4, 8, 16):
        draws = [list(range(16))] if k == 16 else [sorted(rng.choice(16, k, replace=False).tolist()) for _ in range(3)]
        accs, aucs = [], []
        for sub in draws:
            r = run(Xtr[:, sub], Xte[:, sub])
            accs += r["accs"]
            aucs += r["aucs"]
        out["subsets"].append({"k": k, "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "n": len(accs)})
        print("E3 k", k, fmt(out["subsets"][-1]), flush=True)
    for p in range(16):
        out["solo"].append(run(Xtr[:, p : p + 1], Xte[:, p : p + 1]))
    print("E3 solo", [round(s["acc"], 1) for s in out["solo"]], flush=True)
    return out


# ------------------------------------------------------------------ E4
def gdp_mu(eps, delta=1e-5):
    from s4_inversion import gdp_mu as f

    return f(eps, delta)


def e4(G):
    import s2_vertibench_heads as s2

    Xtr, Xte = G
    D = Xtr.shape[-1]
    res = []
    for d in (D, 64, 16):
        comp = s2.make_compressor(d, 32) if d < D else None
        for eps in (None, 64, 8, 1):

            def prep(A, V, T, seed, comp=comp, eps=eps):
                if comp is not None:
                    A, V, T = comp(A, V, T, seed)
                if eps is None:
                    return A, V, T
                C = A.flatten(0, 1).norm(dim=1).median()
                clip = lambda X: X * (C / X.norm(dim=-1, keepdim=True).clamp_min(C))  # noqa: E731
                sigma = 2 * C / gdp_mu(eps)
                g = torch.Generator(device=DEV).manual_seed(1000 + seed)
                return tuple(clip(X) + sigma * torch.randn(X.shape, device=DEV, generator=g) for X in (A, V, T))

            r = run(Xtr, Xte, prep=prep)
            mu = gdp_mu(eps) if eps else None
            r.update({"d": d, "eps_upload": eps, "mu_upload": mu, "mu_16_uploads": mu * 4 if mu else None})
            res.append(r)
            print(f"E4 d={d:5d} eps={eps} {fmt(r)}", flush=True)
    return res


# ------------------------------------------------------------------ E5
def e5(names):
    y = yte.cpu().numpy()
    P = {}
    for n in names:
        Xtr, Xte = feats(n)
        P[n] = run(Xtr, Xte, keep_probs=True)["probs"]
    maj = np.bincount(ytr.cpu().numpy()).argmax()
    out = {"majority_acc": float((y == maj).mean() * 100)}
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(y), (10000, len(y)))
    for i, a in enumerate(names):
        ca = P[a].argmax(1) == y
        out[f"{a}_acc_avgprob"] = float(ca.mean() * 100)
        for b in names[i + 1 :]:
            cb = P[b].argmax(1) == y
            diff = (ca[B].mean(1) - cb[B].mean(1)) * 100
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            out[f"{a}_vs_{b}"] = {"diff": float((ca.mean() - cb.mean()) * 100), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "mcnemar_p": float(p), "n10": n10, "n01": n01}
            print("E5", a, "vs", b, out[f"{a}_vs_{b}"], flush=True)
    return out


# ------------------------------------------------------------------ E7
def e7(G):
    import s2_vertibench_heads as s2

    Xtr, Xte = G
    D = Xtr.shape[-1]
    res = []
    for d in (D, 512, 256, 128, 64, 32, 16):
        for bits in (16, 8, 4, 2, 1):
            r = run(Xtr, Xte, prep=s2.make_compressor(d, bits))  # bits=16: half-precision cast
            r.update({"d": d, "bits": bits, "bytes_per_sample": 16 * d * bits / 8})
            res.append(r)
            print(f"E7 d={d:5d} bits={bits:2d} val {r['val_acc']:.2f} test {fmt(r)}", flush=True)
    return res


# ------------------------------------------------------------------ E6
def e6():
    name = "ssl4eo_s2_vits16"
    if not all(os.path.exists(f"{CACHE}/f7_{name}_{s}.npy") for s in ("train", "test")):
        import subprocess
        import sys

        subprocess.run(f"{sys.executable} -m pip install -q torchgeo", shell=True)
        from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224

        from s7_vertibench_geo import MEAN, STD

        model = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO, num_classes=0).to(DEV).eval()  # pooled CLS features
        m, s = MEAN.to(DEV)[None, :, None, None], STD.to(DEV)[None, :, None, None]
        for split in ("train", "test"):
            A = np.load(f"{CACHE}/full13_{split}.npy", mmap_mode="r")
            out = None
            for i in range(0, len(A), 128):
                xb = torch.from_numpy(np.ascontiguousarray(A[i : i + 128])).to(DEV).float()
                for p in range(16):
                    x = F.interpolate((xb[:, p] - m) / s, size=(224, 224), mode="bilinear", align_corners=False)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        f = model(x).float().cpu().numpy().astype(np.float16)
                    if out is None:
                        out = np.zeros((len(A), 16, f.shape[1]), np.float16)
                    out[i : i + len(f), p] = f
            np.save(f"{CACHE}/f7_{name}_{split}.npy", out)
    Xtr, Xte = feats(name)
    res = {"dim": int(Xtr.shape[-1]), "mlp2": run(Xtr, Xte), "linear": run(Xtr, Xte, arch="linear")}
    print("E6 SSL4EO", fmt(res["mlp2"]), fmt(res["linear"]), flush=True)
    return res


def main():
    main_name = os.environ.get("PE_MAIN", "dinov2_g_168")
    paired = os.environ.get("PE_PAIRED", "clip_l14_224,dinov2_g_168,dinov2_g_224").split(",")
    if not all(os.path.exists(f"{CACHE}/f7_{n}_test.npy") for n in [main_name] + paired):
        from s7_vertibench_geo import extract_all

        extract_all()  # stage-7 encoders (cached when stage 7 already ran on this VM)
    G = feats(main_name)
    out = {}
    for name, fn in (("E1_regularisation", lambda: e1(G)), ("E3_parties", lambda: e3(G)), ("E2_missing", lambda: e2(G)),
                     ("E4_dp", lambda: e4(G)), ("E5_paired", lambda: e5(paired)),
                     ("E7_pareto_val", lambda: e7(G)), ("E6_ssl4eo", e6)):
        if os.environ.get("PE_SKIP", "") and name.split("_")[0] in os.environ["PE_SKIP"].split(","):
            continue
        try:
            out[name] = fn()
        except Exception as e:  # keep the other controls if one fails
            import traceback

            traceback.print_exc()
            out[name] = {"error": repr(e)}
        save_json(os.environ.get("PE_CONTROLS_OUT", "vb_controls.json"), out)
    print("DONE", flush=True)


if __name__ == "__main__":
    seed_all(0)
    main()
