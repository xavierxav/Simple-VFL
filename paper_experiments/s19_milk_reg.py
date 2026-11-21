"""Stage 19 - MILK10k: why does late fusion beat the concatenation head?  Over-fitting, or modality competition?

Hypotheses: (H1) the concatenation MLP over-fits (3k training lesions, 2-3k input dimensions) -> regularisation or
augmentation should close the gap, and the gap should shrink with more training data; (H2) the joint head leans on the
dominant party (dermoscopy) and under-uses the clinical photograph -> a fixed equal-weight late fusion should already help.

  python s19_milk_reg.py views    GPU: features of the 8 dihedral transforms (4 rotations x flip) of every clinical and
                                  dermoscopic image, for each encoder (lesion photographs have no canonical orientation)
  python s19_milk_reg.py heads    -> milk_reg.json
Head variants (clinical + dermoscopic, DINOv2-g / CLIP-L / MONET, same folds and selection as s11/s17):
  late fusion with fixed weight 0.5 | concat + per-party PCA-256 | concat + input dropout 0.5 | concat + Gaussian
  feature noise (sigma 0.5 on standardised features) | concat + feature mixup (alpha 0.4) | learning curves (25/50/100 %
  of the training folds) for dermoscopy alone, concat and late fusion | image augmentation precomputed by the parties:
  (a) each party averages its 8 views and uploads one vector, (b) it uploads the 8 views and the head samples one per
  epoch (test: mean of the views).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from pe_common import DEV, fmt, metrics_from_proba, predict_proba, save_json, seed_all, summarize, train_head
from s11_milk10k import ROOT, SIZE, class_weights, folds, lesions, zscore

OUT = os.environ.get("PE_OUT", "/content/results")
ENCS = ["dinov2_g", "clip_l14", "monet"]
C = 7


def views(encs):
    from pe_encoders import encoder, extract

    imgs = {k: np.load(f"{ROOT}/img_{k}.npy", mmap_mode="r") for k in ("clin", "derm")}
    T = [(r, fl) for fl in (False, True) for r in range(4)]
    for name in encs:
        todo = [k for k in ("clin", "derm") if not os.path.exists(f"{ROOT}/f_{name}_{k}_d8.npy")]
        if not todo:
            continue
        fn, _ = encoder(name)
        for k in todo:
            outs = []
            for r, fl in T:
                outs.append(extract(fn, imgs[k], SIZE, crop=lambda x, r=r, fl=fl: torch.rot90(x.flip(-1) if fl else x, r, dims=(-2, -1))))
            a = np.stack(outs, 1)  # [N, 8, d]; view 0 = identity
            np.save(f"{ROOT}/f_{name}_{k}_d8.npy", a)
            print(name, k, a.shape, flush=True)
        del fn
        torch.cuda.empty_cache()


def late(pa, pb, w):
    pr = np.exp(w * np.log(pa + 1e-9) + (1 - w) * np.log(pb + 1e-9))
    return pr / pr.sum(1, keepdims=True)


def pca(X, tr, d):
    m = X[tr].mean(0)
    _, _, Vt = np.linalg.svd(X[tr] - m, full_matrices=False)
    Z = (X - m) @ Vt[:d].T
    return Z / (Z[tr].std(0) + 1e-6)


def train_mixup(X, y_t, tr, va, w, seed, alpha=0.4, epochs=50, bs=512, patience=10):
    import copy

    from pe_common import make_head

    seed_all(seed)
    m = make_head("mlp2", X.shape[1], C).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)
    g = torch.Generator().manual_seed(seed)
    trt, best, state, bad = torch.as_tensor(tr, device=DEV), 1e9, None, 0
    beta = torch.distributions.Beta(alpha, alpha)
    for _ in range(epochs):
        m.train()
        perm = trt[torch.randperm(len(tr), generator=g).to(DEV)]
        for i in range(0, len(perm), bs):
            j = perm[i : i + bs]
            if len(j) < 2:
                continue
            lam = beta.sample().item()
            k = j[torch.randperm(len(j), device=DEV)]
            o = m(lam * X[j] + (1 - lam) * X[k])
            loss = lam * F.cross_entropy(o, y_t[j], weight=w) + (1 - lam) * F.cross_entropy(o, y_t[k], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.cross_entropy(m(X[va]), y_t[va]).item()
        sched.step(vl)
        if vl < best:
            best, state, bad = vl, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    return m


def heads(encs):
    les = lesions()
    y = les["dx7"].to_numpy()
    Fo = folds(y)
    yt = torch.as_tensor(y.copy(), device=DEV)
    path = os.path.join(OUT, "milk_reg.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for enc in encs:
        if not os.path.exists(f"{ROOT}/f_{enc}_derm.npy"):
            continue
        P = {k: np.load(f"{ROOT}/f_{enc}_{k}.npy").astype(np.float32) for k in ("clin", "derm")}
        out = res.setdefault(enc, {})

        def record(key, runs):
            out[key] = summarize(runs)
            print(f"{enc:10s} {key:40s} {fmt(out[key])}", flush=True)
            save_json("milk_reg.json", res)

        def run(key, make_X, fit="plain", frac=1.0, transform=None, select="loss", wd=0.0):
            """make_X(tr) -> [N, D] numpy (or [N, V, D] with transform); returns per-fold (val, test) probabilities."""
            runs, probs = [], []
            for f, (tr, va, te) in enumerate(Fo):
                if frac < 1:
                    rng = np.random.default_rng(f)
                    tr = np.sort(rng.choice(tr, int(frac * len(tr)), replace=False))
                X = torch.as_tensor(make_X(tr), device=DEV)
                cw = class_weights(y[tr], C)
                if fit == "mixup":
                    m = train_mixup(X, yt, tr, va, torch.tensor(cw, device=DEV), seed=f)
                    Xv, Xt = X[va], X[te]
                elif transform is not None:  # X is [N, V, D]: training batches use a random view, val/test the mean of views
                    Xm = X.mean(1)
                    m = train_head(X[tr], yt[tr], Xm[va], yt[va], C, arch="mlp2", seed=f, class_weight=cw, train_transform=transform)
                    Xv, Xt = Xm[va], Xm[te]
                else:
                    tf = {"dropout": lambda xb: F.dropout(xb, 0.5, True), "noise": lambda xb: xb + 0.5 * torch.randn_like(xb)}.get(fit)
                    m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch="mlp2", seed=f, class_weight=cw, train_transform=tf, select=select, weight_decay=wd)
                    Xv, Xt = X[va], X[te]
                pv, pt = predict_proba(m, Xv).cpu().numpy(), predict_proba(m, Xt).cpu().numpy()
                probs.append((pv, pt))
                runs.append(metrics_from_proba(y[te], pt))
            if key not in out:
                record(key, runs)
            return probs

        def late_runs(key, pa, pb, fixed=None, frac=1.0):
            if key in out:
                return
            runs, ws = [], []
            for f, (tr, va, te) in enumerate(Fo):
                if fixed is None:
                    grid = np.linspace(0, 1, 21)
                    w = grid[int(np.argmax([f1_score(y[va], late(pa[f][0], pb[f][0], g).argmax(1), average="macro", labels=np.arange(C), zero_division=0) for g in grid]))]
                else:
                    w = fixed
                ws.append(float(w))
                runs.append(metrics_from_proba(y[te], late(pa[f][1], pb[f][1], w)))
            record(key, runs)
            out[key]["clin_weight"] = ws

        z = lambda k: (lambda tr: zscore(P[k], tr))  # noqa: E731
        cat = lambda tr: np.concatenate([zscore(P["clin"], tr), zscore(P["derm"], tr)], 1)  # noqa: E731
        pc, pd = run("clin", z("clin")), run("derm", z("derm"))
        run("concat", cat)
        late_runs("late fusion (weight on val)", pc, pd)
        late_runs("late fusion (fixed 0.5)", pc, pd, fixed=0.5)
        if os.environ.get("PE_REG_EXTRA"):  # same early-stopping rule for every head: weighted val loss, or val macro F1
            run("concat|rerun", cat)  # reference re-trained in the same run (features re-extracted on this VM)
            pcr, pdr = run("clin|rerun", z("clin")), run("derm|rerun", z("derm"))
            late_runs("late fusion (fixed 0.5)|rerun", pcr, pdr, fixed=0.5)
            for sel in ("wloss", "f1"):
                run(f"concat|stop {sel}", cat, select=sel)
                pcs, pds = run(f"clin|stop {sel}", z("clin"), select=sel), run(f"derm|stop {sel}", z("derm"), select=sel)
                late_runs(f"late fusion (fixed 0.5)|stop {sel}", pcs, pds, fixed=0.5)
            run("concat + weight decay 1e-3|stop f1", cat, select="f1", wd=1e-3)
        run("concat + PCA-256 per party", lambda tr: np.concatenate([pca(P["clin"], tr, 256), pca(P["derm"], tr, 256)], 1))
        run("concat + input dropout 0.5", cat, fit="dropout")
        run("concat + feature noise 0.5", cat, fit="noise")
        run("concat + feature mixup 0.4", cat, fit="mixup")
        for frac in (0.25, 0.5):
            pcf, pdf = run(f"clin|frac{frac}", z("clin"), frac=frac), run(f"derm|frac{frac}", z("derm"), frac=frac)
            run(f"concat|frac{frac}", cat, frac=frac)
            late_runs(f"late fusion (fixed 0.5)|frac{frac}", pcf, pdf, fixed=0.5)
        # image augmentation precomputed by the parties (8 dihedral views)
        if all(os.path.exists(f"{ROOT}/f_{enc}_{k}_d8.npy") for k in ("clin", "derm")):
            V = {k: np.load(f"{ROOT}/f_{enc}_{k}_d8.npy").astype(np.float32) for k in ("clin", "derm")}
            M = {k: V[k].mean(1) for k in V}
            zm = lambda k: (lambda tr: zscore(M[k], tr))  # noqa: E731
            pcm, pdm = run("clin|views mean", zm("clin")), run("derm|views mean", zm("derm"))
            run("concat|views mean", lambda tr: np.concatenate([zscore(M["clin"], tr), zscore(M["derm"], tr)], 1))
            late_runs("late fusion (fixed 0.5)|views mean", pcm, pdm, fixed=0.5)
            late_runs("late fusion (weight on val)|views mean", pcm, pdm)

            def zv(tr):  # standardise each party with the statistics of all its training views
                outv = []
                for k in ("clin", "derm"):
                    A = V[k]
                    mu, sd = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-6
                    outv.append((A - mu) / sd)
                return np.concatenate(outv, 2)  # [N, 8, Dc+Dd]

            def pick(xb):  # one random view per sample and party
                B, Vn, D = xb.shape
                h = D // 2 if P["clin"].shape[1] == P["derm"].shape[1] else None
                i1 = torch.randint(0, Vn, (B,), device=xb.device)
                i2 = torch.randint(0, Vn, (B,), device=xb.device)
                a = xb[torch.arange(B), i1]
                b = xb[torch.arange(B), i2]
                return torch.cat([a[:, :h], b[:, h:]], 1)

            run("concat|views sampled (x8 upload)", zv, transform=pick)
    save_json("milk_reg.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["views", "heads", "all"])
    ap.add_argument("--encoders", default=",".join(ENCS))
    a = ap.parse_args()
    encs = a.encoders.split(",")
    if a.stage in ("views", "all"):
        views(encs)
    if a.stage in ("heads", "all"):
        heads(encs)


if __name__ == "__main__":
    main()
