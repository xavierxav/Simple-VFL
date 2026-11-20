"""Stage 13 - linear heads that can be trained on homomorphically encrypted uploads.

With frozen party features, the head can be linear.  Two encrypted-training protocols are then classical:
  (a) ridge regression on the concatenated (per-party PCA-compressed, standardised) features: the label owner only
      needs the feature Gram matrix G = X^T X and X^T Y.  Each party uploads its encrypted feature columns ONCE (CKKS);
      the label owner computes the cross-party blocks X_i^T X_j and X_i^T Y homomorphically (multiplicative depth 1,
      2 with class weights), masks them, has them decrypted by the key holders, and solves the ridge system in the
      clear: one upload plus one round trip of masked aggregates; no per-sample feature is ever decrypted.
  (b) iterative logistic regression on encrypted data (Kim et al. 2018, Han et al. 2019): degree-3 polynomial
      sigmoid, a fixed small number of full-batch Nesterov iterations, one-vs-rest.
This stage measures (plaintext emulation, exact up to CKKS rounding) the accuracy of (a) and (b) against the softmax
linear head and the MLP head, for several PCA sizes, on every data set whose features are cached; and (if TenSEAL is
available) the real CKKS cost of protocol (a): encryption, homomorphic Gram / X^T Y, upload size, and the decryption
error, on one complete configuration plus per-operation timings for extrapolation.
Output: he_heads.json (he_heads_milk.json / he_heads_ws.json on the MILK10k / WorldStrat VMs)
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from pe_common import DEV, metrics_from_proba, predict_proba, save_json, split_val, train_head

OUT = os.environ.get("PE_OUT", "/content/results")
SIG3 = (0.5, 0.15012, -0.0015930)  # least-squares degree-3 sigmoid on [-8, 8] (Kim et al. 2018)


# ------------------------------------------------------------------ data sets (parties, labels, splits)
def ds_vertibench(enc):
    c = os.environ.get("PE_CACHE", "/content/cache/vb")
    if not os.path.exists(f"{c}/f7_{enc}_test.npy"):
        return None
    tr = np.load(f"{c}/f7_{enc}_train.npy").astype(np.float32)
    te = np.load(f"{c}/f7_{enc}_test.npy").astype(np.float32)
    ytr, yte = np.load(f"{c}/labels_train.npy"), np.load(f"{c}/labels_test.npy")
    X = np.concatenate([tr, te])
    y = np.concatenate([ytr, yte])
    nt = len(ytr)
    splits = []
    for s in range(5):
        a, v = split_val(nt, s, 0.15, ytr)
        splits.append((a, v, np.arange(nt, len(y))))
    return [X[:, p] for p in range(X.shape[1])], y, splits, "acc"


def ds_milk(enc):
    r = os.environ.get("PE_MILK", "/content/milk")
    if not os.path.exists(f"{r}/f_{enc}_derm.npy"):
        return None
    import s11_milk10k as m

    les = m.lesions()
    y = les["dx7"].to_numpy()
    return [np.load(f"{r}/f_{enc}_clin.npy").astype(np.float32), np.load(f"{r}/f_{enc}_derm.npy").astype(np.float32)], y, m.folds(y), "f1"


def ds_generic(path):
    """npz with parties (list as p0, p1, ...), y, and fold assignment 'fold' (fold f test, f+1 val) or 'split' 0/1/2."""
    if not os.path.exists(path):
        return None
    z = np.load(path)
    parties = [z[f"p{i}"].astype(np.float32) for i in range(int(z["n_parties"]))]
    y = z["y"]
    if "fold" in z:
        f = z["fold"]
        k = int(f.max()) + 1
        splits = [(np.where((f != i) & (f != (i + 1) % k))[0], np.where(f == (i + 1) % k)[0], np.where(f == i)[0]) for i in range(k)]
    else:
        s = z["split"]
        tr_all, te = np.where(s < 2)[0], np.where(s == 2)[0]
        splits = []
        for seed in range(3):
            a, v = split_val(len(tr_all), seed, 0.1, y[tr_all])
            splits.append((tr_all[a], tr_all[v], te))
    return parties, y, splits, str(z["metric"]) if "metric" in z else "acc"


# ------------------------------------------------------------------ party-side preprocessing (no labels)
def party_prep(parties, tr, d):
    """Per-party PCA to d components (fitted on the party's own training rows) + standardisation."""
    out = []
    for P in parties:
        mu = P[tr].mean(0)
        Z = P - mu
        if d is not None and d < P.shape[1]:
            _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False)
            Z = Z @ Vt[:d].T
        sd = Z[tr].std(0) + 1e-6
        out.append((Z / sd).astype(np.float64))
    return np.concatenate(out, 1)


# ------------------------------------------------------------------ heads
def ridge_fit(X, Y, lam, w=None):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    Xw = Xb if w is None else Xb * w[:, None]
    G = Xw.T @ Xb
    R = lam * np.eye(G.shape[0])
    R[-1, -1] = 0
    return np.linalg.solve(G + R, Xw.T @ Y)


def ridge_scores(X, W):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def softmax_t(S, t):
    Z = S / t
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def sel_metric(y, P, metric):
    m = metrics_from_proba(y, P)
    return m["f1"] if metric == "f1" else m["acc"]


def head_ridge(X, y, tr, va, te, C, metric, balanced):
    Y = np.eye(C)[y] * 2 - 1
    w = None
    if balanced:
        cnt = np.bincount(y[tr], minlength=C).astype(float)
        w = (cnt.sum() / (C * np.maximum(cnt, 1)))[y[tr]]
    best = None
    scale = np.trace(X[tr].T @ X[tr]) / X.shape[1]
    for a in np.logspace(-5, 1, 13):
        W = ridge_fit(X[tr], Y[tr], a * scale, w)
        sv = ridge_scores(X[va], W)
        sc = sel_metric(y[va], softmax_t(sv, 0.1), metric)
        if best is None or sc > best[0]:
            best = (sc, W, a)
    _, W, a = best
    sv = ridge_scores(X[va], W)
    t = min((0.01, 0.03, 0.1, 0.3, 1.0), key=lambda t: -np.log(softmax_t(sv, t)[np.arange(len(va)), y[va]] + 1e-12).mean())
    return softmax_t(ridge_scores(X[te], W), t), {"lambda_rel": float(a), "temperature": t}


def sig3(z):
    z = np.clip(z, -8, 8)  # the polynomial is only fitted on [-8, 8]; HE implementations rescale inputs to keep it there
    return SIG3[0] + SIG3[1] * z + SIG3[2] * z**3


def head_polylogistic(X, y, tr, va, te, C, metric, iters=(7, 30), lrs=(0.1, 0.3, 1.0, 3.0)):
    """One-vs-rest logistic regression, full-batch Nesterov GD with the degree-3 sigmoid (encrypted-training emulation)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(C)[y]
    best = None
    for T in iters:
        for lr in lrs:
            W = np.zeros((Xb.shape[1], C))
            V = W.copy()
            for t in range(T):
                g = Xb[tr].T @ (sig3(Xb[tr] @ V) - Y[tr]) / len(tr)
                Wn = V - lr * g
                V = Wn + (t / (t + 3)) * (Wn - W)
                W = Wn
            S = Xb @ W
            sc = sel_metric(y[va], softmax_t(S[va], 1.0), metric)
            if best is None or sc > best[0]:
                best = (sc, S, T, lr)
    _, S, T, lr = best
    t = min((0.03, 0.1, 0.3, 1.0, 3.0), key=lambda t: -np.log(softmax_t(S[va], t)[np.arange(len(va)), y[va]] + 1e-12).mean())
    return softmax_t(S[te], t), {"iters": T, "lr": lr, "temperature": t}


def head_torch(X, y, tr, va, te, C, arch, seed, balanced):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    yt = torch.as_tensor(y, device=DEV)
    cw = None
    if balanced:
        cnt = np.bincount(y[tr], minlength=C).astype(float)
        cw = (cnt.sum() / (C * np.maximum(cnt, 1))).tolist()
    m = train_head(Xt[tr], yt[tr], Xt[va], yt[va], C, arch=arch, seed=seed, class_weight=cw)
    return predict_proba(m, Xt[te]).cpu().numpy(), {}


def evaluate_dataset(name, ds, dims=(None, 256, 64, 16)):
    parties, y, splits, metric = ds
    C = int(y.max()) + 1
    balanced = metric == "f1"
    res = {"metric": metric, "n_parties": len(parties), "dims": [int(p.shape[1]) for p in parties]}
    for d in dims:
        if d is not None and d >= min(p.shape[1] for p in parties):
            continue
        key = "full" if d is None else f"pca{d}"
        rows = {}
        for si, (tr, va, te) in enumerate(splits):
            X = party_prep(parties, tr, d)
            heads = {"mlp2": lambda: head_torch(X, y, tr, va, te, C, "mlp2", si, balanced),
                     "linear_softmax": lambda: head_torch(X, y, tr, va, te, C, "linear", si, balanced),
                     "ridge": lambda: head_ridge(X, y, tr, va, te, C, metric, balanced)}
            if X.shape[1] <= 4096:
                heads["polylogistic"] = lambda: head_polylogistic(X, y, tr, va, te, C, metric)
            for h, fn in heads.items():
                if d is None and h in ("ridge", "polylogistic") and X.shape[1] > 8192:
                    continue
                t = time.time()
                P, info = fn()
                r = metrics_from_proba(y[te], P)
                r.update(info)
                r["secs"] = time.time() - t
                rows.setdefault(h, []).append(r)
        res[key] = {}
        for h, runs in rows.items():
            agg = {}
            for k in ("acc", "auc", "f1", "bacc"):
                v = np.array([r[k] for r in runs]) * 100
                agg[k], agg[k + "_std"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            agg["info"] = [{k: v for k, v in r.items() if k not in ("acc", "auc", "f1", "bacc")} for r in runs]
            res[key][h] = agg
            print(f"{name:28s} {key:7s} {h:15s} acc {agg['acc']:.2f}±{agg['acc_std']:.2f}  F1 {agg['f1']:.2f}±{agg['f1_std']:.2f}  AUC {agg['auc']:.2f}", flush=True)
    return res


# ------------------------------------------------------------------ real CKKS cost of protocol (a)
def he_cost(parties, y, tr, d, C, n_time=32):
    """Encrypt the PCA-d columns of every party (column packing, chunks of 4096 samples), compute the cross-party Gram
    blocks and X^T Y homomorphically for a subset of entries (timed), check against plaintext, extrapolate."""
    import tenseal as ts

    X = party_prep(parties, tr, d)[tr]
    N, K = len(tr), len(parties)
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    slots = 4096
    chunks = [np.arange(i, min(i + slots, N)) for i in range(0, N, slots)]
    rng = np.random.default_rng(0)
    A = rng.integers(0, d, n_time)
    B = rng.integers(d, X.shape[1], n_time)
    need = sorted(set(A.tolist()) | set(B.tolist()))
    t = time.time()
    enc = {j: [ts.ckks_vector(ctx, X[ch, j].tolist()) for ch in chunks] for j in need}
    t_enc = (time.time() - t) / len(need)
    ct_bytes = len(enc[need[0]][0].serialize())
    t = time.time()
    errs = []
    for a, b in zip(A, B):
        v = enc[a][0].dot(enc[b][0])
        for k in range(1, len(chunks)):
            v = v + enc[a][k].dot(enc[b][k])
        ref = float(X[:, a] @ X[:, b])
        errs.append(abs(v.decrypt()[0] - ref) / (abs(ref) + 1e-9 * N))
    t_ctct = (time.time() - t) / n_time
    Y = np.eye(C)[y[tr]]
    t = time.time()
    na = min(8, n_time)
    for a in A[:na]:
        for c in range(C):
            v = enc[a][0].dot(Y[chunks[0], c].tolist())
            for k in range(1, len(chunks)):
                v = v + enc[a][k].dot(Y[chunks[k], c].tolist())
            ref = float(X[:, a] @ Y[:, c])
            errs.append(abs(v.decrypt()[0] - ref) / (abs(ref) + 1e-9 * N))
    t_ptct = (time.time() - t) / (na * C)
    D = K * d
    n_cross = K * (K - 1) // 2 * d * d
    return {"N": int(N), "parties": K, "d_per_party": d, "C": C, "chunks": len(chunks), "ciphertext_bytes": ct_bytes,
            "upload_bytes_per_party": int(d * len(chunks) * ct_bytes), "plain_fp16_bytes_per_party": int(N * d * 2),
            "secs_encrypt_per_column": t_enc, "secs_ctct_inner_product": t_ctct, "secs_ptct_inner_product": t_ptct,
            "max_rel_error": float(max(errs)), "n_cross_entries": n_cross, "n_xty_entries": D * C,
            "extrapolated_secs_server": n_cross * t_ctct + D * C * t_ptct, "extrapolated_secs_encrypt_per_party": d * t_enc,
            "aggregate_bytes_round_trip": int((D + 1) * (D + 1 + C) * 8 * 2)}


def main():
    # one file per VM, so that fetching the results of several VMs into paper_results/ does not overwrite them
    FN = os.environ.get("PE_HE_OUT") or ("he_heads_milk.json" if os.path.isdir("/content/milk") else "he_heads_ws.json" if os.path.isdir("/content/ws2") else "he_heads.json")
    path = os.path.join(OUT, FN)
    res = json.load(open(path)) if os.path.exists(path) else {}
    cands = {f"vertibench/{e}": (lambda e=e: ds_vertibench(e)) for e in ("dinov2_g_224", "clip_l14_224")}
    cands.update({f"milk10k/{e}": (lambda e=e: ds_milk(e)) for e in ("dinov2_g", "clip_l14")})
    for p in sorted(__import__("glob").glob(os.path.join(os.environ.get("PE_HE_DS", "/content/he_ds"), "he_ds_*.npz"))):
        cands[os.path.basename(p)[6:-4]] = (lambda p=p: ds_generic(p))
    for name, load in cands.items():
        if name in res and "pca64" in res[name]:
            continue
        ds = load()
        if ds is None:
            continue
        res[name] = evaluate_dataset(name, ds)
        save_json(FN, res)
    try:
        import tenseal  # noqa: F401
    except Exception:
        import subprocess
        import sys

        subprocess.run(f"{sys.executable} -m pip install -q tenseal", shell=True)
    try:
        res.setdefault("he_cost", {})
        for name, load, d in (("milk10k/dinov2_g", lambda: ds_milk("dinov2_g"), 64), ("vertibench/dinov2_g_224", lambda: ds_vertibench("dinov2_g_224"), 16),
                              ("vertibench/dinov2_g_224", lambda: ds_vertibench("dinov2_g_224"), 64)):
            ds = load()
            if ds is None or f"{name}/pca{d}" in res["he_cost"]:
                continue
            parties, y, splits, _ = ds
            res["he_cost"][f"{name}/pca{d}"] = he_cost(parties, y, splits[0][0], d, int(y.max()) + 1)
            print("HE cost", name, d, res["he_cost"][f"{name}/pca{d}"], flush=True)
            save_json(FN, res)
    except Exception as e:
        import traceback

        traceback.print_exc()
        res["he_cost_error"] = repr(e)
    save_json(FN, res)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
