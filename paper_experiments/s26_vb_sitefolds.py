"""Stage 26 - VertiBench Satellite on site-grouped folds: frozen one-upload configurations fixed in advance vs SplitNN.

The official VertiBench split leaves sibling areas (same WorldStrat site) on both sides (stage 23).  Here all 3,927
areas (official train + test) are re-split into 5 folds, stratified by label and grouped by WorldStrat site (stage 23
mapping; an area without a confident match is its own group), so no test area has a sibling in training.  Nothing is
selected on these folds: the one-upload configurations are those the paper selected on the official split.
  frozen   per fold, 5 seeds (15 % stratified validation carved from the training folds, as s9/s12), MLP head:
             dinov2_g PCA-16 x 2 bit (64 B / sample), dinov2_g PCA-256 x 4 bit (2 KiB), dinov2_g full fp16 (48 KiB),
             clip_l14 full fp16 (32 KiB); s2.make_compressor (per-party PCA and quantiser fitted on the party's training rows)
  splitnn  per fold, the VertiBench SplitResNet recipe of stage 8 (Adam 1e-5, wd 1e-5, StepLR(10, 0.5), batch 32,
             50 epochs, x/255, no validation, last epoch); per epoch: train loss, test acc/AUC/F1/bacc, time; test
             probabilities every 10 epochs; messages and bytes as in split learning (16 x 512 fp16 up + down per sample).
  analysis (CPU, local) paired comparison on the out-of-fold predictions: per-fold differences and a 95 % bootstrap
             interval resampling sites (10,000 draws, seed 0); non-inferiority margin fixed before the runs: 1.0 point
             of accuracy (one upload non-inferior to SplitNN if the lower bound of the interval is above -1.0).

  python s26_vb_sitefolds.py folds                   (local)  -> paper_results/vb_site_folds.npz
  python s26_vb_sitefolds.py data|features|frozen     (GPU)    $PE_RAW raw files -> $PE_CACHE -> $PE_OUT/vb_site_frozen.json(+npz)
  python s26_vb_sitefolds.py splitnn --folds 0,2      (GPU)    -> $PE_OUT/vb_site_splitnn_f{f}.json (+ _probs.npz)
  python s26_vb_sitefolds.py analysis [RESULTS_DIR]   (local)  -> RESULTS_DIR/vb_site_analysis.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "paper_results")
CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
RAW = os.environ.get("PE_RAW", "/content/vb_raw/data/Vertibench")
FOLDS = os.environ.get("PE_SITE_FOLDS", os.path.join(RES, "vb_site_folds.npz"))
K = 5
MARGIN = 1.0  # non-inferiority margin (accuracy points), fixed before any site-fold result
CONFIGS = {  # name -> (encoder, PCA components, bits); full dimension = no PCA; 16 bits = half-precision cast
    "dinov2_g_pca16_2bit": ("dinov2_g_224", 16, 2),
    "dinov2_g_pca256_4bit": ("dinov2_g_224", 256, 4),
    "dinov2_g_full": ("dinov2_g_224", 1536, 16),
    "clip_l14_full": ("clip_l14_224", 1024, 16),
}


# ------------------------------------------------------------------ folds (local, CPU)
def folds():
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    sys.path.insert(0, HERE)
    from s23_audit_analysis import sites

    fp = np.load(os.path.join(RES, "vb_fingerprints.npz"))
    mp = np.load(os.path.join(RES, "vb_ws_mapping.npz"))
    aoi = pd.read_csv(os.path.join(RES, "ws_aoi.csv"))["AOI"].astype(str).to_numpy()
    site, _ = sites(aoi)
    y = np.concatenate([fp["y_train"], fp["y_test"]])
    ws = np.concatenate([mp["train"], mp["test"]])
    assert len(y) == len(ws) == 3927
    names = np.where(ws >= 0, site[np.clip(ws, 0, None)], np.array([f"unmatched{i}" for i in range(len(ws))]))
    _, group = np.unique(names, return_inverse=True)  # anonymous integer ids
    fold = np.full(len(y), -1, np.int8)
    for f, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=0).split(np.zeros(len(y)), y, group)):
        fold[te] = f
    assert (fold >= 0).all()
    for f in range(K):  # no site on both sides
        assert not set(group[fold == f]) & set(group[fold != f])
    n_tr = len(fp["y_train"])
    np.savez_compressed(FOLDS, fold_train=fold[:n_tr], fold_test=fold[n_tr:], group_train=group[:n_tr].astype(np.int32),
                        group_test=group[n_tr:].astype(np.int32), matched_train=ws[:n_tr] >= 0, matched_test=ws[n_tr:] >= 0)
    print("areas", len(y), "sites", group.max() + 1, "unmatched", int((ws < 0).sum()), "sizes", np.bincount(fold).tolist(),
          "labels per fold", [np.bincount(y[fold == f], minlength=4).tolist() for f in range(K)])


def load_folds():
    z = np.load(FOLDS)
    return np.concatenate([z["fold_train"], z["fold_test"]]).astype(int), np.concatenate([z["group_train"], z["group_test"]])


# ------------------------------------------------------------------ data + features (GPU machine)
def data():
    sys.path.insert(0, HERE)
    os.makedirs(CACHE, exist_ok=True)
    for sp in ("train", "test"):
        if not os.path.exists(f"{CACHE}/labels_{sp}.npy"):
            np.save(f"{CACHE}/labels_{sp}.npy", np.load(f"{RAW}/label_{sp}.npy").astype(np.int64))
    fp = np.load(os.path.join(RES, "vb_fingerprints.npz"))  # same areas in the same order as the fold file
    for sp in ("train", "test"):
        assert (np.load(f"{CACHE}/labels_{sp}.npy") == fp[f"y_{sp}"]).all(), sp
    from s3_baselines import pack_full

    for sp in ("train", "test"):
        pack_full(sp)


def features():
    sys.path.insert(0, HERE)
    from s12_vb224 import extract

    print(json.dumps(extract(sorted({c[0] for c in CONFIGS.values()}))), flush=True)


def all_labels():
    return np.concatenate([np.load(f"{CACHE}/labels_train.npy"), np.load(f"{CACHE}/labels_test.npy")])


# ------------------------------------------------------------------ frozen one-upload heads
def frozen():
    import torch

    sys.path.insert(0, HERE)
    from pe_common import DEV, evaluate, metrics_from_proba, predict_proba, save_json, split_val, summarize, train_head
    from s2_vertibench_heads import make_compressor

    fold, _ = load_folds()
    y = all_labels()
    yt = torch.as_tensor(y, device=DEV)
    out_json = os.path.join(os.environ.get("PE_OUT", "/content/results"), "vb_site_frozen.json")
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    probs = {}
    npz = out_json.replace(".json", "_probs.npz")
    if os.path.exists(npz):
        probs = dict(np.load(npz))
    feats = {}
    for name, (enc, d, bits) in CONFIGS.items():
        if name in res and name in probs:
            continue
        if enc not in feats:
            feats = {enc: torch.from_numpy(np.concatenate([np.load(f"{CACHE}/f7_{enc}_{s}.npy") for s in ("train", "test")])).to(DEV).float()}
        X = feats[enc]
        P = np.zeros((5, len(y), 4), np.float32)  # out-of-fold test probabilities, one row per seed
        per_fold, t0 = [], time.time()
        for f in range(K):
            tr_all, te = np.where(fold != f)[0], np.where(fold == f)[0]
            runs, vals = [], []
            for s in range(5):
                a, b = split_val(len(tr_all), s, 0.15, y[tr_all])
                tr, va = tr_all[a], tr_all[b]
                A, V, T = make_compressor(d, bits)(X[tr], X[va], X[te], s)
                m = train_head(A, yt[tr], V, yt[va], 4, seed=s)
                p = predict_proba(m, T).cpu().numpy()
                P[s, te] = p
                runs.append(metrics_from_proba(y[te], p))
                vals.append(evaluate(m, V, yt[va])[0] * 100)
            per_fold.append(dict(summarize(runs), val_acc=float(np.mean(vals)), n_test=int(len(te))))
            print(f"{name} fold {f}: acc {per_fold[-1]['acc']:.2f} f1 {per_fold[-1]['f1']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        oof = [metrics_from_proba(y, P[s]) for s in range(5)]
        res[name] = {"encoder": enc, "d": d, "bits": bits, "bytes_per_sample": 16 * d * bits / 8, "folds": per_fold,
                     "oof_per_seed": summarize(oof), "secs": time.time() - t0}
        probs[name] = P
        save_json("vb_site_frozen.json", res)
        np.savez_compressed(npz, y=y, fold=fold, **probs)


# ------------------------------------------------------------------ SplitNN (stage 8 recipe) per fold
def splitnn(fold_ids, epochs=50):
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, HERE)
    from pe_common import DEV, metrics_from_proba, save_json, seed_all
    from s8_vertibench_repro import SplitResNet

    fold, _ = load_folds()
    y = all_labels()
    A = [np.load(f"{CACHE}/full13_{s}.npy", mmap_mode="r") for s in ("train", "test")]
    n_tr = len(A[0])
    out_dir = os.environ.get("PE_OUT", "/content/results")

    def gather(idx):  # global area indices (train areas first, then test areas) -> uint8 [B,16,13,H,W] on DEV
        idx = np.asarray(idx)
        order = np.argsort(idx)
        s = idx[order]
        parts = [A[0][s[s < n_tr]], A[1][s[s >= n_tr] - n_tr]]
        xb = np.empty((len(idx),) + A[0].shape[1:], np.uint8)
        xb[order] = np.concatenate([p for p in parts if len(p)])
        return torch.from_numpy(xb).to(DEV, non_blocking=True)

    def forward(model, idx):
        xb = gather(idx)
        with torch.autocast("cuda", dtype=torch.float16):
            return model([(xb[:, p].float() / 255).contiguous(memory_format=torch.channels_last) for p in range(16)])

    @torch.no_grad()
    def predict(model, te):
        model.eval()
        P = [F.softmax(forward(model, te[i : i + 128]).float(), 1) for i in range(0, len(te), 128)]
        return torch.cat(P).cpu().numpy()

    for f in fold_ids:
        path = os.path.join(out_dir, f"vb_site_splitnn_f{f}.json")
        if os.path.exists(path) and json.load(open(path)).get("done"):
            continue
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        ytr = torch.as_tensor(y[tr], device=DEV)
        seed_all(f)
        model = SplitResNet().to(DEV).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        scaler = torch.amp.GradScaler()
        g = torch.Generator().manual_seed(f)
        n_batches = int(np.ceil(len(tr) / 32))
        curve, probs, t0 = [], {}, time.time()
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr), generator=g).numpy()
            tot = 0.0
            for i in range(0, len(perm), 32):
                b = perm[i : i + 32]
                loss = F.cross_entropy(forward(model, tr[b]).float(), ytr[b])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(b)
            sched.step()
            t_train = time.time() - t0
            p = predict(model, te)
            m = metrics_from_proba(y[te], p)
            curve.append({"epoch": ep + 1, "train_loss": tot / len(tr), **{f"test_{k}": v * 100 for k, v in m.items()},
                          "secs": time.time() - t0, "secs_train_only": t_train,
                          # sequential messages: every training batch is one upload (16 x 512 activations) and one
                          # gradient download; the test pass of this epoch is one more upload
                          "messages_train": 2 * n_batches * (ep + 1), "bytes_train": 2 * len(tr) * (ep + 1) * 16 * 512 * 2})
            if (ep + 1) % 10 == 0 or ep + 1 == epochs:
                probs[f"ep{ep + 1}"] = p.astype(np.float32)
            print(f"fold {f} ep {ep + 1}/{epochs} loss {tot / len(tr):.4f} test acc {m['acc'] * 100:.2f} f1 {m['f1'] * 100:.2f} ({time.time() - t0:.0f}s)", flush=True)
            out = {"fold": f, "seed": f, "n_train": int(len(tr)), "n_test": int(len(te)), "curve": curve, "gpu": torch.cuda.get_device_name(),
                   "recipe": "s8 SplitResNet: Adam 1e-5 wd 1e-5, StepLR(10,0.5), bs 32, x/255, no val, last epoch", "done": ep + 1 == epochs}
            save_json(f"vb_site_splitnn_f{f}.json", out)
            np.savez_compressed(path.replace(".json", "_probs.npz"), te=te, y=y[te], **probs)
        del model, opt
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ analysis (local)
def analysis(rdir):
    from sklearn.metrics import f1_score

    fold, group = load_folds()
    fz = np.load(os.path.join(rdir, "vb_site_frozen_probs.npz"))
    y = fz["y"]
    sn = np.full((len(y), 4), np.nan, np.float32)
    sn_folds = []
    for f in range(K):
        js = json.load(open(os.path.join(rdir, f"vb_site_splitnn_f{f}.json")))
        assert js["done"], f
        z = np.load(os.path.join(rdir, f"vb_site_splitnn_f{f}_probs.npz"))
        sn[z["te"]] = z[f"ep{len(js['curve'])}"]
        sn_folds.append(js["curve"][-1])
    assert np.isfinite(sn).all()
    c_sn = (sn.argmax(1) == y).astype(float)
    rng = np.random.default_rng(0)
    G = group.max() + 1
    members = [np.where(group == g)[0] for g in range(G)]
    draws = [np.concatenate([members[g] for g in rng.integers(0, G, G)]) for _ in range(10000)]

    def f1(pred, idx):
        return f1_score(y[idx], pred[idx], average="macro", labels=np.arange(4), zero_division=0) * 100

    out = {"margin_acc_points": MARGIN, "n_areas": int(len(y)), "n_sites": int(G), "bootstrap": "sites resampled with replacement, 10,000 draws, seed 0",
           "splitnn": {"oof_acc": c_sn.mean() * 100, "oof_f1": f1(sn.argmax(1), np.arange(len(y))),
                       "fold_acc": [c["test_acc"] for c in sn_folds], "fold_f1": [c["test_f1"] for c in sn_folds]}}
    for name in CONFIGS:
        P = fz[name]  # [seeds, N, 4]
        c = (P.argmax(2) == y).mean(0)  # per-area correctness averaged over the seeds (expected single-run accuracy)
        d_acc = c - c_sn
        boot = np.array([d_acc[ix].mean() * 100 for ix in draws])
        pred_s = P.argmax(2)
        f1_one = np.mean([f1(pred_s[s], np.arange(len(y))) for s in range(len(P))])
        boot_f1 = np.array([np.mean([f1(pred_s[s], ix) for s in range(len(P))]) - f1(sn.argmax(1), ix) for ix in draws[:2000]])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"oof_acc": c.mean() * 100, "oof_f1_mean_seeds": f1_one, "delta_acc": d_acc.mean() * 100, "delta_acc_ci95": [lo, hi],
                     "delta_f1": f1_one - out["splitnn"]["oof_f1"], "delta_f1_ci95": np.percentile(boot_f1, [2.5, 97.5]).tolist(),
                     "delta_f1_bootstrap_draws": 2000,
                     "fold_delta_acc": [float(c[fold == f].mean() * 100 - c_sn[fold == f].mean() * 100) for f in range(K)],
                     "folds_ahead": int(sum(c[fold == f].mean() > c_sn[fold == f].mean() for f in range(K))),
                     "non_inferior": bool(lo > -MARGIN), "superior": bool(lo > 0)}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "fold_delta_acc"}, default=float), flush=True)
    json.dump(out, open(os.path.join(rdir, "vb_site_analysis.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["folds", "data", "features", "frozen", "splitnn", "analysis"])
    ap.add_argument("rdir", nargs="?", default=RES)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    if a.stage == "folds":
        folds()
    elif a.stage == "data":
        data()
    elif a.stage == "features":
        features()
    elif a.stage == "frozen":
        frozen()
    elif a.stage == "splitnn":
        splitnn([int(x) for x in a.folds.split(",")], a.epochs)
    else:
        analysis(a.rdir)


if __name__ == "__main__":
    main()
