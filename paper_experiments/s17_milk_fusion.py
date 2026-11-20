"""Stage 17 - MILK10k: does the head decide whether the second party helps?  (companion of s16 for WorldStrat)

Reuses the frozen features of s11 ($PE_MILK).  Parties: clinical photo (A), dermoscopy (B), metadata (C).
For each encoder (and for parties that upload two encoders' features side by side):
  single-party heads (A, B), concat MLP (s11), late fusion (weighted mean of log-probabilities, weight on the
  validation fold), branch head (per-party projection + auxiliary per-party losses, with/without modality dropout),
  and the error-overlap diagnostic (both wrong vs. independent errors, oracle accuracy).
Same folds, class-balanced loss and model selection as s11.  -> milk_fusion.json

  python s17_milk_fusion.py [--encoders dinov2_g,clip_l14,monet,dinov2_g+clip_l14]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from pe_common import DEV, fmt, metrics_from_proba, predict_proba, save_json, summarize, train_head
from s11_milk10k import ROOT, class_weights, folds, lesions, meta_features, zscore
from s16_ws_fusion import BranchHead, fit_custom

OUT = os.environ.get("PE_OUT", "/content/results")


def load(enc, view):
    return np.concatenate([np.load(f"{ROOT}/f_{e}_{view}.npy").astype(np.float32) for e in enc.split("+")], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="dinov2_g,clip_l14,monet,dinov2_g+clip_l14,dinov2_g+monet")
    a = ap.parse_args()
    les = lesions()
    y = les["dx7"].to_numpy()
    C = 7
    Fo = folds(y)
    meta = meta_features(les)
    yt = torch.as_tensor(y, device=DEV)
    path = os.path.join(OUT, "milk_fusion.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for enc in a.encoders.split(","):
        if not all(os.path.exists(f"{ROOT}/f_{e}_derm.npy") for e in enc.split("+")):
            print("skip", enc, flush=True)
            continue
        P = {"clin": load(enc, "clin"), "derm": load(enc, "derm"), "meta": meta}
        out = res.setdefault(enc, {})

        def record(key, runs):
            out[key] = summarize(runs)
            print(f"{enc:18s} {key:34s} {fmt(out[key])}", flush=True)
            save_json("milk_fusion.json", res)

        prob = {}
        for parts in (["clin"], ["derm"], ["clin", "derm"], ["clin", "derm", "meta"]):
            key = "+".join(parts) + "|concat"
            runs, pv, pt = [], [], []
            for f, (tr, va, te) in enumerate(Fo):
                X = torch.as_tensor(np.concatenate([zscore(P[p], tr) for p in parts], 1), device=DEV)
                m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch="mlp2", seed=f, class_weight=class_weights(y[tr], C))
                pv.append(predict_proba(m, X[va]).cpu().numpy())
                pt.append(predict_proba(m, X[te]).cpu().numpy())
                runs.append(metrics_from_proba(y[te], pt[-1]))
            prob["+".join(parts)] = (pv, pt)
            if key not in out:
                record(key, runs)
        # late fusion of the two single-party heads (+ the metadata head trained on its own when available)
        key = "clin+derm|late fusion"
        if key not in out:
            runs, ws = [], []
            for f, (tr, va, te) in enumerate(Fo):
                la, lb = np.log(prob["clin"][0][f] + 1e-9), np.log(prob["derm"][0][f] + 1e-9)
                grid = np.linspace(0, 1, 21)
                vals = [f1_score(y[va], (w * la + (1 - w) * lb).argmax(1), average="macro", labels=np.arange(C), zero_division=0) for w in grid]
                w = grid[int(np.argmax(vals))]
                ws.append(float(w))
                pr = np.exp(w * np.log(prob["clin"][1][f] + 1e-9) + (1 - w) * np.log(prob["derm"][1][f] + 1e-9))
                runs.append(metrics_from_proba(y[te], pr / pr.sum(1, keepdims=True)))
            record(key, runs)
            out[key]["clin_weight"] = ws
        # error overlap of the single-party heads
        both, either, ea, eb = [], [], [], []
        for f, (tr, va, te) in enumerate(Fo):
            ra, rb = prob["clin"][1][f].argmax(1) == y[te], prob["derm"][1][f].argmax(1) == y[te]
            both.append(100 * np.mean(~ra & ~rb))
            either.append(100 * np.mean(ra | rb))
            ea.append(100 * np.mean(~ra))
            eb.append(100 * np.mean(~rb))
        out["diag clin vs derm"] = {"err_a": float(np.mean(ea)), "err_b": float(np.mean(eb)), "both_wrong": float(np.mean(both)),
                                    "oracle_acc": float(np.mean(either)), "both_wrong_if_independent": float(np.mean(ea) * np.mean(eb) / 100)}
        print(enc, "diag", out["diag clin vs derm"], flush=True)
        # branch heads
        for parts, pdrop in ((["clin", "derm"], 0.0), (["clin", "derm"], 0.3), (["clin", "derm", "meta"], 0.3)):
            key = "+".join(parts) + f"|branch aux md{pdrop}"
            if key in out:
                continue
            runs = []
            for f, (tr, va, te) in enumerate(Fo):
                Z = [zscore(P[p], tr) for p in parts]
                X = torch.as_tensor(np.concatenate(Z, 1), device=DEV)
                m = BranchHead([z.shape[1] for z in Z], C).to(DEV)
                w = torch.tensor(class_weights(y[tr], C), dtype=torch.float32, device=DEV)
                trt = torch.as_tensor(tr, device=DEV)
                K = len(parts)

                def loss_fn(idx, m=m, X=X, w=w, trt=trt, K=K):
                    j = trt[idx]
                    drop = None
                    if pdrop > 0:  # with probability pdrop, zero exactly one (random) party for this sample
                        hit = torch.rand(len(j), device=DEV) < pdrop
                        which = torch.randint(0, K, (len(j),), device=DEV)
                        drop = [(~(hit & (which == k))).float() for k in range(K)]
                    o, ax = m(X[j], drop=drop, aux=True)
                    return F.cross_entropy(o, yt[j], weight=w) + 0.5 * sum(F.cross_entropy(q, yt[j], weight=w) for q in ax)

                m = fit_custom(m, loss_fn, lambda m=m, X=X, va=va: m(X[va]), yt[va], seed=f, n=len(tr))
                with torch.no_grad():
                    runs.append(metrics_from_proba(y[te], m(X[te]).softmax(1).cpu().numpy()))
            record(key, runs)
    save_json("milk_fusion.json", res)


if __name__ == "__main__":
    main()
