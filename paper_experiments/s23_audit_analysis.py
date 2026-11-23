"""Stage 23 (CPU, local) - analysis of the third-review audit results.

Reads, from paper_results/: vb_fingerprints.npz and ws_fingerprints.npz + ws_aoi.csv (stage 22 / chain B1), the kept
test probabilities of the frozen heads (vb_frozen_probs.npz), of the reproduced VertiBench SplitNN
(vb_repro_splitresnet_preds_probs.npz) and of CLIP-B split learning (vb_split_clipb_probs.npz), the nearest public
training location of every test upload (vb_nnpublic.npz), and the out-of-fold WorldStrat probabilities
(ws_sensors_oof_grouped.npz).  Writes vb_audit_analysis.json:
  mapping   every VertiBench sample matched to its WorldStrat area by image fingerprints (band means and a 4x4 B4
            thumbnail of each revisit, matched by correlation), with the match score and gap;
  sibling   test samples that are tiles of a multi-tile site whose other tiles are in the training set;
  subsets   accuracy of every kept method on all test samples, on those with a sibling and on the others;
  paired    exact McNemar test and bootstrap interval of each frozen head against the SplitNN reference
            (seed-averaged probabilities, same test samples);
  nnpublic  how often the nearest public training upload is a sibling tile, and the label match without siblings;
  ws_f1     per-class F1 of the WorldStrat heads from the grouped out-of-fold predictions.
usage: python s23_audit_analysis.py [paper_results]
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "paper_results")


def sites(aoi):  # same rule as s15_worldstrat_sensors.sites
    aoi = pd.Series(aoi).astype(str)
    multi = aoi.str.match(r"^(Amnesty|ASMSpotter)")
    return np.where(multi, aoi.str.replace(r"-\d+-\d+$", "", regex=True), aoi), multi.values


def load(name):
    p = os.path.join(R, name)
    return dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else None


def match(vb, ws, seen=None, chunk=64, min_score=0.95, min_gap=0.05):
    """vb [n,16,29] (train and test together), ws [m,16,29] -> index into ws (-1: no confident match), score, gap to
    the second best area.  VertiBench rescales the brightness of every image (by about 1.7, not uniformly), so band
    means do not match; the 4x4 B4 thumbnail of each revisit, centred and normalised, does.  Score of a candidate area:
    mean over the VertiBench revisits of their best correlation with a revisit WorldStrat holds (the revisit order may
    differ).  Areas missing from the WorldStrat arrays (3 of the 3,928 are left out by stage 15) must not be forced onto
    a neighbour: matches below min_score or within min_gap of the second best area are rejected."""
    seen = np.ones(ws.shape[:2], bool) if seen is None else seen.astype(bool)

    def zn(x):
        x = x - x.mean(-1, keepdims=True)
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    V, W = zn(vb[..., 13:].astype(np.float32)), zn(ws[..., 13:].astype(np.float32))
    m = len(W)
    Wf = W.reshape(-1, W.shape[-1])
    mask = np.where(seen.reshape(-1), 0.0, -9.0).astype(np.float32)
    vok = np.abs(vb.astype(np.float32)).sum(-1) > 0  # revisits absent from VertiBench too (all-zero images) are ignored
    idx, score, gap = np.zeros(len(V), int), np.zeros(len(V)), np.zeros(len(V))
    for a in range(0, len(V), chunk):
        Vb = V[a : a + chunk]
        S = (Vb.reshape(-1, Vb.shape[-1]) @ Wf.T + mask).reshape(len(Vb), 16, m, 16).max(-1)  # [b,16 revisits,m]
        ok = vok[a : a + chunk][:, :, None]
        S = (S * ok).sum(1) / np.maximum(ok.sum(1), 1)  # [b,m]
        top = np.argsort(-S, 1)[:, :2]
        r = np.arange(len(Vb))
        idx[a : a + len(Vb)], score[a : a + len(Vb)] = top[:, 0], S[r, top[:, 0]]
        gap[a : a + len(Vb)] = S[r, top[:, 0]] - S[r, top[:, 1]]
    idx[(score < min_score) | (gap < min_gap)] = -1
    return idx, score, gap


def acc_ci(c):
    c = np.asarray(c, float)
    return {"n": int(len(c)), "acc": float(c.mean() * 100) if len(c) else None,
            "ci95": float(1.96 * np.sqrt(c.mean() * (1 - c.mean()) / max(len(c), 1)) * 100) if len(c) else None}


def paired(pa, pb, y, n_boot=10000, seed=0):
    a, b = pa.argmax(1) == y, pb.argmax(1) == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))
    diff = (a[idx].mean(1) - b[idx].mean(1)) * 100
    return {"diff": float((a.mean() - b.mean()) * 100), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "mcnemar_p": float(p), "a_only": n10, "b_only": n01}


def f1_per_class(y, pred, C):
    out = []
    for c in range(C):
        tp, fp, fn = ((pred == c) & (y == c)).sum(), ((pred == c) & (y != c)).sum(), ((pred != c) & (y == c)).sum()
        out.append(float(2 * tp / max(2 * tp + fp + fn, 1) * 100))
    return out


def main():
    res = {}
    vfp, wfp = load("vb_fingerprints.npz"), load("ws_fingerprints.npz")
    aoi_path = os.path.join(R, "ws_aoi.csv")
    sib_te = None
    if vfp is not None and wfp is not None and os.path.exists(aoi_path):
        aoi = pd.read_csv(aoi_path)["AOI"].values
        site, multi = sites(aoi)
        W = wfp["fp"]
        ntr = len(vfp["train"])
        ia, da, ma = match(np.concatenate([vfp["train"], vfp["test"]]), W, wfp.get("seen"))
        # the VertiBench label is the provider: a match whose provider disagrees with the majority provider of its
        # label is rejected as well
        prov = np.array([a.split("-")[0].split(" ")[0] for a in aoi])
        ya = np.concatenate([vfp["y_train"], vfp["y_test"]])
        ok = ia >= 0
        maj = {c: pd.Series(prov[ia[ok & (ya == c)]]).mode().iloc[0] for c in np.unique(ya[ok])}
        bad = ok & np.array([maj.get(c) != (prov[j] if j >= 0 else None) for c, j in zip(ya, ia)])
        ia[bad] = -1
        itr, ite, mtr, mte = ia[:ntr], ia[ntr:], ma[:ntr], ma[ntr:]
        utr, ute = set(itr[itr >= 0]), set(ite[ite >= 0])
        res["mapping"] = {"train_unique": int(len(utr)), "test_unique": int(len(ute)), "n_train": int(ntr),
                          "n_test": int(len(ite)), "unmatched_train": int((itr < 0).sum()), "unmatched_test": int((ite < 0).sum()),
                          "provider_mismatch": int(bad.sum()), "label_to_provider": {int(k): v for k, v in maj.items()},
                          "train_test_same_area": int(len(utr & ute)), "duplicate_matches": int(ok.sum() - len(set(ia[ia >= 0]))),
                          "score_median": float(np.median(da)), "gap_median": float(np.median(ma)),
                          "gap_p5": float(np.percentile(ma[ia >= 0], 5)), "score_p5": float(np.percentile(da[ia >= 0], 5))}
        tr_sites = set(site[list(utr)])
        sib_te = np.array([j >= 0 and multi[j] and site[j] in tr_sites for j in ite])
        res["sibling"] = {"test_multi_tile": int(sum(multi[j] for j in ite if j >= 0)), "test_with_train_sibling": int(sib_te.sum()),
                          "share": float(sib_te.mean() * 100), "train_multi_tile": int(sum(multi[j] for j in itr if j >= 0)),
                          "test_label_counts_sibling": np.bincount(vfp["y_test"][sib_te], minlength=4).tolist()}
        np.savez_compressed(os.path.join(R, "vb_ws_mapping.npz"), train=itr, test=ite, sibling_test=sib_te)
        print("mapping", res["mapping"], "\nsibling", res["sibling"])

    methods = {}
    fr = load("vb_frozen_probs.npz")
    y = None
    if fr is not None:
        y = fr.pop("y")
        for k, v in fr.items():
            methods[f"frozen {k}"] = v.astype(np.float32)  # [seeds,785,4]
    for name, f in (("SplitNN VertiBench recipe", "vb_repro_splitresnet_preds_probs.npz"), ("split CLIP-B", "vb_split_clipb_probs.npz")):
        d = load(f)
        if d is None:
            continue
        y = d.pop("y") if y is None else (d.pop("y"), y)[1]
        groups = {}
        for k, v in d.items():
            groups.setdefault(k.split("__seed")[0] if "__seed" in k else "", []).append(v.astype(np.float32))
        for g, vs in groups.items():
            methods[f"{name} {g}".strip()] = np.stack(vs)
    if methods:
        res["subsets"] = {}
        for m, P in methods.items():
            C_ = P.argmax(2) == y[None]  # [seeds, n]: accuracies are means over seeds, as in the tables
            def sub(mask):
                r = acc_ci(C_[:, mask].mean(0))
                r["acc"] = float(C_[:, mask].mean() * 100)
                r["acc_std"] = float(C_[:, mask].mean(1).std(ddof=1) * 100) if len(C_) > 1 else 0.0
                r["acc_ensemble"] = float((P[:, mask].mean(0).argmax(1) == y[mask]).mean() * 100)
                return r
            e = {"seeds": len(P), "all": sub(np.ones(len(y), bool))}
            if sib_te is not None:
                e["sibling"], e["no_sibling"] = sub(sib_te), sub(~sib_te)
            res["subsets"][m] = e
            print(f"{m:45s}", {k: (v["acc"] if isinstance(v, dict) else v) for k, v in e.items()})
        ref = methods.get("SplitNN VertiBench recipe")
        if ref is not None:
            res["paired"] = {}
            for m, P in methods.items():
                if m.startswith("SplitNN"):
                    continue
                n = min(len(P), len(ref))  # same number of seeds on both sides of the seed-averaged comparison
                a, b = P[:n].mean(0), ref[:n].mean(0)
                res["paired"][m] = {"seeds": n, "all": paired(a, b, y)}
                if sib_te is not None:
                    res["paired"][m]["no_sibling"] = paired(a[~sib_te], b[~sib_te], y[~sib_te])
                print("paired vs SplitNN", m, res["paired"][m]["all"])

    nn = load("vb_nnpublic.npz")
    if nn is not None and vfp is not None:
        loc, ytr, yte = nn["loc"].astype(int), vfp["y_train"], vfp["y_test"]
        lm = ytr[loc] == yte[:, None]  # [785,16]
        e = {"label_match_all": float(lm.mean() * 100)}
        if sib_te is not None:
            nn_sib = np.array([[ite[i] >= 0 and itr[loc[i, p]] >= 0 and multi[ite[i]] and site[itr[loc[i, p]]] == site[ite[i]]
                                for p in range(16)] for i in range(len(ite))])
            e.update({"nn_is_sibling_given_sibling": float(nn_sib[sib_te].mean() * 100),
                      "label_match_sibling": float(lm[sib_te].mean() * 100), "label_match_no_sibling": float(lm[~sib_te].mean() * 100)})
        res["nnpublic"] = e
        print("nnpublic", e)

    oof = load("ws_sensors_oof_grouped.npz")
    if oof is not None:
        meta = json.load(open(os.path.join(R, "ws_sensors_grouped.json"))) if os.path.exists(os.path.join(R, "ws_sensors_grouped.json")) else {}
        res["ws_f1"] = {}
        for task in ("ipcc", "lccs", "smod"):
            if f"{task}__y" not in oof:
                continue
            yy = oof[f"{task}__y"]
            C = int(yy.max()) + 1
            out = {"classes": meta.get(task, {}).get("classes"), "counts": np.bincount(yy, minlength=C).tolist(), "configs": {}}
            for k, v in oof.items():
                if k.startswith(task + "__") and k.split("__", 1)[1] not in ("y", "sel", "fold"):
                    out["configs"][k.split("__", 1)[1]] = f1_per_class(yy, v.astype(np.float32).argmax(1), C)
            res["ws_f1"][task] = out
    json.dump(res, open(os.path.join(R, "vb_audit_analysis.json"), "w"), indent=1)
    print("wrote vb_audit_analysis.json")


if __name__ == "__main__":
    main()
