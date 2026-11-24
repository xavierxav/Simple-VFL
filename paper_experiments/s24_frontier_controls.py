"""Stage 24 - controls for the communication/accuracy frontier (fourth review).

  python s24_frontier_controls.py vb   [--encoders resnet18_in1k_224,resnet50_in1k_224]
      VertiBench at 224 px, non-foundation control: ImageNet-supervised ResNet-18 / ResNet-50 (torchvision default
      weights, global-average-pooled penultimate features, 512-d / 2048-d), extracted for the 16 parties exactly like
      the foundation encoders (s12_vb224.extract: same per-band standardisation and resize, s7_vb_encoders.prep);
      one-shot heads on the official split (s12_vb224.heads: 5 seeds, MLP and linear) and the per-party PCA-k x
      quantisation-bits grid with validation accuracies (s9_controls E7_pareto_val: s9.run + s2.make_compressor).
      -> $PE_OUT/vb224_controls_resnet.json (or $PE_CONTROLS_OUT), keyed by encoder:
         {enc: {"cost": {...}, "heads": {"dim", "mlp2", "linear"}, "E7_pareto_val": [...], "E7_selected": {...}}}
  python s24_frontier_controls.py milk [--encoders dinov2_b,clip_b16,clip_l14,dinov2_g]
      MILK10k (2 parties: clinical, dermoscopic), frozen features of s11 (f_{enc}_{clin,derm}.npy), same 5 folds,
      head of the main MILK10k result ('clin+derm|mlp2|bal': MLP, class-weighted loss, stopping on the val. loss):
      (i) per-party PCA-k (fitted on the party's training fold only, as s19 'concat + PCA-256 per party') x scalar
      quantisation (s2 party_quantizer, 1/99 % training quantiles; 16 bits = half-precision cast), with the validation
      macro F1 / accuracy of every point and the points selected on validation per byte budget;
      (ii) the few-round logit head of the VertiBench four-logit design (pe_common.PartyLogits, 'logits2'): one MLP
      per party -> C=7 logits, summed by the label owner, same early stopping, epochs_run kept per fold.
      -> $PE_OUT/milk_frontier.json:
         {"n_lesions", "classes", "K", "C", enc: {"dim", "heads": {"clin+derm|mlp2|bal", "clin+derm|logits2|bal"},
                                                  "E7_pareto_val": [...], "E7_selected": {...}}}
Every grid point: summarize() of the runs (acc, auc, f1, bacc in %, mean/std/list) + val_acc[s] (+ val_f1[s] on MILK)
+ d, bits, bytes_per_sample (one-time upload per sample, all parties: K * d * bits / 8).  JSON saved after every point;
a re-run resumes (points already in the file are skipped).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from pe_common import DEV, OUT, fmt, metrics_from_proba, predict_proba, save_json, seed_all, summarize, train_head

BITS = (16, 8, 4, 2, 1)
DIMS = (512, 256, 128, 64, 32, 16)  # E7 grid below the full dimension (s9_controls.e7)


def load_json(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def grid_dims(D):
    return [D] + [d for d in DIMS if d < D]  # s9.e7 uses (D, 512, ...): no duplicate point when D <= 512


def select(grid, key, K, D):
    """Validation-selected points as in make_assets2 (fig_pareto): best `key` among the points within each budget.
    Budgets: powers of two from the smallest point up to, and including, the full fp16 upload K*D*2 bytes."""
    full = K * D * 2
    lo = min(p["bytes_per_sample"] for p in grid)
    budgets = sorted({2**j for j in range(40) if lo <= 2**j <= full} | {full})
    out = {}
    for B in budgets:
        cand = [p for p in grid if p["bytes_per_sample"] <= B]
        if cand:
            best = max(cand, key=lambda p: p[key])
            out[str(int(B))] = {k: best[k] for k in ("d", "bits", "bytes_per_sample", key, "acc", "acc_std", "auc", "auc_std", "f1", "f1_std", "bacc", "bacc_std") if k in best}
    return {"by": key, "budgets": out}


# ================================================================== A: VertiBench, ImageNet ResNets
def _with_resnet18(orig):
    """s12_vb224.build plus 'resnet18_in1k_224' (same image pipeline as 'resnet50_in1k_224'); other names unchanged."""
    if getattr(orig, "_s24", False):
        return orig

    def build(name):
        if name != "resnet18_in1k_224":
            return orig(name)
        import torchvision

        from s7_vb_encoders import prep

        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")  # torchvision default weights for ResNet-18
        m.fc = nn.Identity()  # global-average-pooled 512-d penultimate features
        m.to(DEV).eval()
        return m, (lambda x: m(prep(x, 224)))

    build._s24 = True
    return build


def vb(encoders, out_name):
    import s12_vb224 as s12

    s12.build = _with_resnet18(s12.build)  # s12.extract looks build up at call time
    import s2_vertibench_heads as s2
    import s9_controls as s9  # loads the VertiBench labels (stage-3 cache)

    res = load_json(out_name)
    for name in encoders:
        r = res.setdefault(name, {})
        cost = s12.extract([name])  # skipped when f7_{name}_{train,test}.npy are cached
        if name in cost:
            r["cost"] = cost[name]
            save_json(out_name, res)
        if not os.path.exists(f"{s9.CACHE}/f7_{name}_test.npy"):
            print("!! no features for", name, flush=True)
            continue
        if "heads" not in r:
            h = s12.heads([name])  # 5 seeds, mlp2 + linear, as the encoder table (vb224_encoders.json)
            r["heads"] = h.get(name)
            save_json(out_name, res)
        G = s9.feats(name)
        D = G[0].shape[-1]
        grid = r.setdefault("E7_pareto_val", [])
        done = {(p["d"], p["bits"]) for p in grid}
        for d in grid_dims(D):
            for bits in BITS:
                if (d, bits) in done:
                    continue
                p = s9.run(G[0], G[1], prep=s2.make_compressor(d, bits))  # bits=16: half-precision cast
                p.update({"d": d, "bits": bits, "bytes_per_sample": 16 * d * bits / 8})
                grid.append(p)
                print(f"{name} E7 d={d:5d} bits={bits:2d} val {p['val_acc']:.2f} test {fmt(p)}", flush=True)
                save_json(out_name, res)
        r["E7_selected"] = select(grid, "val_acc", 16, D)
        save_json(out_name, res)
        del G
        torch.cuda.empty_cache()
    print("DONE", flush=True)


# ================================================================== B: MILK10k
def party_quantizer(Ztr, bits):
    """Identical to s2_vertibench_heads.party_quantizer (s2 loads the VertiBench labels at import)."""
    lo = torch.quantile(Ztr, 0.01, dim=0)
    hi = torch.quantile(Ztr, 0.99, dim=0)
    L = 2**bits
    step = (hi - lo).clamp_min(1e-6) / L

    def q(Z):
        i = ((Z - lo) / step).floor().clamp(0, L - 1)
        return lo + (i + 0.5) * step

    return q


class PartyPCA:
    """s19_milk_reg.pca (centre, SVD of the training rows, project, divide by the training std of each component) with
    the SVD computed once per party and fold and reused for every k; k = D: s11_milk10k.zscore (the main head)."""

    def __init__(self, X, tr):
        self.X, self.tr, self.Vt = X, tr, None

    def __call__(self, d):
        from s11_milk10k import zscore

        X, tr = self.X, self.tr
        if d >= X.shape[1]:
            return zscore(X, tr)
        if self.Vt is None:
            self.m = X[tr].mean(0)
            _, _, self.Vt = np.linalg.svd(X[tr] - self.m, full_matrices=False)
        Z = (X - self.m) @ self.Vt[:d].T
        return Z / (Z[tr].std(0) + 1e-6)


def quantise(Z, tr, bits):
    """One party's [N, d] numpy upload -> float32 tensor on DEV after scalar quantisation fitted on its training rows."""
    Z = torch.as_tensor(np.ascontiguousarray(Z, dtype=np.float32), device=DEV)
    if bits == 16:
        return Z.half().float()
    if bits < 16:
        return party_quantizer(Z[torch.as_tensor(tr, device=DEV)], bits)(Z)
    return Z


def milk_runs(y, Fo, make_X, arch="mlp2"):
    """5 folds of the main MILK10k head on make_X(f, tr) -> [N, D] tensor; test and validation metrics, epochs."""
    from s11_milk10k import class_weights

    C = int(y.max()) + 1
    yt = torch.as_tensor(y, device=DEV)
    runs, vals, ep = [], [], []
    for f, (tr, va, te) in enumerate(Fo):
        X = make_X(f, tr)
        m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch=arch, seed=f, class_weight=class_weights(y[tr], C))
        runs.append(metrics_from_proba(y[te], predict_proba(m, X[te]).cpu().numpy()))
        vals.append(metrics_from_proba(y[va], predict_proba(m, X[va]).cpu().numpy()))
        ep.append(int(getattr(m, "epochs_run", 0)))
    r = summarize(runs)
    for k in ("acc", "f1", "bacc"):
        r[f"val_{k}s"] = [v[k] * 100 for v in vals]
        r[f"val_{k}"] = float(np.mean(r[f"val_{k}s"]))
    r["epochs_run"] = ep
    r["params"] = int(sum(p.numel() for p in m.parameters()))
    return r


def milk(encoders, out_name):
    from s11_milk10k import CLASSES7, ROOT, folds, lesions, zscore

    les = lesions()
    y = les["dx7"].to_numpy().copy()
    Fo = folds(y)
    K, C = 2, int(y.max()) + 1
    res = load_json(out_name)
    res.update({"n_lesions": int(len(y)), "classes": CLASSES7, "K": K, "C": C, "parties": ["clin", "derm"],
                "head": "clin+derm|mlp2|bal (s11): MLP, class-weighted cross-entropy, stopping on the unweighted val. loss"})
    for enc in encoders:
        if not all(os.path.exists(f"{ROOT}/f_{enc}_{k}.npy") for k in ("clin", "derm")):
            print("!! no features for", enc, flush=True)
            continue
        P = [np.load(f"{ROOT}/f_{enc}_{k}.npy").astype(np.float32) for k in ("clin", "derm")]
        D = P[0].shape[1]
        r = res.setdefault(enc, {})
        r["dim"] = D
        H = r.setdefault("heads", {})
        cat = lambda f, tr: torch.as_tensor(np.concatenate([zscore(Xp, tr) for Xp in P], 1), device=DEV)  # noqa: E731  (s11 heads)
        if "clin+derm|mlp2|bal" not in H:  # reference: the main MILK10k head re-trained on this VM's features
            H["clin+derm|mlp2|bal"] = milk_runs(y, Fo, cat)
            print(f"{enc:9s} reference mlp2|bal {fmt(H['clin+derm|mlp2|bal'])}", flush=True)
            save_json(out_name, res)
        key = f"clin+derm|logits{K}|bal"
        if key not in H:  # few rounds: per training pass each party sends C logits and receives their C gradients
            h = milk_runs(y, Fo, cat, arch=f"logits{K}")
            n_tr, n_va = np.mean([len(tr) for tr, _, _ in Fo]), np.mean([len(va) for _, va, _ in Fo])
            h["K"], h["C"] = K, C
            h["bytes_per_train_sample"] = float(np.mean(h["epochs_run"]) * 2 * K * C * 2)  # make_assets2: training passes, fp16
            h["bytes_per_train_sample_incl_val"] = float(np.mean([e * (2 * n_tr + n_va) * K * C * 2 / n_tr for e in h["epochs_run"]]))  # s22
            H[key] = h
            print(f"{enc:9s} {key} {fmt(h)} epochs {h['epochs_run']} -> {h['bytes_per_train_sample']:.0f} B/sample", flush=True)
            save_json(out_name, res)
        grid = r.setdefault("E7_pareto_val", [])
        done = {(p["d"], p["bits"]) for p in grid}
        pcas = {}  # (fold, party) -> PartyPCA (one SVD per party and fold)

        def make_X(f, tr, d, bits):
            parts = []
            for i, Xp in enumerate(P):
                pc = pcas.setdefault((f, i), PartyPCA(Xp, tr))
                parts.append(quantise(pc(d), tr, bits))
            return torch.cat(parts, 1)

        for d in grid_dims(D):
            for bits in BITS:
                if (d, bits) in done:
                    continue
                p = milk_runs(y, Fo, lambda f, tr, d=d, bits=bits: make_X(f, tr, d, bits))
                p.update({"d": d, "bits": bits, "bytes_per_sample": K * d * bits / 8})
                grid.append(p)
                print(f"{enc:9s} d={d:5d} bits={bits:2d} val F1 {p['val_f1']:.2f} test {fmt(p)}", flush=True)
                save_json(out_name, res)
        r["E7_selected"] = select(grid, "val_f1", K, D)
        r["E7_selected_val_acc"] = select(grid, "val_acc", K, D)
        save_json(out_name, res)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vb", "milk"])
    ap.add_argument("--encoders", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seed_all(0)
    if a.stage == "vb":
        vb((a.encoders or "resnet18_in1k_224,resnet50_in1k_224").split(","), a.out or os.environ.get("PE_CONTROLS_OUT", "vb224_controls_resnet.json"))
    else:
        milk((a.encoders or "dinov2_b,clip_b16,clip_l14,dinov2_g").split(","), a.out or "milk_frontier.json")


if __name__ == "__main__":
    main()
