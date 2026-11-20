"""Stage 16 - WorldStrat SPOT + Sentinel-2: can the label owner make the second party help?

Runs after s15 (reuses its arrays and per-revisit features in $PE_WS2).  Everything stays one-shot: each party
uploads once; what changes is (i) the local summary the Sentinel-2 party uploads and (ii) the head.

  python s16_ws_fusion.py extra     GPU: party-local cloud score per revisit; centre crops (~340 m, the footprint of
                                    the land-cover label) of SPOT (1.5 m) and of every S2 revisit (RGB, NIR, SWIR);
                                    SPOT 4x4 tiles (263 px each) encoded with the shared frozen encoder
  python s16_ws_fusion.py heads     -> ws_fusion.json

Head / summary variants (5-fold CV, class-balanced loss, macro F1 primary, as s15):
  S2 summaries   mean (s15) | median | clear-mean (8 least cloudy revisits) | mean+std (x2 upload) |
                 set of 16 vectors (x16 upload) with a weight-shared per-revisit MLP + attention pooling,
                 random revisit subsets at training time
  fusion heads   concat MLP (s15) | per-party PCA-256 + concat | late fusion (mean log-probs, weight on val) |
                 branch head: per-party projection + auxiliary per-party losses + modality dropout
  diagnostics    agreement of the single-party heads on the test folds: both wrong, oracle (either right)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from pe_common import DEV, evaluate_all, fmt, metrics_from_proba, predict_proba, save_json, seed_all, summarize, train_head

ROOT = os.environ.get("PE_WS2", "/content/ws2")
OUT = os.environ.get("PE_OUT", "/content/results")
NIR, SWIR, RGB = [7, 3, 2], [12, 8, 3], [3, 2, 1]
B2, B10 = 1, 10  # blue, cirrus (L1C order B1..B8, B8A, B9, B10, B11, B12)


# ------------------------------------------------------------------ GPU stage
def extra():
    from transformers import AutoModel

    SP = np.load(f"{ROOT}/spot_rgb.npy", mmap_mode="r")
    S2 = np.load(f"{ROOT}/s2_l1c.npy", mmap_mode="r")
    N = len(SP)
    st = np.load(f"{ROOT}/band_stats.npz")
    spm, sps = torch.tensor(st["sp_m"], device=DEV)[None, :, None, None], torch.tensor(st["sp_s"], device=DEV)[None, :, None, None]
    s2m, s2s = torch.tensor(st["s2_m"], device=DEV)[None, :, None, None], torch.tensor(st["s2_s"], device=DEV)[None, :, None, None]
    cost = {}
    if not os.path.exists(f"{ROOT}/s2_cloud.npy"):  # party-local cloud score: mean blue + cirrus reflectance, and share of bright pixels
        t = time.time()
        sc = np.zeros((N, 16, 3), np.float32)
        for i in range(0, N, 64):
            blk = np.asarray(S2[i : i + 64, :, [B2, B10]], dtype=np.float32) / 255  # [b,16,2,158,158]
            sc[i : i + 64, :, 0] = blk[:, :, 0].mean((2, 3))
            sc[i : i + 64, :, 1] = blk[:, :, 1].mean((2, 3))
            sc[i : i + 64, :, 2] = (blk[:, :, 0] > 0.25).mean((2, 3))
        np.save(f"{ROOT}/s2_cloud.npy", sc)
        cost["cloud_score"] = time.time() - t
        print("cloud score", f"{time.time() - t:.0f}s", flush=True)
    g = AutoModel.from_pretrained("facebook/dinov2-giant").to(DEV).eval()

    def run(path, n, bs, load):
        if os.path.exists(path):
            return
        out = np.zeros((n, 1536), np.float16)
        t = time.time()
        for i in range(0, n, bs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out[i : i + bs] = g(pixel_values=load(i, min(n, i + bs))).pooler_output.float().cpu().numpy()
        torch.cuda.synchronize()
        cost[os.path.basename(path)] = time.time() - t
        np.save(path, out)
        print(path, out.shape, f"{time.time() - t:.0f}s", flush=True)

    c0 = (1050 - 224) // 2  # 224 px x 1.5 m = 336 m around the centre

    def spot_centre(i, j):
        x = torch.from_numpy(np.ascontiguousarray(SP[i:j, :, c0 : c0 + 224, c0 : c0 + 224])).to(DEV).float()
        return (x - spm) / sps

    run(f"{ROOT}/f_spot_dinov2_g_centre.npy", N, 64, spot_centre)
    T = 4
    ts = 1050 // T  # 262 px tiles (1.5 m), 16 per area

    def spot_tiles(i, j):  # i, j index the flattened [N*16] tile list
        a, b = i // 16, (j - 1) // 16 + 1
        x = torch.from_numpy(np.ascontiguousarray(SP[a:b, :, : ts * T, : ts * T])).to(DEV).float()
        x = (x - spm) / sps
        x = x.reshape(b - a, 3, T, ts, T, ts).permute(0, 2, 4, 1, 3, 5).reshape((b - a) * 16, 3, ts, ts)
        x = x[i - a * 16 : j - a * 16]
        return F.interpolate(x, size=(266, 266), mode="bilinear", align_corners=False)  # multiple of 14

    run(f"{ROOT}/f_spot_dinov2_g_tiles.npy", N * 16, 64, spot_tiles)
    s2flat = S2.reshape(N * 16, 13, 158, 158)
    k0 = (158 - 34) // 2  # 34 px x 10 m = 340 m around the centre

    def s2_centre(i, j, bands):
        x = torch.from_numpy(np.ascontiguousarray(s2flat[i:j, :, k0 : k0 + 34, k0 : k0 + 34])).to(DEV).float()
        x = (x - s2m) / s2s
        return F.interpolate(x[:, bands], size=(224, 224), mode="bilinear", align_corners=False)

    for name, bands in (("rgb", RGB), ("nir", NIR), ("swir", SWIR)):
        run(f"{ROOT}/f_s2_dinov2_g_{name}_centre.npy", N * 16, 256, lambda i, j, b=bands: s2_centre(i, j, b))
    p = f"{ROOT}/feature_secs_s16.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(cost)
    json.dump(old, open(p, "w"))


# ------------------------------------------------------------------ heads
def labels(df, task):
    col = {"ipcc": df["IPCC Class"].fillna("Other"), "smod": df["SMOD Class"], "lccs": df["LCCS class"]}[task].astype(str)
    counts = col.value_counts()
    keep = counts[counts >= 40].index if task == "lccs" else counts.index
    classes = sorted(keep)
    return np.array([classes.index(c) if c in classes else -1 for c in col]), classes


def folds(y, k=5):
    from sklearn.model_selection import StratifiedKFold

    f = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        f[te] = i
    return f


def std_fit(A, tr):
    """Standardise with training statistics; works for [N, d] and [N, R, d] (statistics over N and R)."""
    ax = (0,) if A.ndim == 2 else (0, 1)
    m, s = A[tr].mean(ax, keepdims=True)[0], A[tr].std(ax, keepdims=True)[0] + 1e-6
    return (A - m) / s


def pca_fit(A, tr, d):
    m = A[tr].mean(0)
    _, _, Vt = np.linalg.svd(A[tr] - m, full_matrices=False)
    Z = (A - m) @ Vt[:d].T
    return Z / (Z[tr].std(0) + 1e-6)


class SetHead(nn.Module):
    """Optional plain vectors (other parties) + a set of R per-revisit vectors.

    Per-revisit vectors go through one weight-shared MLP (phi), are pooled by attention (masked revisits ignored),
    concatenated with the projected plain vectors and classified."""

    def __init__(self, d_set, d_plain, n_cls, h=512):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_set, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h))
        self.att = nn.Linear(h, 1)
        self.plain = nn.Sequential(nn.Linear(d_plain, h), nn.LayerNorm(h), nn.GELU()) if d_plain else None
        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(h * (2 if d_plain else 1), n_cls))
        self.aux_set = nn.Linear(h, n_cls)
        self.aux_plain = nn.Linear(h, n_cls) if d_plain else None

    def forward(self, s, p=None, mask=None, drop=None, aux=False):
        z = self.phi(s)  # [B, R, h]
        a = self.att(z).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e4)
        zs = (z * a.softmax(1).unsqueeze(-1)).sum(1)
        parts = [zs]
        if self.plain is not None:
            zp = self.plain(p)
            parts = [zp, zs]
        if drop is not None:  # modality dropout: zero one branch for some training samples
            parts = [q * keep.unsqueeze(1) for q, keep in zip(parts, drop)]
        o = self.out(torch.cat(parts, 1))
        if not aux:
            return o
        return o, self.aux_set(zs), (self.aux_plain(zp) if self.plain is not None else None)


class BranchHead(nn.Module):
    """Per-party projection + joint classifier + auxiliary per-party classifiers (modality dropout at training)."""

    def __init__(self, dims, n_cls, h=512):
        super().__init__()
        self.dims = dims
        self.proj = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()) for d in dims])
        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(h * len(dims), n_cls))
        self.aux = nn.ModuleList([nn.Linear(h, n_cls) for _ in dims])

    def forward(self, x, drop=None, aux=False):
        xs = torch.split(x, self.dims, 1)
        zs = [p(v) for p, v in zip(self.proj, xs)]
        zz = zs if drop is None else [z * k.unsqueeze(1) for z, k in zip(zs, drop)]
        o = self.out(torch.cat(zz, 1))
        return (o, [a(z) for a, z in zip(self.aux, zs)]) if aux else o


def fit_custom(model, batch_loss, predict_val, yva, seed, epochs=60, lr=1e-3, wd=1e-4, n=None, bs=256, patience=12):
    """Generic loop: Adam, ReduceLROnPlateau, best-val-loss checkpoint (same selection rule as train_head)."""
    import copy

    seed_all(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            loss = batch_loss(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(predict_val(), yva).item()
        sched.step(vl)
        if vl < best:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def heads():
    df = pd.read_csv(f"{ROOT}/aoi.csv")
    N = len(df)
    ok = np.load(f"{ROOT}/spot_seen.npy") & np.load(f"{ROOT}/s2_seen.npy").all(1)

    def per_revisit(name):
        return np.load(f"{ROOT}/f_s2_{name}.npy").astype(np.float32).reshape(N, 16, -1)

    S2R = np.concatenate([per_revisit("dinov2_g_rgb"), per_revisit("dinov2_g_nir"), per_revisit("dinov2_g_swir")], 2)  # [N,16,4608]
    have_c = os.path.exists(f"{ROOT}/f_s2_dinov2_g_swir_centre.npy")
    S2C = np.concatenate([per_revisit(f"dinov2_g_{b}_centre") for b in ("rgb", "nir", "swir")], 2) if have_c else None
    cloud = np.load(f"{ROOT}/s2_cloud.npy") if os.path.exists(f"{ROOT}/s2_cloud.npy") else None
    P = {"SPOT": np.load(f"{ROOT}/f_spot_dinov2_g_1.5m.npy").astype(np.float32)}
    if os.path.exists(f"{ROOT}/f_spot_dinov2_g_centre.npy"):
        P["SPOT centre"] = np.load(f"{ROOT}/f_spot_dinov2_g_centre.npy").astype(np.float32)
    if os.path.exists(f"{ROOT}/f_spot_dinov2_g_tiles.npy"):
        P["SPOT tiles mean"] = np.load(f"{ROOT}/f_spot_dinov2_g_tiles.npy").astype(np.float32).reshape(N, 16, -1).mean(1)
    P["S2 mean"] = S2R.mean(1)
    P["S2 median"] = np.median(S2R, 1)
    P["S2 mean+std"] = np.concatenate([S2R.mean(1), S2R.std(1)], 1)
    if cloud is not None:
        score = cloud[:, :, 0] + cloud[:, :, 1] + cloud[:, :, 2]
        clear = np.argsort(score, 1)[:, :8]
        P["S2 clear-mean"] = np.take_along_axis(S2R, clear[:, :, None], 1).mean(1)
    if S2C is not None:
        P["S2 centre mean"] = S2C.mean(1)
        if cloud is not None:
            P["S2 centre clear-mean"] = np.take_along_axis(S2C, clear[:, :, None], 1).mean(1)
    if "SPOT centre" in P:
        P["SPOT full+centre"] = np.concatenate([P["SPOT"], P["SPOT centre"]], 1)
    path = os.path.join(OUT, "ws_fusion.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    single = [k for k in P if not k.startswith("SPOT full+")] + (["SPOT full+centre"] if "SPOT full+centre" in P else [])
    concat = [["SPOT", "S2 mean"], ["SPOT", "S2 median"], ["SPOT", "S2 mean+std"], ["SPOT", "S2 clear-mean"], ["SPOT centre", "S2 centre mean"],
              ["SPOT full+centre", "S2 clear-mean"], ["SPOT full+centre", "S2 centre clear-mean"], ["SPOT tiles mean", "S2 clear-mean"]]
    concat = [c for c in concat if all(p in P for p in c)]
    for task in ("ipcc", "lccs", "smod"):
        y_all, classes = labels(df, task)
        sel = np.where(ok & (y_all >= 0))[0]
        y = y_all[sel]
        C = len(classes)
        f = folds(y)
        out = res.setdefault(task, {})
        out["classes"], out["counts"] = classes, np.bincount(y, minlength=C).tolist()
        yt = torch.as_tensor(y, device=DEV)
        split = [(np.where((f != k) & (f != (k + 1) % 5))[0], np.where(f == (k + 1) % 5)[0], np.where(f == k)[0]) for k in range(5)]

        def cw(tr):
            cnt = np.bincount(y[tr], minlength=C).astype(float)
            return torch.tensor(cnt.sum() / (C * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)

        def record(key, runs):
            out[key] = summarize(runs)
            print(f"{task:5s} {key:46s} {fmt(out[key])}", flush=True)
            save_json("ws_fusion.json", res)

        probas = {}  # single-party test probabilities per fold, for late fusion and diagnostics

        def plain(key, parts, pca=None):
            runs, pv, pt = [], [], []
            for k, (tr, va, te) in enumerate(split):
                Z = [(pca_fit(P[p][sel], tr, pca) if pca else std_fit(P[p][sel], tr)) for p in parts]
                X = torch.as_tensor(np.concatenate(Z, 1), device=DEV)
                m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch="mlp2", seed=k, class_weight=cw(tr).tolist())
                runs.append(evaluate_all(m, X[te], yt[te]))
                pv.append(predict_proba(m, X[va]).cpu().numpy())
                pt.append(predict_proba(m, X[te]).cpu().numpy())
            if key not in out:
                record(key, runs)
            return pv, pt

        need = {"SPOT", "S2 mean", "S2 clear-mean", "SPOT full+centre", "S2 centre clear-mean"}
        for p in single:
            if p in need or f"{p}|mlp2" not in out:
                probas[p] = plain(f"{p}|mlp2", [p])
        for parts in concat:
            key = " + ".join(parts) + "|concat"
            if key not in out:
                runs = []
                for k, (tr, va, te) in enumerate(split):
                    X = torch.as_tensor(np.concatenate([std_fit(P[p][sel], tr) for p in parts], 1), device=DEV)
                    m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch="mlp2", seed=k, class_weight=cw(tr).tolist())
                    runs.append(evaluate_all(m, X[te], yt[te]))
                record(key, runs)
        key = "SPOT + S2 clear-mean|pca256 concat"
        if key not in out and "S2 clear-mean" in P:
            plain(key, ["SPOT", "S2 clear-mean"], pca=256)
        # late fusion: weighted mean of log-probabilities, weight chosen on the validation fold
        for a, b in (("SPOT", "S2 mean"), ("SPOT", "S2 clear-mean"), ("SPOT full+centre", "S2 centre clear-mean")):
            key = f"{a} + {b}|late fusion"
            if key in out or not probas.get(a) or not probas.get(b):
                continue
            runs, ws = [], []
            for k, (tr, va, te) in enumerate(split):
                la, lb = np.log(probas[a][0][k] + 1e-9), np.log(probas[b][0][k] + 1e-9)
                grid = np.linspace(0, 1, 21)
                vals = [f1_score(y[va], (w * la + (1 - w) * lb).argmax(1), average="macro", labels=np.arange(C), zero_division=0) for w in grid]
                w = grid[int(np.argmax(vals))]
                ws.append(float(w))
                ta, tb = np.log(probas[a][1][k] + 1e-9), np.log(probas[b][1][k] + 1e-9)
                pr = np.exp(w * ta + (1 - w) * tb)
                runs.append(metrics_from_proba(y[te], pr / pr.sum(1, keepdims=True)))
            record(key, runs)
            out[key]["spot_weight"] = ws
        # diagnostics: do the two parties fail on the same areas?
        for a, b in (("SPOT", "S2 mean"), ("SPOT", "S2 clear-mean")):
            if not probas.get(a) or not probas.get(b):
                continue
            both, either, ea, eb = [], [], [], []
            for k, (tr, va, te) in enumerate(split):
                ra, rb = probas[a][1][k].argmax(1) == y[te], probas[b][1][k].argmax(1) == y[te]
                both.append(100 * np.mean(~ra & ~rb))
                either.append(100 * np.mean(ra | rb))
                ea.append(100 * np.mean(~ra))
                eb.append(100 * np.mean(~rb))
            out[f"diag {a} vs {b}"] = {"err_a": float(np.mean(ea)), "err_b": float(np.mean(eb)), "both_wrong": float(np.mean(both)),
                                       "oracle_acc": float(np.mean(either)), "both_wrong_if_independent": float(np.mean(ea) * np.mean(eb) / 100)}
            print(task, "diag", a, b, out[f"diag {a} vs {b}"], flush=True)
        # branch head: per-party projections, auxiliary losses, modality dropout
        for parts, pdrop in ((["SPOT", "S2 clear-mean"], 0.0), (["SPOT", "S2 clear-mean"], 0.3), (["SPOT full+centre", "S2 centre clear-mean"], 0.3)):
            key = " + ".join(parts) + f"|branch aux md{pdrop}"
            if key in out or not all(p in P for p in parts):
                continue
            runs = []
            for k, (tr, va, te) in enumerate(split):
                Z = [std_fit(P[p][sel], tr) for p in parts]
                X = torch.as_tensor(np.concatenate(Z, 1), device=DEV)
                m = BranchHead([z.shape[1] for z in Z], C).to(DEV)
                w = cw(tr)
                trt = torch.as_tensor(tr, device=DEV)

                def loss_fn(idx, m=m, X=X, w=w, trt=trt):
                    j = trt[idx]
                    drop = None
                    if pdrop > 0:
                        r = torch.rand(len(j), device=DEV)
                        drop = [(r >= pdrop / 2).float(), ((r < pdrop / 2) | (r >= pdrop)).float()]  # drop party 0 or party 1, never both
                    o, ax = m(X[j], drop=drop, aux=True)
                    return F.cross_entropy(o, yt[j], weight=w) + 0.5 * sum(F.cross_entropy(a, yt[j], weight=w) for a in ax)

                m = fit_custom(m, loss_fn, lambda m=m, X=X, va=va: m(X[va]), yt[va], seed=k, n=len(tr))
                with torch.no_grad():
                    runs.append(metrics_from_proba(y[te], m(X[te]).softmax(1).cpu().numpy()))
            record(key, runs)
        # set head on the 16 revisits (weight-shared per-revisit network), alone and with SPOT
        for with_spot, src, keep_clear, tag in ((False, S2R, False, "S2 set16"), (True, S2R, False, "SPOT + S2 set16"),
                                                (True, S2R, True, "SPOT + S2 set8 clear"), (True, S2C, True, "SPOT full+centre + S2 centre set8 clear")):
            key = f"{tag}|set attn"
            if key in out or src is None or (keep_clear and cloud is None) or (tag.startswith("SPOT full") and "SPOT full+centre" not in P):
                continue
            runs = []
            S = src[sel] if not keep_clear else np.take_along_axis(src, clear[:, :, None], 1)[sel]
            spot_key = "SPOT full+centre" if tag.startswith("SPOT full") else "SPOT"
            for k, (tr, va, te) in enumerate(split):
                St = torch.as_tensor(std_fit(S, tr), device=DEV)
                Pt = torch.as_tensor(std_fit(P[spot_key][sel], tr), device=DEV) if with_spot else None
                m = SetHead(St.shape[-1], Pt.shape[1] if with_spot else 0, C).to(DEV)
                w = cw(tr)
                trt = torch.as_tensor(tr, device=DEV)
                R = St.shape[1]

                def loss_fn(idx, m=m, St=St, Pt=Pt, w=w, trt=trt, R=R):
                    j = trt[idx]
                    mask = torch.rand(len(j), R, device=DEV) < 0.7  # random revisit subsets
                    mask[torch.arange(len(j)), torch.randint(0, R, (len(j),), device=DEV)] = True
                    drop = None
                    if Pt is not None:
                        r = torch.rand(len(j), device=DEV)
                        drop = [(r >= 0.15).float(), ((r < 0.15) | (r >= 0.3)).float()]
                    o, a_s, a_p = m(St[j], Pt[j] if Pt is not None else None, mask=mask, drop=drop, aux=True)
                    loss = F.cross_entropy(o, yt[j], weight=w) + 0.5 * F.cross_entropy(a_s, yt[j], weight=w)
                    return loss + (0.5 * F.cross_entropy(a_p, yt[j], weight=w) if a_p is not None else 0)

                m = fit_custom(m, loss_fn, lambda m=m, St=St, Pt=Pt, va=va: m(St[va], Pt[va] if Pt is not None else None), yt[va], seed=k, n=len(tr))
                with torch.no_grad():
                    runs.append(metrics_from_proba(y[te], m(St[te], Pt[te] if Pt is not None else None).softmax(1).cpu().numpy()))
            record(key, runs)
            out[key]["upload_vectors_s2"] = int(S.shape[1])
    save_json("ws_fusion.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extra", "heads", "all"])
    a = ap.parse_args()
    if a.stage in ("extra", "all"):
        extra()
    if a.stage in ("heads", "all"):
        heads()


if __name__ == "__main__":
    main()
