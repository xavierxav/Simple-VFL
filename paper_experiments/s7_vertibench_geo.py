"""Stage 7 - why does CLIP beat DINOv2 on VertiBench Satellite?  Encoder ablations + geography.

(A) re-extract frozen features from the packed raw arrays of stage 3 (full13_{train,test}.npy) for controlled variants:
    input resolution (DINOv2-g at 168/224 px, CLIP-L at 224/168 px), pooling (DINOv2 [CLS; mean patch]),
    other image-text encoders (CLIP-B/16, SigLIP-B/16), small and random references.
(B) heads on the official split (5 seeds, MLP and linear), as in stage 2.
(C) recover the WorldStrat location of every VertiBench sample: VertiBench was built from the WorldStrat AOIs, so each
    sample is matched to a WorldStrat POI by the similarity of the DINOv2-g features of its 16 revisits (one-to-one
    assignment inside each provider class), with diagnostics (mutual best match, margin, revisit alignment).
(D) with these coordinates: location-only random forest on the official split, nearest-neighbour geolocation from the
    features (how much each encoder knows about *where* an image is), and random vs spatial 5-fold cross-validation
    of the frozen-feature heads on the pooled 3.9k samples.
Outputs: vb_encoders.json, vb_geo.json, vb_poi_match.csv (VertiBench index -> WorldStrat POI, lon, lat, confidence).
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from pe_common import DEV, OUT, auc_macro_ovr, evaluate, fmt, save_json, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
MEAN = torch.tensor([81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384])
STD = torch.tensor([37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326])
RGB = [3, 2, 1]
PROVIDERS = ["Amnesty POI", "ASMSpotter", "Landcover", "UNHCR"]  # WorldStrat 'sources' order (s5_worldstrat.SOURCES)


def prep(x, size):  # x: [B,13,H,W] float (0..255) -> normalised RGB [B,3,size,size], same for every encoder
    m, s = MEAN[RGB].to(x.device)[None, :, None, None], STD[RGB].to(x.device)[None, :, None, None]
    return F.interpolate((x[:, RGB] - m) / s, size=(size, size), mode="bilinear", align_corners=False)


# ------------------------------------------------------------------ (A) encoders
def build_encoders():
    from transformers import AutoConfig, AutoModel, CLIPVisionModel, SiglipVisionModel

    enc = {}
    g = AutoModel.from_pretrained("facebook/dinov2-giant").to(DEV).eval()
    enc["dinov2_g_168"] = lambda x: g(prep(x, 168)).pooler_output
    enc["dinov2_g_224"] = lambda x: g(prep(x, 224)).pooler_output

    def cls_mean(x):
        h = g(prep(x, 168)).last_hidden_state  # already layer-normed in HF Dinov2Model
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    enc["dinov2_g_168_clsmean"] = cls_mean
    s = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    enc["dinov2_s_168"] = lambda x: s(prep(x, 168)).pooler_output
    torch.manual_seed(0)
    r = AutoModel.from_config(AutoConfig.from_pretrained("facebook/dinov2-base")).to(DEV).eval()
    enc["dinov2_b_random_168"] = lambda x: r(prep(x, 168)).pooler_output
    cl = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    enc["clip_l14_224"] = lambda x: cl(pixel_values=prep(x, 224)).pooler_output
    enc["clip_l14_168"] = lambda x: cl(pixel_values=prep(x, 168), interpolate_pos_encoding=True).pooler_output
    cb = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval()
    enc["clip_b16_224"] = lambda x: cb(pixel_values=prep(x, 224)).pooler_output
    try:
        sg = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(DEV).eval()
        enc["siglip_b16_224"] = lambda x: sg(pixel_values=prep(x, 224)).pooler_output
    except Exception as e:  # optional
        print("!! siglip unavailable", repr(e), flush=True)
    return enc


def extract_all(chunk=128):
    enc = None
    for split in ("train", "test"):
        todo = [n for n in ("dinov2_g_168", "dinov2_g_224", "dinov2_g_168_clsmean", "dinov2_s_168", "dinov2_b_random_168", "clip_l14_224", "clip_l14_168", "clip_b16_224", "siglip_b16_224")
                if not os.path.exists(f"{CACHE}/f7_{n}_{split}.npy")]
        if not todo:
            continue
        enc = enc or build_encoders()
        todo = [n for n in todo if n in enc]
        A = np.load(f"{CACHE}/full13_{split}.npy", mmap_mode="r")  # [N,16,13,158,158] uint8 (stage 3 'pack')
        N = len(A)
        out = {n: None for n in todo}
        t0 = time.time()
        for i in range(0, N, chunk):
            xb = torch.from_numpy(np.ascontiguousarray(A[i : i + chunk])).to(DEV)
            for p in range(16):
                x = xb[:, p].float()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    for n in list(todo):
                        try:
                            f = enc[n](x).float().cpu().numpy().astype(np.float16)
                        except TypeError as e:  # e.g. no interpolate_pos_encoding in an old transformers
                            print("!! drop", n, repr(e), flush=True)
                            todo.remove(n)
                            out.pop(n)
                            continue
                        if out[n] is None:
                            out[n] = np.zeros((N, 16, f.shape[1]), np.float16)
                        out[n][i : i + len(f), p] = f
            if i % 1024 == 0:
                print(f"extract {split} {i + len(xb)}/{N} ({time.time() - t0:.0f}s)", flush=True)
        for n, a in out.items():
            assert np.isfinite(a.astype(np.float32)).all(), n
            np.save(f"{CACHE}/f7_{n}_{split}.npy", a)
        print(f"extract {split} done in {time.time() - t0:.0f}s: {list(out)}", flush=True)


def load_feats(n):
    return [torch.from_numpy(np.load(f"{CACHE}/f7_{n}_{s}.npy")).to(DEV).float() for s in ("train", "test")]


# ------------------------------------------------------------------ (B) heads on the official split
def official_heads(names, ytr, yte, ncls):
    res = {}
    for n in names:
        Xtr, Xte = load_feats(n)
        r = {"dim": int(Xtr.shape[-1])}
        for arch in ("mlp2", "linear"):
            runs = []
            for s in range(5):
                tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
                tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
                m = train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], ncls, arch=arch, seed=s)
                acc, auc = evaluate(m, Xte, yte)
                runs.append({"acc": acc, "auc": auc})
            r[arch] = summarize(runs)
        res[n] = r
        print(f"official {n:22s} d={r['dim']:5d} mlp {fmt(r['mlp2'])} | linear {fmt(r['linear'])}", flush=True)
    return res


# ------------------------------------------------------------------ (C) VertiBench <-> WorldStrat matching
def match_to_worldstrat(vb_all, y_all):
    """vb_all: [N,16,1536] DINOv2-g features of VertiBench (train then test); y_all: [N] labels, used ONLY afterwards
    as a diagnostic (provider agreement), never to restrict the matching: otherwise the test labels would leak into
    the recovered coordinates.  Returns one row per VertiBench sample: POI, lon, lat and match diagnostics."""
    import s5_worldstrat as s5

    meta, _, ws = s5.load_all()  # ws: [M,16,1536] DINOv2-g features of the WorldStrat Sentinel-2 L2A RGB revisits
    ws = ws.float()
    prov = meta["POI"].str.split("-").str[0].to_numpy()
    # VertiBench label index -> provider, from the class frequencies alone (Amnesty 4.8, ASM 8.9, Landcover 61.3, UNHCR 25.0 %)
    vb_counts = np.bincount(y_all, minlength=4)
    ws_counts = np.array([(prov == p).sum() for p in PROVIDERS])
    vb_rank, ws_rank = np.argsort(vb_counts), np.argsort(ws_counts)
    lab2prov = {int(vb_rank[k]): PROVIDERS[ws_rank[k]] for k in range(4)}
    print("VertiBench label -> provider", lab2prov, "counts", vb_counts.tolist(), ws_counts.tolist(), flush=True)

    def unit(X):  # remove each data set's mean (sensor level L1C vs L2A, preprocessing), then L2-normalise
        X = X - X.mean((0, 1), keepdim=True)
        return F.normalize(X, dim=-1)

    V, W = unit(torch.as_tensor(vb_all, device=DEV).float()), unit(ws.to(DEV))
    Vm, Wm = F.normalize(V.mean(1), dim=-1), F.normalize(W.mean(1), dim=-1)
    S = (Vm @ Wm.T).cpu().numpy()  # [N,M] order-invariant similarity (mean over revisits), all samples x all POIs
    t0 = time.time()
    r, c = linear_sum_assignment(-S)  # global one-to-one assignment, label-free
    print(f"global assignment {S.shape} in {time.time() - t0:.0f}s", flush=True)
    best_w, best_v = S.argmax(1), S.argmax(0)
    srt = np.sort(S, 1)
    rows = []
    for i, j in zip(r, c):
        C = (V[i] @ W[j].T).cpu().numpy()  # revisit alignment: is party p closest to WorldStrat revisit p? (chance 1/16)
        rows.append({
            "vb_index": int(i), "label": int(y_all[i]), "provider": prov[j], "poi": meta["POI"].iloc[j], "lon": float(meta["lon"].iloc[j]), "lat": float(meta["lat"].iloc[j]),
            "sim": float(S[i, j]), "margin": float(S[i, j] - (srt[i, -2] if j == best_w[i] else srt[i, -1])),
            "mutual_best": bool(best_w[i] == j and best_v[j] == i), "revisit_alignment": float((C.argmax(1) == np.arange(16)).mean()),
            "provider_agrees": bool(lab2prov[int(y_all[i])] == prov[j]),
        })
    df = pd.DataFrame(rows)
    df["confident"] = df["mutual_best"]  # label-free criterion
    print("matched", len(df), "of", len(V), "| mutual best", int(df["mutual_best"].sum()), "| provider agreement", df["provider_agrees"].mean(),
          "| median revisit alignment", df["revisit_alignment"].median(), flush=True)
    return df


# ------------------------------------------------------------------ (D) geography
def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def great_circle_km(a, b):
    return 6371.0 * np.arccos(np.clip((a * b).sum(1), -1, 1))


def nn_geolocation(feat_tr, feat_te, xyz_tr, xyz_te):
    """Predict the location of a test sample by the location of its nearest training sample in feature space."""
    A = F.normalize(feat_tr.mean(1), dim=-1)
    B = F.normalize(feat_te.mean(1), dim=-1)
    j = (B @ A.T).argmax(1).cpu().numpy()
    d = great_circle_km(xyz_te, xyz_tr[j])
    return {"median_km": float(np.median(d)), "within_25km": float((d < 25).mean() * 100), "within_100km": float((d < 100).mean() * 100), "within_500km": float((d < 500).mean() * 100)}


def spatial_folds(xyz_all, n_clusters=25, n_folds=5):
    from sklearn.cluster import KMeans

    cl = KMeans(n_clusters, n_init=10, random_state=0).fit_predict(xyz_all)
    sizes = np.bincount(cl, minlength=n_clusters)
    fold_of, load = {}, np.zeros(n_folds)
    for c in np.argsort(-sizes):
        f = int(np.argmin(load))
        fold_of[c] = f
        load[f] += sizes[c]
    return np.array([fold_of[c] for c in cl])


def cv_items(fold):
    out = []
    for f in range(5):
        te = np.where(fold == f)[0]
        va = np.where(fold == (f + 1) % 5)[0]
        tr = np.where((fold != f) & (fold != (f + 1) % 5))[0]
        out.append((tr, va, te))
    return out


def main():
    import sys

    geo_only = "--geo-only" in sys.argv  # features and vb_encoders.json already there: redo (C) and (D) only
    ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    ncls = int(ytr.max()) + 1
    extract_all()
    names = [n for n in ("dinov2_g_168", "dinov2_g_224", "dinov2_g_168_clsmean", "clip_l14_224", "clip_l14_168", "clip_b16_224", "siglip_b16_224", "dinov2_s_168", "dinov2_b_random_168")
             if os.path.exists(f"{CACHE}/f7_{n}_test.npy")]
    if not geo_only:
        enc_res = official_heads(names, ytr, yte, ncls)
        save_json("vb_encoders.json", enc_res)

    # ---- matching
    g_tr, g_te = load_feats("dinov2_g_168")
    y_all = np.concatenate([ytr.cpu().numpy(), yte.cpu().numpy()])
    df = match_to_worldstrat(torch.cat([g_tr, g_te]).cpu(), y_all)
    ntr = len(ytr)
    df["split"] = np.where(df["vb_index"] < ntr, "train", "test")
    df["split_index"] = np.where(df["vb_index"] < ntr, df["vb_index"], df["vb_index"] - ntr)
    df.to_csv(os.path.join(OUT, "vb_poi_match.csv"), index=False)
    geo = {"n_samples": int(len(y_all)), "n_matched": int(len(df)), "n_confident": int(df["confident"].sum()),
           "n_mutual_best": int(df["mutual_best"].sum()), "median_margin": float(df["margin"].median()),
           "revisit_alignment_median": float(df["revisit_alignment"].median()), "revisit_alignment_mean": float(df["revisit_alignment"].mean()),
           "distinct_pois": int(df["poi"].nunique()), "provider_agreement": float(df["provider_agrees"].mean()),
           "provider_agreement_confident": float(df.loc[df["confident"], "provider_agrees"].mean())}
    save_json("vb_geo.json", geo)

    conf = df[df["confident"]].sort_values("vb_index")
    idx = conf["vb_index"].to_numpy()
    XYZ = np.full((len(y_all), 3), np.nan)
    XYZ[idx] = xyz(conf["lon"].to_numpy(), conf["lat"].to_numpy())
    lonlat = np.full((len(y_all), 2), np.nan)
    lonlat[idx] = conf[["lon", "lat"]].to_numpy()
    tr_ok = np.array([i for i in idx if i < ntr])
    te_ok = np.array([i for i in idx if i >= ntr])

    # ---- (D1) location-only random forest on the official split (matched samples)
    rf = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(lonlat[tr_ok], y_all[tr_ok])
    p = np.zeros((len(te_ok), ncls))
    p[:, rf.classes_] = rf.predict_proba(lonlat[te_ok])
    geo["official_location_rf"] = {"acc": float((p.argmax(1) == y_all[te_ok]).mean() * 100), "auc": auc_macro_ovr(y_all[te_ok], p) * 100, "n_test": int(len(te_ok))}
    print("official split, location-only RF", geo["official_location_rf"], flush=True)
    alla = df.sort_values("vb_index")  # same on every matched sample, confident or not
    ia = alla["vb_index"].to_numpy()
    tra, tea = ia[ia < ntr], ia[ia >= ntr]
    ll = np.full((len(y_all), 2), np.nan)
    ll[ia] = alla[["lon", "lat"]].to_numpy()
    rf = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(ll[tra], y_all[tra])
    p = np.zeros((len(tea), ncls))
    p[:, rf.classes_] = rf.predict_proba(ll[tea])
    geo["official_location_rf_all_matched"] = {"acc": float((p.argmax(1) == y_all[tea]).mean() * 100), "auc": auc_macro_ovr(y_all[tea], p) * 100, "n_test": int(len(tea))}
    print("official split, location-only RF, all matched", geo["official_location_rf_all_matched"], flush=True)

    # ---- (D2) nearest-neighbour geolocation from the frozen features (official split, matched samples)
    geo["nn_geolocation"] = {}
    for n in names:
        Ftr, Fte = load_feats(n)
        Fall = torch.cat([Ftr, Fte])
        geo["nn_geolocation"][n] = nn_geolocation(Fall[torch.as_tensor(tr_ok, device=DEV)], Fall[torch.as_tensor(te_ok, device=DEV)], XYZ[tr_ok], XYZ[te_ok])
        print("NN geolocation", n, geo["nn_geolocation"][n], flush=True)
    save_json("vb_geo.json", geo)

    # ---- (D3) random vs spatial 5-fold CV on the pooled matched samples
    yc = y_all[idx]
    folds = {"random": StratifiedKFold(5, shuffle=True, random_state=0), "spatial": spatial_folds(XYZ[idx])}
    rnd = np.zeros(len(idx), int)
    for f, (_, te) in enumerate(folds["random"].split(np.zeros(len(idx)), yc)):
        rnd[te] = f
    folds["random"] = rnd
    geo["cv"] = {"fold_sizes": {k: np.bincount(v, minlength=5).tolist() for k, v in folds.items()},
                 "majority_rate": {k: [float(np.bincount(yc[v == f], minlength=ncls).max() / (v == f).sum() * 100) for f in range(5)] for k, v in folds.items()}}
    yct = torch.as_tensor(yc, device=DEV)
    for reg, fold in folds.items():
        items = cv_items(fold)
        res = {}
        runs = []
        for f, (tr, va, te) in enumerate(items):
            c = RandomForestClassifier(500, random_state=f, n_jobs=-1).fit(lonlat[idx][tr], yc[tr])
            pp = np.zeros((len(te), ncls))
            pp[:, c.classes_] = c.predict_proba(lonlat[idx][te])
            runs.append({"acc": float((pp.argmax(1) == yc[te]).mean()), "auc": auc_macro_ovr(yc[te], pp)})
        res["location_rf"] = summarize(runs)
        print(f"[{reg}] location_rf {fmt(res['location_rf'])}", flush=True)
        for n in names:
            Ftr, Fte = load_feats(n)
            X = torch.cat([Ftr, Fte])[torch.as_tensor(idx, device=DEV)]
            runs = []
            for f, (tr, va, te) in enumerate(items):
                t = [torch.as_tensor(a, device=DEV) for a in (tr, va, te)]
                m = train_head(X[t[0]], yct[t[0]], X[t[1]], yct[t[1]], ncls, arch="mlp2", seed=f)
                acc, auc = evaluate(m, X[t[2]], yct[t[2]])
                runs.append({"acc": acc, "auc": auc})
            res[n] = summarize(runs)
            print(f"[{reg}] {n:22s} {fmt(res[n])}", flush=True)
        geo["cv"][reg] = res
        save_json("vb_geo.json", geo)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
