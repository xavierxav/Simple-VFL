"""Stage 5 - WorldStrat (SPOT + Sentinel) re-evaluation on the cached DINOv2-g features.

Adds what the original repo lacked: a geographically disjoint (spatial block) split, a one-shot SPOT+Sentinel
concatenation (no tuned fusion weight), a capacity-matched SPOT control, and a location-only baseline evaluated
under the same two protocols. Writes $PE_OUT/ws_results.json.

usage: python s5_worldstrat.py [--labels sources,ipcc,simplified_ipcc]
"""

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pe_common import DEV, auc_macro_ovr, evaluate, fmt, robust_download, save_json, summarize, train_head

ROOT = os.environ.get("PE_WS", "/content/ws")
HF = "https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main"
META_URL = "https://zenodo.org/records/6810792/files/metadata.csv?download=1"
EXCLUDED = ["ASMSpotter-1-1-1", "Landcover-1062091", "UNHCR-HUNs002665"]
SOURCES = {"Amnesty POI": 0, "ASMSpotter": 1, "Landcover": 2, "UNHCR": 3}
IPCC = {"Settlement": 0, "Forest": 1, "Agriculture": 2, "Other": 3, "Grassland": 4, "Water": 5, "Wetland": 6}
SIMPLE = {"Settlement": 0, "Forest": 1, "Agriculture": 2, "Other": 3, "Grassland": 3, "Water": 3, "Wetland": 3}
MAPS = {"sources": (SOURCES, 4), "ipcc": (IPCC, 7), "simplified_ipcc": (SIMPLE, 4)}


def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)


def ensure_data():
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(f"{ROOT}/metadata.csv"):
        robust_download(META_URL, f"{ROOT}/metadata.csv", 1000)
    if not glob.glob(f"{ROOT}/sent/**/*_class_token.npy", recursive=True):
        robust_download(f"{HF}/extracted_features_giant_rgb_light_preprocess.zip", f"{ROOT}/sent.zip", 388_000_000)
        sh(f"mkdir -p {ROOT}/sent && cd {ROOT}/sent && unzip -q -o ../sent.zip")
    if not glob.glob(f"{ROOT}/spot/**/*_class_token.npy", recursive=True):
        robust_download(f"{HF}/extracted_features_giant_ps_normalized.7z", f"{ROOT}/spot.7z", 22_000_000)
        sh("which 7z >/dev/null 2>&1 || apt-get install -y -q p7zip-full >/dev/null 2>&1")
        sh(f"mkdir -p {ROOT}/spot && cd {ROOT}/spot && 7z x -y -bso0 -bsp0 ../spot.7z")


def load_all():
    ensure_data()
    df = pd.read_csv(f"{ROOT}/metadata.csv", sep=None, engine="python")
    df = df.rename(columns={df.columns[0]: "POI"})
    df = df.loc[~df["POI"].isin(EXCLUDED)]
    meta = df[["POI", "IPCC Class", "lon", "lat"]].drop_duplicates().reset_index(drop=True)
    idx = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/**/*_class_token.npy", recursive=True)}
    spot, sent = [], []
    for poi in meta["POI"]:
        spot.append(np.load(idx[f"{poi}_ps_class_token.npy"]).reshape(-1))
        sent.append(np.stack([np.load(idx[f"{poi}-{n}-L2A_data_class_token.npy"]).reshape(-1) for n in range(1, 17)]))
    print("POIs", len(meta), "spot", np.stack(spot).shape, "sent", np.stack(sent).shape, flush=True)
    return meta, torch.tensor(np.stack(spot), dtype=torch.float32), torch.tensor(np.stack(sent), dtype=torch.float32)


def labels_of(meta, label_type):
    m, ncls = MAPS[label_type]
    key = (lambda r: r["POI"].split("-")[0]) if label_type == "sources" else (lambda r: r["IPCC Class"])
    other = m.get("Other")  # IPCC labels: missing/unknown -> "Other"; sources: every POI prefix must be known
    y = np.array([m.get(key(r), other) if isinstance(key(r), str) else other for _, r in meta.iterrows()])
    assert all(v is not None for v in y), "unknown label"
    return y.astype(np.int64), ncls


# ---------------------------------------------------------------- splits
def random_split(y, seed):
    """Same recipe as simple_vfl.dataset.load_dataset.split_satellite_metadata."""
    tv, te = train_test_split(range(len(y)), test_size=0.15, random_state=42, stratify=y)
    tr, va = train_test_split(tv, test_size=0.2, random_state=seed, stratify=y[tv])
    return np.array(tr), np.array(va), np.array(te)


def spatial_folds(meta, n_clusters=25, n_folds=5):
    """Whole spatial clusters go to one fold (greedy size balancing): test POIs share no region with train POIs."""
    lon, lat = np.radians(meta["lon"].to_numpy()), np.radians(meta["lat"].to_numpy())
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    cl = KMeans(n_clusters, n_init=10, random_state=0).fit_predict(xyz)
    sizes = np.bincount(cl, minlength=n_clusters)
    fold_of, load = {}, np.zeros(n_folds)
    for c in np.argsort(-sizes):
        f = int(np.argmin(load))
        fold_of[c] = f
        load[f] += sizes[c]
    fold = np.array([fold_of[c] for c in cl])
    out = []
    for f in range(n_folds):
        te = np.where(fold == f)[0]
        va = np.where(fold == (f + 1) % n_folds)[0]
        tr = np.where((fold != f) & (fold != (f + 1) % n_folds))[0]
        out.append((tr, va, te))
    return out, cl


# ---------------------------------------------------------------- settings
def pick_view(X, g):  # X [N,V,d] -> random revisit [N,d]
    return X[torch.arange(len(X), device=DEV), torch.randint(0, X.shape[1], (len(X),), generator=g).to(DEV)]


def settings(spot, sent):
    S = spot[:, None]  # [N,1,d]
    both = torch.cat([S, sent], 1)  # [N,17,d]
    rnd_view = lambda xb: xb[torch.arange(len(xb), device=DEV), torch.randint(0, xb.shape[1], (len(xb),), device=DEV)]  # noqa: E731

    def rnd_spot_plus_view(xb):  # SPOT + one random Sentinel revisit -> [B,2,d]
        v = xb[torch.arange(len(xb), device=DEV), torch.randint(1, 17, (len(xb),), device=DEV)]
        return torch.stack([xb[:, 0], v], 1)

    def fixed_spot_plus_view(X):
        g = torch.Generator(device="cpu").manual_seed(321)
        v = X[torch.arange(len(X), device=DEV), torch.randint(1, 17, (len(X),), generator=g).to(DEV)]
        return torch.stack([X[:, 0], v], 1)

    return {
        # name: (X, arch, hidden, train_transform, eval_transform(X)->X)
        "spot": (S, "mlp2", 1000, None, None),
        "spot_wide2000": (S, "mlp2", 2000, None, None),  # capacity control for the SPOT+Sentinel fusion
        "sent16_concat": (sent, "mlp2", 1000, None, None),
        "sent1": (sent, "mlp2", 1000, rnd_view, lambda X: pick_view(X, torch.Generator().manual_seed(321))),
        "spot_sent16_concat": (both, "mlp2", 1000, None, None),
        "spot_sent_mean2": (torch.stack([spot, sent.mean(1)], 1), "mlp2", 1000, None, None),
        "spot_sent1": (both, "mlp2", 1000, rnd_spot_plus_view, fixed_spot_plus_view),
        "spot_linear": (S, "linear", 0, None, None),
        "sent16_linear": (sent, "linear", 0, None, None),
    }


def fit_eval(X, y_t, tr, va, te, ncls, arch, hidden, tf, ev, seed):
    tr_t, va_t, te_t = (torch.as_tensor(a, device=DEV) for a in (tr, va, te))
    Xva, Xte = X[va_t], X[te_t]
    if ev is not None:
        Xva, Xte = ev(Xva), ev(Xte)
    m = train_head(X[tr_t], y_t[tr_t], Xva, y_t[va_t], ncls, arch=arch, hidden=hidden, seed=seed, train_transform=tf)
    acc, auc = evaluate(m, Xte, y_t[te_t])
    return {"acc": acc, "auc": auc}


def location_rf(meta, y, tr, te, ncls, seed):
    xy = meta[["lon", "lat"]].to_numpy()
    clf = RandomForestClassifier(n_estimators=500, random_state=42 + seed, n_jobs=-1).fit(xy[tr], y[tr])
    p = np.zeros((len(te), ncls))
    p[:, clf.classes_] = clf.predict_proba(xy[te])
    return {"acc": float((p.argmax(1) == y[te]).mean()), "auc": auc_macro_ovr(y[te], p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="ipcc,simplified_ipcc,sources")
    a = ap.parse_args()
    meta, spot, sent = load_all()
    spot, sent = spot.to(DEV), sent.to(DEV)
    S = settings(spot, sent)
    folds, clusters = spatial_folds(meta)
    print("spatial fold sizes (train/val/test):", [(len(t), len(v), len(e)) for t, v, e in folds], flush=True)
    prev = os.path.join(os.environ.get("PE_OUT", "/content/results"), "ws_results.json")
    results = json.load(open(prev)) if os.path.exists(prev) else {}  # merge: re-running one label keeps the others
    results.update({"n_pois": len(meta), "spatial_fold_sizes": [[len(t), len(v), len(e)] for t, v, e in folds]})
    for label in a.labels.split(","):
        y, ncls = labels_of(meta, label)
        y_t = torch.as_tensor(y, device=DEV)
        results[label] = {"class_counts": np.bincount(y, minlength=ncls).tolist()}
        regimes = {
            "random": [(random_split(y, s), s) for s in range(5)],
            "spatial": [(f, i) for i, f in enumerate(folds)],
        }
        for reg, items in regimes.items():
            res = {"location_rf": [location_rf(meta, y, tr, te, ncls, s) for (tr, va, te), s in items]}
            res["location_rf"] = summarize(res["location_rf"])
            print(f"[{label}/{reg}] location_rf", fmt(res["location_rf"]), flush=True)
            if reg == "spatial":
                res["test_class_counts"] = [np.bincount(y[te], minlength=ncls).tolist() for (_, _, te), _ in items]
            for name, (X, arch, hidden, tf, ev) in S.items():
                t = time.time()
                runs = [fit_eval(X, y_t, tr, va, te, ncls, arch, hidden, tf, ev, s) for (tr, va, te), s in items]
                res[name] = summarize(runs)
                print(f"[{label}/{reg}] {name:20s} {fmt(res[name])}  ({time.time() - t:.0f}s)", flush=True)
            results[label][reg] = res
            save_json("ws_results.json", results)
    save_json("ws_results.json", results)


if __name__ == "__main__":
    main()
