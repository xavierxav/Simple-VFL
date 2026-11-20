"""Stage 15 - WorldStrat with two genuinely different sensors: when does the second party help?

Party A holds the SPOT 6/7 pan-sharpened image of a 1.58 km area (1.5 m), party B the 16 Sentinel-2 L1C revisits
(10 m, 13 bands); the label owner holds one of three labels of the WorldStrat metadata:
  ipcc   IPCC land-cover class (7 classes)
  smod   GHSL degree of urbanisation (8 classes: rural very low density ... urban centre, water)
  lccs   ESA-CCI LCCS land-cover class (the 18 classes with >= 40 areas; the others are left out)
Encoders (frozen): DINOv2-g on SPOT RGB at 1.5 / 3 / 6 / 10 m ground sampling (input 1050 / 518 / 266 / 168 px) and
CLIP-L at 224 px; for every Sentinel-2 revisit DINOv2-g on RGB, on a NIR false-colour composite (B8-B4-B3) and on a
SWIR composite (B12-B8A-B4), the Sentinel-2 model SSL4EO-S12 ViT-S/16 on all 13 bands, CLIP-L on RGB.  The Sentinel
party averages its 16 revisit features locally and uploads one vector per area.
Protocol: stratified 5-fold cross-validation (fold f test, f+1 validation), class-balanced loss, macro F1 (primary),
balanced accuracy, accuracy, macro AUC.

  python s15_worldstrat_sensors.py prepare    download (Zenodo, ~69 GB) and convert to compact arrays
  python s15_worldstrat_sensors.py features
  python s15_worldstrat_sensors.py heads      -> ws_sensors.json, he_ds_worldstrat_*.npz
"""

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pe_common import DEV, evaluate_all, fmt, robust_download, save_json, summarize, train_head

ROOT = os.environ.get("PE_WS2", "/content/ws2")
OUT = os.environ.get("PE_OUT", "/content/results")
ZEN = "https://zenodo.org/records/6810792/files"
EXCLUDED = ["ASMSpotter-1-1-1", "Landcover-1062091", "UNHCR-HUNs002665"]
GSD = {"1.5m": 1050, "3m": 518, "6m": 266, "10m": 168}
NIR, SWIR, RGB = [7, 3, 2], [12, 8, 3], [3, 2, 1]  # L1C band order B1..B8, B8A, B9..B12


def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)


def meta():
    robust_download(f"{ZEN}/metadata.csv?download=1", f"{ROOT}/metadata.csv", 10_000_000)
    df = pd.read_csv(f"{ROOT}/metadata.csv", sep=None, engine="python")
    df = df.rename(columns={df.columns[0]: "AOI"})
    df = df.loc[~df["AOI"].isin(EXCLUDED)].drop_duplicates("AOI").reset_index(drop=True)
    return df[["AOI", "IPCC Class", "SMOD Class", "LCCS class"]]


def _stream(path):
    if subprocess.run("which pigz", shell=True, capture_output=True).returncode != 0:
        subprocess.run("apt-get install -y -q pigz > /dev/null 2>&1", shell=True)
    p = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 24)
    return p, tarfile.open(fileobj=p.stdout, mode="r|")


def prepare():
    import tifffile

    os.makedirs(ROOT, exist_ok=True)
    df = meta()
    aoi = {a: i for i, a in enumerate(df["AOI"])}
    N = len(df)
    import shutil

    print("free disk GB:", shutil.disk_usage(ROOT).free / 1e9, flush=True)
    if not os.path.exists(f"{ROOT}/spot_done"):
        robust_download(f"{ZEN}/hr_dataset.tar.gz?download=1", f"{ROOT}/hr.tar.gz", 41_000_000_000)
        SP = np.lib.format.open_memmap(f"{ROOT}/spot_rgb.npy", "w+", np.uint16, (N, 3, 1050, 1050))
        seen = np.zeros(N, bool)
        p, tf = _stream(f"{ROOT}/hr.tar.gz")
        t = time.time()
        for m in tf:
            if not m.name.endswith("_ps.tiff"):
                continue
            name = os.path.basename(m.name)[: -len("_ps.tiff")]
            if name not in aoi:
                continue
            a = tifffile.imread(io.BytesIO(tf.extractfile(m).read()))  # [1054,1054,4] uint16 pan-sharpened RGBN
            SP[aoi[name]] = a[2:1052, 2:1052, :3].transpose(2, 0, 1)
            seen[aoi[name]] = True
            if seen.sum() % 500 == 0:
                print(f"SPOT {seen.sum()} {time.time() - t:.0f}s", flush=True)
        p.wait()
        SP.flush()
        np.save(f"{ROOT}/spot_seen.npy", seen)
        print("SPOT done", seen.sum(), "of", N, flush=True)
        open(f"{ROOT}/spot_done", "w").close()
        os.remove(f"{ROOT}/hr.tar.gz")
    if not os.path.exists(f"{ROOT}/s2_done"):
        robust_download(f"{ZEN}/lr_dataset_l1c.tar.gz?download=1", f"{ROOT}/l1c.tar.gz", 27_000_000_000)
        S2 = np.lib.format.open_memmap(f"{ROOT}/s2_l1c.npy", "w+", np.uint8, (N, 16, 13, 158, 158))
        seen = np.zeros((N, 16), bool)
        p, tf = _stream(f"{ROOT}/l1c.tar.gz")
        t = time.time()
        for m in tf:
            mm = re.search(r"([^/]+)-(\d+)-L1C_data\.tiff$", m.name)
            if not mm or mm.group(1) not in aoi:
                continue
            i, r = aoi[mm.group(1)], int(mm.group(2)) - 1
            if r >= 16:
                continue
            try:
                a = tifffile.imread(io.BytesIO(tf.extractfile(m).read()))  # [159,158,13] float32 reflectance
            except Exception as e:  # a corrupt tile: leave the revisit unseen (its area is then left out)
                print("!! unreadable", m.name, repr(e), flush=True)
                continue
            a = a[:158, :158]
            if a.shape[:2] != (158, 158):  # a few tiles are 159 x 157: pad the missing column/row by edge replication
                a = np.pad(a, ((0, 158 - a.shape[0]), (0, 158 - a.shape[1]), (0, 0)), mode="edge")
            a = np.nan_to_num(a.transpose(2, 0, 1), nan=0.0)
            S2[i, r] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)  # reflectance 0..1 in 8 bits
            seen[i, r] = True
            if seen.sum() % 5000 == 0:
                print(f"L1C {seen.sum()} images {time.time() - t:.0f}s", flush=True)
        p.wait()
        S2.flush()
        np.save(f"{ROOT}/s2_seen.npy", seen)
        print("L1C done", seen.sum(), "of", seen.size, flush=True)
        open(f"{ROOT}/s2_done", "w").close()
        os.remove(f"{ROOT}/l1c.tar.gz")
    df.to_csv(f"{ROOT}/aoi.csv", index=False)


# ------------------------------------------------------------------ features
def band_stats(A, axes, n=400):
    idx = np.sort(np.random.default_rng(0).choice(len(A), min(n, len(A)), replace=False))
    s = np.asarray(A[idx], dtype=np.float32)
    return s.mean(axis=axes), s.std(axis=axes) + 1e-6


def features():
    from transformers import AutoModel, CLIPVisionModel

    SP = np.load(f"{ROOT}/spot_rgb.npy", mmap_mode="r")
    S2 = np.load(f"{ROOT}/s2_l1c.npy", mmap_mode="r")
    N = len(SP)
    sp_m, sp_s = band_stats(SP, (0, 2, 3), 200)
    s2_m, s2_s = band_stats(S2, (0, 1, 3, 4), 200)
    np.savez(f"{ROOT}/band_stats.npz", sp_m=sp_m, sp_s=sp_s, s2_m=s2_m, s2_s=s2_s)
    spm, sps = torch.tensor(sp_m, device=DEV)[None, :, None, None], torch.tensor(sp_s, device=DEV)[None, :, None, None]
    s2m, s2s = torch.tensor(s2_m, device=DEV)[None, :, None, None], torch.tensor(s2_s, device=DEV)[None, :, None, None]
    g = AutoModel.from_pretrained("facebook/dinov2-giant").to(DEV).eval()
    cl = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    cost = {}

    def run(path, fn, n, d, bs, load):
        if os.path.exists(path):
            return
        out = np.zeros((n, d), np.float16)
        t = time.time()
        for i in range(0, n, bs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out[i : i + bs] = fn(load(i, min(n, i + bs))).float().cpu().numpy()
        torch.cuda.synchronize()
        cost[os.path.basename(path)] = time.time() - t
        np.save(path, out)
        print(path, out.shape, f"{time.time() - t:.0f}s", flush=True)

    def spot(i, j, size):
        x = (torch.from_numpy(np.ascontiguousarray(SP[i:j])).to(DEV).float() - spm) / sps
        return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False) if size != 1050 else x

    for k, size in GSD.items():
        run(f"{ROOT}/f_spot_dinov2_g_{k}.npy", lambda x: g(pixel_values=x).pooler_output, N, 1536, 4 if size > 600 else 32,
            lambda i, j, size=size: spot(i, j, size))
    run(f"{ROOT}/f_spot_clip_l14_224.npy", lambda x: cl(pixel_values=x, interpolate_pos_encoding=False).pooler_output, N, 1024, 64,
        lambda i, j: spot(i, j, 224))

    s2flat = S2.reshape(N * 16, 13, 158, 158)

    def s2(i, j, bands):
        x = (torch.from_numpy(np.ascontiguousarray(s2flat[i:j])).to(DEV).float() - s2m) / s2s
        return F.interpolate(x[:, bands], size=(224, 224), mode="bilinear", align_corners=False)

    for name, bands in (("rgb", RGB), ("nir", NIR), ("swir", SWIR)):
        run(f"{ROOT}/f_s2_dinov2_g_{name}.npy", lambda x: g(pixel_values=x).pooler_output, N * 16, 1536, 128, lambda i, j, b=bands: s2(i, j, b))
    run(f"{ROOT}/f_s2_clip_l14_rgb.npy", lambda x: cl(pixel_values=x).pooler_output, N * 16, 1024, 128, lambda i, j: s2(i, j, RGB))
    del g, cl
    torch.cuda.empty_cache()
    try:
        subprocess.run("pip install -q torchgeo", shell=True)
        from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224

        ss = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO, num_classes=0).to(DEV).eval()
        run(f"{ROOT}/f_s2_ssl4eo.npy", ss, N * 16, 384, 128, lambda i, j: s2(i, j, list(range(13))))
    except Exception as e:
        print("!! SSL4EO unavailable", repr(e), flush=True)
    json.dump(cost, open(f"{ROOT}/feature_secs.json", "w"))


# ------------------------------------------------------------------ heads
def labels(df, task):
    if task == "ipcc":
        col = df["IPCC Class"].fillna("Other").astype(str)
    elif task == "smod":
        col = df["SMOD Class"].astype(str)
    else:
        col = df["LCCS class"].astype(str)
    counts = col.value_counts()
    keep = counts[counts >= 40].index if task == "lccs" else counts.index
    classes = sorted(keep)
    y = np.array([classes.index(c) if c in classes else -1 for c in col])
    return y, classes


def folds(y, k=5):
    from sklearn.model_selection import StratifiedKFold

    f = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        f[te] = i
    return f


def heads():
    df = pd.read_csv(f"{ROOT}/aoi.csv")
    N = len(df)
    ok = np.load(f"{ROOT}/spot_seen.npy") & np.load(f"{ROOT}/s2_seen.npy").all(1)

    def s2mean(name):
        return np.load(f"{ROOT}/f_s2_{name}.npy").astype(np.float32).reshape(N, 16, -1).mean(1)

    P = {f"SPOT {k}": np.load(f"{ROOT}/f_spot_dinov2_g_{k}.npy").astype(np.float32) for k in GSD}
    P["SPOT CLIP-L"] = np.load(f"{ROOT}/f_spot_clip_l14_224.npy").astype(np.float32)
    P["S2 RGB"] = s2mean("dinov2_g_rgb")
    P["S2 MS"] = np.concatenate([P["S2 RGB"], s2mean("dinov2_g_nir"), s2mean("dinov2_g_swir")], 1)
    P["S2 CLIP-L"] = s2mean("clip_l14_rgb")
    if os.path.exists(f"{ROOT}/f_s2_ssl4eo.npy"):
        P["S2 SSL4EO"] = s2mean("ssl4eo")
    configs = {k: [k] for k in P}
    for k in GSD:
        for s in ("S2 RGB", "S2 MS", "S2 SSL4EO"):
            if s in P:
                configs[f"SPOT {k} + {s}"] = [f"SPOT {k}", s]
    configs["SPOT CLIP-L + S2 CLIP-L"] = ["SPOT CLIP-L", "S2 CLIP-L"]
    path = os.path.join(OUT, "ws_sensors.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for task in ("ipcc", "smod", "lccs"):
        y_all, classes = labels(df, task)
        sel = np.where(ok & (y_all >= 0))[0]
        y = y_all[sel]
        C = len(classes)
        f = folds(y)
        out = res.setdefault(task, {})
        out["classes"], out["counts"] = classes, np.bincount(y, minlength=C).tolist()
        yt = torch.as_tensor(y, device=DEV)
        for cname, parts in configs.items():
            for arch in ("mlp2", "linear"):
                key = f"{cname}|{arch}"
                if key in out or (arch == "linear" and "+" not in cname and cname not in ("SPOT 1.5m", "S2 MS")):
                    continue
                runs = []
                for k in range(5):
                    tr, va, te = np.where((f != k) & (f != (k + 1) % 5))[0], np.where(f == (k + 1) % 5)[0], np.where(f == k)[0]
                    Z = []
                    for p in parts:
                        A = P[p][sel]
                        Z.append((A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6))
                    X = torch.as_tensor(np.concatenate(Z, 1), device=DEV)
                    cnt = np.bincount(y[tr], minlength=C).astype(float)
                    m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch=arch, seed=k, class_weight=(cnt.sum() / (C * np.maximum(cnt, 1))).tolist())
                    runs.append(evaluate_all(m, X[te], yt[te]))
                out[key] = summarize(runs)
                print(f"{task:5s} {key:34s} {fmt(out[key])}", flush=True)
            save_json("ws_sensors.json", res)
        if task == "ipcc":
            for a, b in (("SPOT 1.5m", "S2 MS"), ("SPOT 6m", "S2 MS")):
                os.makedirs(os.environ.get("PE_HE_DS", "/content/he_ds"), exist_ok=True); np.savez(f"{os.environ.get('PE_HE_DS', '/content/he_ds')}/he_ds_worldstrat_ipcc_{a.split()[1]}_{b.replace(' ', '')}.npz", n_parties=2, p0=P[a][sel].astype(np.float16),
                         p1=P[b][sel].astype(np.float16), y=y, fold=f, metric="f1")
    save_json("ws_sensors.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "features", "heads", "all"])
    a = ap.parse_args()
    if a.stage in ("prepare", "all"):
        prepare()
    if a.stage in ("features", "all"):
        features()
    if a.stage in ("heads", "all"):
        heads()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
