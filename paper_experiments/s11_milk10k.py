"""Stage 11 - MILK10k: a naturally two-party medical image VFL task (clinical close-up photo + dermoscopy of a lesion).

MILK10k (ISIC 2025) pairs, for 5,240 skin lesions, a clinical close-up photograph and a dermoscopic image, with
patient metadata and a histopathology/expert diagnosis (11 classes).  Scenario: the referring practitioner holds the
clinical photo, the dermatologist holds the dermoscopy, (optionally) a third party holds the metadata; the label
owner holds the diagnosis.  We group the 11 diagnoses into 7 classes (the 5 rare ones, 9-52 lesions each, form
"other") and use a stratified 5-fold cross-validation over lesions (fold f = test, fold f+1 = validation).
Metrics: macro F1 (primary, the classes are imbalanced: BCC is 48 %), balanced accuracy, accuracy, macro AUC.

  python s11_milk10k.py prepare            download + decode (224 px, short side resize + centre crop)
  python s11_milk10k.py features           frozen features (encoders below), full images and left/right halves
  python s11_milk10k.py heads              one-shot heads on every party combination   -> milk_heads.json
  python s11_milk10k.py split              split learning: ResNet-18 scratch/ImageNet, ViT-B LoRA / fine-tuned
                                            (the price of freezing, same backbone)    -> milk_split.json
"""

import argparse
import glob
import os
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from pe_common import DEV, fmt, metrics_from_proba, predict_proba, robust_download, save_json, summarize, train_head

ROOT = os.environ.get("PE_MILK", "/content/milk")
S3 = "https://isic-archive.s3.amazonaws.com/challenges/milk10k"
DOI = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456"
CLASSES11 = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
CLASSES7 = ["BCC", "NV", "BKL", "SCCKA", "MEL", "AKIEC", "OTHER"]
TO7 = {c: (CLASSES7.index(c) if c in CLASSES7 else 6) for c in CLASSES11}
ENCODERS = ["dinov2_g", "clip_l14", "dinov2_b", "clip_b16", "biomedclip", "monet"]
SIZE = 224


def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)


# ------------------------------------------------------------------ data
def _decode(path):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(im.crop((l, t, l + SIZE, t + SIZE))).transpose(2, 0, 1).copy()


def prepare():
    os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(f"{ROOT}/lesions.csv") and os.path.exists(f"{ROOT}/img_derm.npy"):
        return
    try:
        robust_download(f"{S3}/MILK10k_Training_Metadata.csv", f"{ROOT}/meta.csv", 100_000)
        robust_download(f"{S3}/MILK10k_Training_GroundTruth.csv", f"{ROOT}/gt.csv", 100_000)
        robust_download(f"{S3}/MILK10k_Training_Input.zip", f"{ROOT}/input.zip", 300_000_000)
        sh(f"cd {ROOT} && unzip -q -o input.zip")
    except Exception as e:  # fall back to the DOI bundle (same images, ISIC-archive metadata)
        print("!! challenge files unavailable, using the DOI bundle:", repr(e), flush=True)
        robust_download(f"{DOI}/milk10k.zip", f"{ROOT}/milk10k.zip", 350_000_000)
        sh(f"cd {ROOT} && unzip -q -o milk10k.zip")
        sh(f"cp {ROOT}/supplements/training_gt.csv {ROOT}/gt.csv; cp {ROOT}/metadata.csv {ROOT}/meta.csv")
    meta, gt = pd.read_csv(f"{ROOT}/meta.csv"), pd.read_csv(f"{ROOT}/gt.csv")
    print("meta columns:", list(meta.columns)[:12], "gt columns:", list(gt.columns), flush=True)
    paths = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{ROOT}/**/ISIC_*.jpg", recursive=True)}
    meta["kind"] = np.where(meta["image_type"].str.startswith("derm"), "derm", "clin")
    rows = []
    for lid, g in meta.groupby("lesion_id"):
        k = dict(zip(g["kind"], g["isic_id"]))
        if "derm" not in k or "clin" not in k or k["derm"] not in paths or k["clin"] not in paths:
            continue
        r = g.iloc[0]
        rows.append({"lesion_id": lid, "clin": k["clin"], "derm": k["derm"], "age": r.get("age_approx"), "sex": r.get("sex"),
                     "site": r.get("site", r.get("anatom_site_general")), "skin_tone": r.get("skin_tone_class")})
    les = pd.DataFrame(rows).merge(gt, on="lesion_id")
    les["dx11"] = les[CLASSES11].to_numpy().argmax(1)
    les["dx7"] = [TO7[CLASSES11[i]] for i in les["dx11"]]
    print(len(les), "lesions; 7-class counts", np.bincount(les["dx7"], minlength=7).tolist(), flush=True)
    from multiprocessing import Pool

    for kind in ("clin", "derm"):
        t = time.time()
        with Pool(min(16, os.cpu_count() or 4)) as pool:
            arr = np.stack(pool.map(_decode, [paths[i] for i in les[kind]], chunksize=32))
        np.save(f"{ROOT}/img_{kind}.npy", arr)
        print(kind, arr.shape, f"{time.time() - t:.0f}s", flush=True)
    les.drop(columns=CLASSES11).to_csv(f"{ROOT}/lesions.csv", index=False)


def lesions():
    return pd.read_csv(f"{ROOT}/lesions.csv")


def meta_features(les):
    """Third party: age (standardised, missing -> 0 + indicator), sex, anatomical site, skin tone (one-hot)."""
    age = pd.to_numeric(les["age"], errors="coerce")
    cols = [((age - age.mean()) / age.std()).fillna(0).to_numpy()[:, None], age.isna().to_numpy()[:, None].astype(float)]
    for c in ("sex", "site", "skin_tone"):
        cols.append(pd.get_dummies(les[c].astype(str)).to_numpy().astype(float))
    return np.concatenate(cols, 1).astype(np.float32)


def folds(y, k=5):
    from sklearn.model_selection import StratifiedKFold

    f = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=0).split(np.zeros(len(y)), y)):
        f[te] = i
    return [(np.where((f != i) & (f != (i + 1) % k))[0], np.where(f == (i + 1) % k)[0], np.where(f == i)[0]) for i in range(k)]


# ------------------------------------------------------------------ frozen features
def features(encoders):
    from pe_encoders import encoder, extract

    imgs = {k: np.load(f"{ROOT}/img_{k}.npy", mmap_mode="r") for k in ("clin", "derm")}
    left, right = (lambda x: x[..., : SIZE // 2]), (lambda x: x[..., SIZE // 2 :])
    for name in encoders:
        todo = [(k, v, c) for k in ("clin", "derm") for v, c in ((k, None), (k + "_left", left), (k + "_right", right))
                if not os.path.exists(f"{ROOT}/f_{name}_{v}.npy")]
        if not todo:
            continue
        try:
            fn, dim = encoder(name)
        except Exception as e:
            print("!! encoder unavailable", name, repr(e), flush=True)
            continue
        for k, v, c in todo:
            t = time.time()
            a = extract(fn, imgs[k], SIZE, crop=c)
            assert np.isfinite(a.astype(np.float32)).all(), (name, v)
            np.save(f"{ROOT}/f_{name}_{v}.npy", a)
            print(f"{name} {v} {a.shape} {time.time() - t:.0f}s", flush=True)
        del fn
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ one-shot heads
CONFIGS = {"clin": ["clin"], "derm": ["derm"], "meta": ["meta"], "clin+derm": ["clin", "derm"], "derm+meta": ["derm", "meta"],
           "clin+meta": ["clin", "meta"], "clin+derm+meta": ["clin", "derm", "meta"],
           "derm halves": ["derm_left", "derm_right"], "derm left": ["derm_left"], "clin halves": ["clin_left", "clin_right"]}
VARIANTS = [("mlp2", False), ("mlp2", True), ("linear", True)]  # (head, class-balanced loss)


def class_weights(y, n):
    c = np.bincount(y, minlength=n).astype(float)
    return (c.sum() / (n * np.maximum(c, 1))).tolist()


def zscore(X, tr):
    m, s = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - m) / s


def heads(encoders):
    les = lesions()
    y = les["dx7"].to_numpy()
    F = folds(y)
    meta = meta_features(les)
    path = os.path.join(os.environ.get("PE_OUT", "/content/results"), "milk_heads.json")
    res = __import__("json").load(open(path)) if os.path.exists(path) else {}
    res["n_lesions"], res["class_counts"], res["classes"] = int(len(y)), np.bincount(y, minlength=7).tolist(), CLASSES7
    res["fold_test_counts"] = [np.bincount(y[te], minlength=7).tolist() for _, _, te in F]
    yt = torch.as_tensor(y, device=DEV)
    for name in encoders:
        if not os.path.exists(f"{ROOT}/f_{name}_derm.npy"):
            continue
        feats = {v: np.load(f"{ROOT}/f_{name}_{v}.npy").astype(np.float32) for v in ("clin", "derm", "clin_left", "clin_right", "derm_left", "derm_right")}
        feats["meta"] = meta
        out = res.setdefault(name, {})
        for cname, parts in CONFIGS.items():
            for arch, bal in VARIANTS:
                key = f"{cname}|{arch}|{'bal' if bal else 'plain'}"
                if key in out or (name != ENCODERS[0] and parts == ["meta"]):
                    continue
                runs = []
                for f, (tr, va, te) in enumerate(F):
                    X = torch.as_tensor(np.concatenate([zscore(feats[p], tr) for p in parts], 1), device=DEV)
                    m = train_head(X[tr], yt[tr], X[va], yt[va], 7, arch=arch, seed=f, class_weight=class_weights(y[tr], 7) if bal else None)
                    runs.append(metrics_from_proba(y[te], predict_proba(m, X[te]).cpu().numpy()))
                out[key] = summarize(runs)
                print(f"{name:10s} {key:32s} {fmt(out[key])}", flush=True)
            save_json("milk_heads.json", res)
    save_json("milk_heads.json", res)


# ------------------------------------------------------------------ split learning (price of freezing)
def split(kinds_list, epochs_of):
    from pe_split import train_split

    les = lesions()
    y = les["dx7"].to_numpy()
    F = folds(y)
    parties = [np.load(f"{ROOT}/img_clin.npy"), np.load(f"{ROOT}/img_derm.npy")]
    path = os.path.join(os.environ.get("PE_OUT", "/content/results"), "milk_split.json")
    res = __import__("json").load(open(path)) if os.path.exists(path) else {}
    for kind in kinds_list:
        if kind in res:
            continue
        runs, infos = [], []
        for f, (tr, va, te) in enumerate(F):
            m, info = train_split(parties, y, tr, va, te, [kind, kind], 7, epochs=epochs_of(kind), bs=64, seed=f,
                                  class_weight=class_weights(y[tr], 7), log=lambda s: print(kind, f, s, flush=True))
            info.pop("test_proba")
            runs.append(m)
            infos.append(info)
            print(kind, "fold", f, {k: round(v * 100, 2) for k, v in m.items()}, f"{info['secs']:.0f}s", flush=True)
        res[kind] = summarize(runs)
        res[kind].update({"secs_per_fold": [i["secs"] for i in infos], "bytes_per_train_sample": infos[0]["bytes_per_train_sample"],
                          "epochs": infos[0]["epochs"], "trainable_params": infos[0]["trainable_params"], "peak_mem_gb": max(i["peak_mem_gb"] for i in infos)})
        print(kind, fmt(res[kind]), flush=True)
        save_json("milk_split.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "features", "heads", "split", "all"])
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--kinds", default="resnet18,resnet18_in,dinov2_b_lora,clip_b16_lora,dinov2_b_ft")
    a = ap.parse_args()
    enc = a.encoders.split(",")
    if a.stage in ("prepare", "all"):
        prepare()
    if a.stage in ("features", "all"):
        features(enc)
    if a.stage in ("heads", "all"):
        heads(enc)
    if a.stage in ("split", "all"):
        split(a.kinds.split(","), lambda k: 30 if k == "resnet18" else 20 if k == "resnet18_in" else 10)


if __name__ == "__main__":
    main()
