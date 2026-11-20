"""Stage 14 - the classic image-VFL benchmarks (artificially split images, multi-view objects), under the exact
protocols of the papers we compare with, so that their published numbers can be quoted next to ours.

CIFAR-10 / CIFAR-100 (50k train, 10k test; 10% of the training set held out for validation, 3 seeds):
  * VFLAIR (Zou et al., ICLR 2024): K=2 top/bottom halves (16x32), K=4 quadrants (16x16), ResNet-18 bottoms trained
    with FedSGD for 30-40 epochs (published: CIFAR-10 splitVFL 79.8 / 76.2 %, CIFAR-100 42.3 / 38.2 %).
  * FedOnce (Wu et al., 2022): K=4 quadrants, CIFAR-10 (one-shot 58.7 %, SplitNN 68.3 %, centralised 81.1 %).
  * Sun et al. (ICCV 2023), limited overlap: K=2 left/right halves, only N_o in {256, 512, 1024, 2048} aligned
    labelled samples (one-shot VFL 78.2 / 81.1 / 85.3 / 86.1 %; few-shot VFL + fine-tuning 80.4 / 84.1 / 86.4 / 87.5 %).
Every piece is upsampled (bilinear, antialiased) by 7x to the encoder's 224 px grid, keeping its aspect ratio
(16x32 -> 112x224); the frozen features of each piece are uploaded once and the label owner trains the usual head.
ModelNet40 multi-view (12 rendered views per object; 1 view per party, or 3 views per party averaged locally) if
the rendered images are available (MVCNN 'modelnet40_images_new_12x' layout).
Outputs: classic.json; he_ds_*.npz (party features for stage 13).
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from pe_common import DEV, evaluate_all, fmt, save_json, split_val, summarize, train_head

ROOT = os.environ.get("PE_CLASSIC", "/content/classic")
OUT = os.environ.get("PE_OUT", "/content/results")
ENCODERS = ["dinov2_g", "clip_l14"]
PIECES = {  # name: (rows, cols) slices of the 32x32 image, output size (H, W)
    "full": ((0, 32), (0, 32), (224, 224)),
    "top": ((0, 16), (0, 32), (112, 224)), "bottom": ((16, 32), (0, 32), (112, 224)),
    "left": ((0, 32), (0, 16), (224, 112)), "right": ((0, 32), (16, 32), (224, 112)),
    "q0": ((0, 16), (0, 16), (224, 224)), "q1": ((0, 16), (16, 32), (224, 224)),
    "q2": ((16, 32), (0, 16), (224, 224)), "q3": ((16, 32), (16, 32), (224, 224)),
}


def cifar(name):
    import torchvision

    path = f"{ROOT}/{name}_images.npz"
    if not os.path.exists(path):
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        tr, te = cls(f"{ROOT}/raw", train=True, download=True), cls(f"{ROOT}/raw", train=False, download=True)
        X = np.concatenate([tr.data, te.data]).transpose(0, 3, 1, 2)  # [60000,3,32,32] uint8
        y = np.concatenate([tr.targets, te.targets]).astype(np.int64)
        np.savez(path, X=X, y=y, n_train=len(tr.data))
    z = np.load(path)
    return z["X"], z["y"], int(z["n_train"])


def features_cifar(name, encoders, pieces):
    from pe_encoders import encoder, extract

    X, _, _ = cifar(name)
    for enc in encoders:
        todo = [p for p in pieces if not os.path.exists(f"{ROOT}/f_{name}_{enc}_{p}.npy")]
        if not todo:
            continue
        fn, _ = encoder(enc)
        for p in todo:
            (r0, r1), (c0, c1), size = PIECES[p]
            t = time.time()
            a = extract(fn, X[:, :, r0:r1, c0:c1], size, bs=256)
            np.save(f"{ROOT}/f_{name}_{enc}_{p}.npy", a)
            print(f"{name} {enc} {p} {a.shape} {time.time() - t:.0f}s", flush=True)
        del fn
        torch.cuda.empty_cache()


def heads_cifar(name, encoders, res, seeds=(0, 1, 2)):
    _, y, ntr = cifar(name)
    C = int(y.max()) + 1
    yt = torch.as_tensor(y, device=DEV)
    te = np.arange(ntr, len(y))
    configs = {"centralised (full image)": ["full"], "K=2 top/bottom": ["top", "bottom"], "solo bottom half": ["bottom"],
               "K=2 left/right": ["left", "right"], "K=4 quadrants": ["q0", "q1", "q2", "q3"], "solo quadrant": ["q3"]}
    for enc in encoders:
        if not os.path.exists(f"{ROOT}/f_{name}_{enc}_q3.npy"):
            continue
        F = {p: torch.as_tensor(np.load(f"{ROOT}/f_{name}_{enc}_{p}.npy"), device=DEV).float() for p in PIECES if os.path.exists(f"{ROOT}/f_{name}_{enc}_{p}.npy")}
        out = res.setdefault(name, {}).setdefault(enc, {})
        for cname, parts in configs.items():
            if not all(p in F for p in parts):
                continue
            X = torch.stack([F[p] for p in parts], 1)
            for arch in ("mlp2", "linear"):
                key = f"{cname}|{arch}"
                if key in out:
                    continue
                runs = []
                for s in seeds:
                    tr, va = split_val(ntr, s, 0.1, y[:ntr])
                    m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch=arch, seed=s)
                    runs.append(evaluate_all(m, X[te], yt[te]))
                out[key] = summarize(runs)
                print(f"{name} {enc:9s} {key:34s} {fmt(out[key])}", flush=True)
            save_json("classic.json", res)
        # Sun et al.: limited aligned samples, left/right halves
        if name == "cifar10" and "left" in F:
            X = torch.stack([F["left"], F["right"]], 1)
            for n_o in (256, 512, 1024, 2048):
                key = f"aligned {n_o} left/right|mlp2"
                if key in out:
                    continue
                runs = []
                for s in range(5):
                    rng = np.random.default_rng(s)
                    al = rng.choice(ntr, n_o, replace=False)
                    a, v = split_val(n_o, s, 0.15, None)
                    m = train_head(X[al[a]], yt[al[a]], X[al[v]], yt[al[v]], C, arch="mlp2", seed=s)
                    runs.append(evaluate_all(m, X[te], yt[te]))
                out[key] = summarize(runs)
                print(f"{name} {enc:9s} {key:34s} {fmt(out[key])}", flush=True)
            save_json("classic.json", res)
        if enc == encoders[0]:  # party features for the encrypted-head stage (top/bottom, 2 parties)
            split = np.zeros(len(y), int)
            split[te] = 2
            os.makedirs(os.environ.get("PE_HE_DS", "/content/he_ds"), exist_ok=True); np.savez(f"{os.environ.get('PE_HE_DS', '/content/he_ds')}/he_ds_{name}_{enc}_topbottom.npz", n_parties=2, p0=F["top"].cpu().numpy().astype(np.float16),
                     p1=F["bottom"].cpu().numpy().astype(np.float16), y=y, split=split, metric="acc")


# ------------------------------------------------------------------ ModelNet40, 12 views
def modelnet_index(root):
    """Rendered 12-view ModelNet40, either the MVCNN layout <class>/<train|test>/<object>_<v>.png or the HF mirror
    layout .../<class>/<train|test>/<object>.off/<object>.<v>.png.  Returns dict split -> [(label, [12 paths])]."""
    files = glob.glob(f"{root}/**/*.png", recursive=True)
    groups = {}
    for f in files:
        parts = f.split(os.sep)
        k = max(i for i, p in enumerate(parts) if p in ("train", "test"))
        split, cls = parts[k], parts[k - 1]
        obj = parts[k + 1] if len(parts) > k + 2 else re.split(r"[._]v?\d+\.png$", parts[-1])[0]
        groups.setdefault((split, cls, obj), []).append(f)
    classes = sorted({c for _, c, _ in groups})
    view = lambda f: int(re.findall(r"(\d+)\.png$", f)[0])  # noqa: E731
    out = {"train": [], "test": []}
    for (split, cls, obj), fs in sorted(groups.items()):
        if len(fs) == 12:
            out[split].append((classes.index(cls), sorted(fs, key=view)))
    return out, classes


def features_modelnet(root, encoders):
    from PIL import Image

    from pe_encoders import encoder, extract

    idx, classes = modelnet_index(root)
    items = idx["train"] + idx["test"]
    y = np.array([c for c, _ in items])
    np.savez(f"{ROOT}/modelnet_meta.npz", y=y, n_train=len(idx["train"]), classes=np.array(classes))
    imgs_path = f"{ROOT}/modelnet_imgs.npy"
    if not os.path.exists(imgs_path):
        from multiprocessing import Pool

        with Pool(min(16, os.cpu_count() or 4)) as pool:
            arr = pool.map(_load_views, [fs for _, fs in items], chunksize=16)
        np.save(imgs_path, np.stack(arr))  # [N,12,224,224] grey-level renderings
    A = np.load(imgs_path, mmap_mode="r")
    for enc in encoders:
        path = f"{ROOT}/f_modelnet40_{enc}.npy"
        if os.path.exists(path):
            continue
        fn, d = encoder(enc)
        out = np.zeros((len(A), 12, d), np.float16)
        t = time.time()
        for v in range(12):
            out[:, v] = extract(fn, A[:, v, None], 224, bs=128)  # grey -> 3 channels inside extract
        np.save(path, out)
        print("modelnet40", enc, out.shape, f"{time.time() - t:.0f}s", flush=True)


def _load_views(fs):
    from PIL import Image

    return np.stack([np.asarray(Image.open(f).convert("L").resize((224, 224), Image.BILINEAR)) for f in fs])


def heads_modelnet(encoders, res, seeds=(0, 1, 2)):
    if not os.path.exists(f"{ROOT}/modelnet_meta.npz"):
        return
    z = np.load(f"{ROOT}/modelnet_meta.npz")
    y, ntr = z["y"], int(z["n_train"])
    C = int(y.max()) + 1
    yt = torch.as_tensor(y, device=DEV)
    te = np.arange(ntr, len(y))
    for enc in encoders:
        if not os.path.exists(f"{ROOT}/f_modelnet40_{enc}.npy"):
            continue
        V = torch.as_tensor(np.load(f"{ROOT}/f_modelnet40_{enc}.npy"), device=DEV).float()  # [N,12,d]
        configs = {"K=12 (1 view/party)": V, "K=4 (3 views/party, local mean)": V.view(len(V), 4, 3, -1).mean(2),
                   "solo 1 view": V[:, :1], "centralised mean of 12 views": V.mean(1, keepdim=True)}
        out = res.setdefault("modelnet40", {}).setdefault(enc, {})
        for cname, X in configs.items():
            for arch in ("mlp2", "linear"):
                key = f"{cname}|{arch}"
                if key in out:
                    continue
                runs = []
                for s in seeds:
                    tr, va = split_val(ntr, s, 0.1, y[:ntr])
                    m = train_head(X[tr], yt[tr], X[va], yt[va], C, arch=arch, seed=s)
                    runs.append(evaluate_all(m, X[te], yt[te]))
                out[key] = summarize(runs)
                print(f"modelnet40 {enc:9s} {key:38s} {fmt(out[key])}", flush=True)
        save_json("classic.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cifar10,cifar100,modelnet40")
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--modelnet-root", default=os.environ.get("PE_MODELNET", "/content/modelnet"))
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    enc = a.encoders.split(",")
    path = os.path.join(OUT, "classic.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for ds in a.datasets.split(","):
        if ds.startswith("cifar"):
            pieces = list(PIECES) if ds == "cifar10" else ["full", "top", "bottom", "q0", "q1", "q2", "q3"]
            features_cifar(ds, enc, pieces)
            heads_cifar(ds, enc, res)
        elif ds == "modelnet40" and glob.glob(f"{a.modelnet_root}/**/*.png", recursive=True):
            features_modelnet(a.modelnet_root, enc)
            heads_modelnet(enc, res)
    save_json("classic.json", res)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
