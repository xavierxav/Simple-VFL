"""Stage 1 - download VertiBench raw arrays and cache frozen features for several backbones.

Run on a Colab GPU:  python s1_vertibench_features.py [--backbones a,b,c]
Outputs (in $PE_CACHE, default /content/cache/vb):
  feat_{backbone}_{split}.npy   float16 [N, 16, d]   (one row per sample, one slot per party)
  labels_{split}.npy            int64   [N]
  thumb64_{split}.npy           uint8   [N, 16, 3, 64, 64]   RGB thumbnails (inversion attack)
  thumb13_{split}.npy           uint8   [N, 16, 13, 80, 80]  all bands (SSL / SplitNN baselines)
Resumable: finished backbones are skipped.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
WORK = os.environ.get("PE_WORK", "/content/vb_raw")
ARCHIVE = os.path.join(WORK, "Vertibench.7z")
URL = "https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z"
EXPECTED = 8_222_228_129
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same per-band statistics as the original extraction scripts (computed on train+test, unsupervised).
MEAN = torch.tensor(
    [81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384]
)
STD = torch.tensor(
    [37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326]
)
RGB = [3, 2, 1]
# Four 3-band composites covering all 13 bands (B1..B12 -> idx 0..12; B10 cirrus is almost empty).
GROUPS = [[3, 2, 1], [7, 3, 2], [11, 7, 3], [12, 11, 7]]


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def ensure_data():
    root = os.path.join(WORK, "data")
    if os.path.isdir(os.path.join(root, "Vertibench")) and len(glob.glob(os.path.join(root, "Vertibench", "satellite_party*.npy"))) == 62832:
        return os.path.join(root, "Vertibench")
    os.makedirs(WORK, exist_ok=True)
    if not os.path.exists(ARCHIVE) or os.path.getsize(ARCHIVE) != EXPECTED:
        for attempt in range(40):  # resumable: the HF CDN sometimes drops long streams
            subprocess.run(f"curl --http1.1 -L -C - -s -o {ARCHIVE} {URL}", shell=True)
            if os.path.exists(ARCHIVE) and os.path.getsize(ARCHIVE) == EXPECTED:
                break
            print("download attempt", attempt, "size", os.path.getsize(ARCHIVE) if os.path.exists(ARCHIVE) else 0, flush=True)
            time.sleep(5)
    assert os.path.getsize(ARCHIVE) == EXPECTED, os.path.getsize(ARCHIVE)
    os.makedirs(root, exist_ok=True)
    t = time.time()
    if shutil.which("7z") is None:
        subprocess.run("apt-get install -y -q p7zip-full > /dev/null 2>&1", shell=True)
    if shutil.which("7z"):
        sh(f"7z x -y -bso0 -bsp0 -o{root} {ARCHIVE}")
    else:
        sh(f"{sys.executable} -m pip install -q py7zr")
        import py7zr

        with py7zr.SevenZipFile(ARCHIVE) as z:
            z.extractall(path=root)
    print("extracted in", round(time.time() - t), "s", flush=True)
    n = len(glob.glob(os.path.join(root, "Vertibench", "satellite_party*.npy")))
    assert n == 62832, n
    return os.path.join(root, "Vertibench")


# --------------------------------------------------------------------------- backbones
def prep(x, chans, size):
    """x: [B,13,H,W] float on DEV -> normalised, resized [B,len(chans),size,size]."""
    m = MEAN[chans].to(x.device)[None, :, None, None]
    s = STD[chans].to(x.device)[None, :, None, None]
    return F.interpolate((x[:, chans] - m) / s, size=(size, size), mode="bilinear", align_corners=False)


_HF_CACHE = {}


def build(name):
    """Return fn(x13)->[B,d] features, plus output dim."""
    from transformers import AutoConfig, AutoModel

    def hf_model(path):
        if path not in _HF_CACHE:
            _HF_CACHE[path] = AutoModel.from_pretrained(path).to(DEV).eval()
        return _HF_CACHE[path]

    hf = {
        "dinov2_s": "facebook/dinov2-small",
        "dinov2_b": "facebook/dinov2-base",
        "dinov2_l": "facebook/dinov2-large",
        "dinov2_g": "facebook/dinov2-giant",
    }
    if name in hf:
        m = hf_model(hf[name])
        return (lambda x: m(prep(x, RGB, 168)).pooler_output), None
    if name == "dinov2_g_4comp":
        m = hf_model(hf["dinov2_g"])

        def fn(x):
            return torch.cat([m(prep(x, g, 168)).pooler_output for g in GROUPS], 1)

        return fn, None
    if name == "dinov2_b_random":
        torch.manual_seed(0)
        m = AutoModel.from_config(AutoConfig.from_pretrained(hf["dinov2_b"])).to(DEV).eval()
        return (lambda x: m(prep(x, RGB, 168)).pooler_output), None
    if name == "clip_l14":
        from transformers import CLIPVisionModel

        m = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
        return (lambda x: m(pixel_values=prep(x, RGB, 224)).pooler_output), None
    if name == "resnet50_in1k":
        import timm

        m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).to(DEV).eval()
        return (lambda x: m(prep(x, RGB, 224))), None
    if name == "ssl4eo_vits16":  # Sentinel-2 SSL backbone (13 bands), best effort
        from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224

        m = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO).to(DEV).eval()
        return (lambda x: m(prep(x, list(range(13)), 224))), None
    raise ValueError(name)


ALL = ["dinov2_g", "dinov2_s", "dinov2_b", "dinov2_l", "clip_l14", "resnet50_in1k", "dinov2_b_random", "dinov2_g_4comp", "ssl4eo_vits16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(ALL))
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    data = ensure_data()
    names = [b for b in a.backbones.split(",") if b]
    todo = [b for b in names if not all(os.path.exists(f"{CACHE}/feat_{b}_{s}.npy") for s in ("train", "test"))]
    need_thumbs = not os.path.exists(f"{CACHE}/thumb13_test.npy")
    print("backbones to extract:", todo, "thumbs:", need_thumbs, flush=True)

    fns = {}
    for b in todo:
        try:
            fns[b], _ = build(b)
            print("loaded", b, flush=True)
        except Exception as e:  # optional backbones must not kill the run
            print("!! could not load", b, repr(e), flush=True)
    for split in ("train", "test"):
        y = np.load(f"{data}/label_{split}.npy")
        np.save(f"{CACHE}/labels_{split}.npy", y.astype(np.int64))
        N = len(y)
        print(split, "N =", N, "labels", np.bincount(y), flush=True)
        feats = {b: None for b in fns}
        if need_thumbs:
            t64 = np.zeros((N, 16, 3, 64, 64), np.uint8)
            t13 = np.zeros((N, 16, 13, 80, 80), np.uint8)
        t0 = time.time()
        for party in range(16):
            for s in range(0, N, a.chunk):
                idx = list(range(s, min(N, s + a.chunk)))
                x = np.stack([np.load(f"{data}/satellite_party{party}_{split}_{i}.npy") for i in idx]).astype(np.float32)
                if party == 0 and s == 0:
                    print("raw array", x.shape, x.dtype, "min/max", x.min(), x.max(), flush=True)
                xt = torch.from_numpy(x).to(DEV)
                if need_thumbs:
                    t64[idx, party] = F.interpolate(xt[:, RGB], size=64, mode="area").clamp(0, 255).round().byte().cpu().numpy()
                    t13[idx, party] = F.interpolate(xt, size=80, mode="area").clamp(0, 255).round().byte().cpu().numpy()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                    for b, fn in fns.items():
                        f = fn(xt).float().cpu().numpy().astype(np.float16)
                        if feats[b] is None:
                            feats[b] = np.zeros((N, 16, f.shape[1]), np.float16)
                            print(" ", b, "dim", f.shape[1], flush=True)
                        feats[b][idx, party] = f
            print(f"{split} party {party} done, {time.time() - t0:.0f}s", flush=True)
        for b, arr in feats.items():
            assert np.isfinite(arr.astype(np.float32)).all(), f"non-finite features for {b}"
            np.save(f"{CACHE}/feat_{b}_{split}.npy", arr)
        if need_thumbs:
            np.save(f"{CACHE}/thumb64_{split}.npy", t64)
            np.save(f"{CACHE}/thumb13_{split}.npy", t13)
    print("DONE", os.listdir(CACHE), flush=True)


if __name__ == "__main__":
    main()
