"""Stage 12 - VertiBench at the encoders' native 224 px: encoder table, compute cost (FLOPs, time) per party.

Extracts (from the packed raw arrays of stage 3, full13_{split}.npy) the pooled features of a range of frozen encoders
at 224 px (same per-band standardisation of the RGB bands as stage 7), trains the one-shot heads on the official split
(5 seeds, MLP and linear), and measures for every encoder and for the VertiBench SplitNN bottom (ResNet-18 with a
9x9 stride-2 13-band input convolution, 158 px) the forward FLOPs per image.  Output: vb224_encoders.json and the
cached features f7_{name}_{split}.npy used by stages 9, 10, 4 and 13.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pe_common import DEV, evaluate_all, fmt, save_json, split_val, summarize, train_head
from s7_vb_encoders import prep

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
NAMES = ["dinov2_g_224", "clip_l14_224", "dinov2_l_224", "dinov2_b_224", "dinov2_s_224", "clip_b16_224", "siglip_b16_224",
         "resnet50_in1k_224", "dinov2_b_random_224"]


def build(name):
    from transformers import AutoConfig, AutoModel, CLIPVisionModel, SiglipVisionModel

    if name.startswith("dinov2") and "random" not in name:
        m = AutoModel.from_pretrained(f"facebook/dinov2-{dict(s='small', b='base', l='large', g='giant')[name.split('_')[1]]}")
        f = lambda x: m(pixel_values=prep(x, 224)).pooler_output  # noqa: E731
    elif name == "dinov2_b_random_224":
        torch.manual_seed(0)
        m = AutoModel.from_config(AutoConfig.from_pretrained("facebook/dinov2-base"))
        f = lambda x: m(pixel_values=prep(x, 224)).pooler_output  # noqa: E731
    elif name.startswith("clip"):
        m = CLIPVisionModel.from_pretrained({"clip_l14_224": "openai/clip-vit-large-patch14", "clip_b16_224": "openai/clip-vit-base-patch16"}[name])
        f = lambda x: m(pixel_values=prep(x, 224)).pooler_output  # noqa: E731
    elif name == "siglip_b16_224":
        m = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        f = lambda x: m(pixel_values=prep(x, 224)).pooler_output  # noqa: E731
    elif name == "resnet50_in1k_224":
        import torchvision

        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()
        f = lambda x: m(prep(x, 224))  # noqa: E731
    else:
        raise ValueError(name)
    m.to(DEV).eval()
    return m, f


def flops_per_image(f, shape=(1, 13, 158, 158)):
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.rand(shape, device=DEV) * 255
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        f(x)
    return int(fc.get_total_flops())


def vertibench_bottom_flops():
    import torchvision

    r = torchvision.models.resnet18(weights=None)
    r.conv1 = nn.Conv2d(13, 64, 9, stride=2, padding=3, bias=False)
    r.fc = nn.Identity()
    r.to(DEV).eval()
    return flops_per_image(lambda x: r(x / 255))


def extract(names, chunk=128):
    res = {}
    for name in names:
        if all(os.path.exists(f"{CACHE}/f7_{name}_{s}.npy") for s in ("train", "test")):
            continue
        try:
            m, f = build(name)
        except Exception as e:
            print("!! unavailable", name, repr(e), flush=True)
            continue
        t0 = time.time()
        for split in ("train", "test"):
            A = np.load(f"{CACHE}/full13_{split}.npy", mmap_mode="r")
            out = None
            for i in range(0, len(A), chunk):
                xb = torch.from_numpy(np.ascontiguousarray(A[i : i + chunk])).to(DEV)
                for p in range(16):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        z = f(xb[:, p].float()).float().cpu().numpy().astype(np.float16)
                    if out is None:
                        out = np.zeros((len(A), 16, z.shape[1]), np.float16)
                    out[i : i + len(z), p] = z
            assert np.isfinite(out.astype(np.float32)).all(), name
            np.save(f"{CACHE}/f7_{name}_{split}.npy", out)
        torch.cuda.synchronize()
        res[name] = {"extract_secs_all_images": time.time() - t0, "params": int(sum(p.numel() for p in m.parameters())),
                     "flops_per_image": flops_per_image(f)}
        print(f"extracted {name} in {res[name]['extract_secs_all_images']:.0f}s, {res[name]['flops_per_image'] / 1e9:.1f} GFLOPs/image", flush=True)
        del m, f
        torch.cuda.empty_cache()
    return res


def heads(names):
    ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    ncls = int(ytr.max()) + 1
    out = {}
    for name in names:
        if not os.path.exists(f"{CACHE}/f7_{name}_test.npy"):
            continue
        Xtr, Xte = [torch.from_numpy(np.load(f"{CACHE}/f7_{name}_{s}.npy")).to(DEV).float() for s in ("train", "test")]
        out[name] = {"dim": int(Xtr.shape[-1])}
        for arch in ("mlp2", "linear"):
            runs = []
            for s in range(5):
                tr, va = split_val(len(ytr), s, 0.15, ytr.cpu().numpy())
                m = train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], ncls, arch=arch, seed=s)
                runs.append(evaluate_all(m, Xte, yte))
            out[name][arch] = summarize(runs)
            print(f"{name:20s} {arch:6s} {fmt(out[name][arch])}", flush=True)
    return out


def main():
    import json

    path = os.path.join(os.environ.get("PE_OUT", "/content/results"), "vb224_encoders.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    cost = extract(NAMES)
    res.setdefault("cost", {}).update(cost)
    res["cost"]["vertibench_resnet18_bottom_158px"] = {"flops_per_image": vertibench_bottom_flops(), "params": 11_250_000}
    save_json("vb224_encoders.json", res)
    res["heads"] = heads(NAMES + ["ssl4eo_s2_vits16"])
    save_json("vb224_encoders.json", res)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
