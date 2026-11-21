"""Stage 7 - why does CLIP beat DINOv2 on VertiBench Satellite?  Encoder ablations.

(A) re-extract frozen features from the packed raw arrays of stage 3 (full13_{train,test}.npy) for controlled variants:
    input resolution (DINOv2-g at 168/224 px, CLIP-L at 224/168 px), pooling (DINOv2 [CLS; mean patch]),
    other image-text encoders (CLIP-B/16, SigLIP-B/16), small and random references.
(B) heads on the official split (5 seeds, MLP and linear), as in stage 2.
Also the shared input standardisation (prep) and feature extraction used by stages 9 and 12.
Outputs: vb_encoders.json.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from pe_common import DEV, evaluate, fmt, save_json, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
MEAN = torch.tensor([81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384])
STD = torch.tensor([37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326])
RGB = [3, 2, 1]


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


def main():
    ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    ncls = int(ytr.max()) + 1
    extract_all()
    names = [n for n in ("dinov2_g_168", "dinov2_g_224", "dinov2_g_168_clsmean", "clip_l14_224", "clip_l14_168", "clip_b16_224", "siglip_b16_224", "dinov2_s_168", "dinov2_b_random_168")
             if os.path.exists(f"{CACHE}/f7_{n}_test.npy")]
    save_json("vb_encoders.json", official_heads(names, ytr, yte, ncls))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
