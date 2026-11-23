"""Stage 22 - VertiBench audit requested by the third review.

  preds     frozen MLP heads (CLIP-L, DINOv2-g, CLIP-B at 224 px), 5 seeds: test probabilities kept for paired tests
            and for scores by subset (test areas with / without a tile of the same site in the training set);
            the few-round "four logits per party" head (one MLP per party -> C logits, summed by the label owner)
  linkage   an attacker holding the candidate images: re-encode ALL test images (16 parties x 785) from the pixels,
            with the parties' preprocessing and with three others, and match the stored uploads against them
  nnpublic  index of the training location whose CLIP-L embedding (any revisit) is nearest to each test upload
  split     split learning with CLIP ViT-B/16 bottoms on the 16 parties (LoRA, and frozen with the same recipe:
            same model class, batches, augmentation, head and schedule), seeds from PE_SEEDS
  fingerprint  per-image statistics (band means + 4x4 B4 thumbnail) to map VertiBench samples to WorldStrat areas

Features: stage-7/12 cache (f7_*_224); images: full13_{train,test}.npy.  Outputs in $PE_OUT:
  vb_audit.json, vb_frozen_probs.npz, vb_split_clipb.json, vb_split_clipb_probs.npz, vb_fingerprints.npz
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from pe_common import DEV, evaluate_all, metrics_from_proba, predict_proba, save_json, seed_all, split_val, summarize, train_head
from s7_vb_encoders import MEAN, RGB, STD, prep

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
OUT = os.environ.get("PE_OUT", "/content/results")
SEEDS = [0, 1, 2, 3, 4]
NAMES = ["clip_l14_224", "dinov2_g_224", "clip_b16_224"]
if os.path.exists(f"{CACHE}/labels_train.npy"):  # absent on the WorldStrat VM, which only uses fingerprint()
    ytr = np.load(f"{CACHE}/labels_train.npy")
    yte = np.load(f"{CACHE}/labels_test.npy")
    C = int(ytr.max()) + 1


def load(path):
    p = os.path.join(OUT, path)
    return json.load(open(p)) if os.path.exists(p) else {}


def feats(name):
    return [torch.from_numpy(np.load(f"{CACHE}/f7_{name}_{s}.npy")).to(DEV).float() for s in ("train", "test")]


# ------------------------------------------------------------------ frozen heads with kept probabilities, logits head
def preds(res):
    yt_tr, yt_te = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yte, device=DEV)
    probs = {}
    for name in NAMES:
        if not os.path.exists(f"{CACHE}/f7_{name}_test.npy"):
            continue
        Xtr, Xte = feats(name)
        K, d = Xtr.shape[1], Xtr.shape[2]
        for arch in ("mlp2", f"logits{K}"):
            key = f"{name}|{arch}"
            if key in res.get("heads", {}):
                continue
            runs, P, ep = [], [], []
            for s in SEEDS:
                tr, va = split_val(len(ytr), s, 0.15, ytr)
                tr, va = torch.as_tensor(tr, device=DEV), torch.as_tensor(va, device=DEV)
                m = train_head(Xtr[tr], yt_tr[tr], Xtr[va], yt_tr[va], C, arch=arch, seed=s)
                runs.append(evaluate_all(m, Xte, yt_te))
                P.append(predict_proba(m, Xte).cpu().numpy())
                ep.append(int(getattr(m, "epochs_run", 0)))
            r = summarize(runs)
            n_tr, n_va = int(len(ytr) * 0.85), len(ytr) - int(len(ytr) * 0.85)
            if arch.startswith("logits"):
                # per epoch: C logits up + C gradients down per training sample and party, C logits per validation sample
                r["bytes_per_train_sample"] = float(np.mean([e * (2 * n_tr + n_va) * K * C * 2 / n_tr for e in ep]))
                r["epochs_run"] = ep
            r["params"] = int(sum(p.numel() for p in m.parameters()))
            res.setdefault("heads", {})[key] = r
            probs[f"{name}__{arch}"] = np.stack(P).astype(np.float32)
            print(key, f"acc {r['acc']:.2f}+-{r['acc_std']:.2f} f1 {r.get('f1', float('nan')):.2f} epochs {ep}", flush=True)
            save_json("vb_audit.json", res)
            old = dict(np.load(os.path.join(OUT, "vb_frozen_probs.npz"))) if os.path.exists(os.path.join(OUT, "vb_frozen_probs.npz")) else {}
            old.pop("y", None)
            old.update(probs)
            np.savez_compressed(os.path.join(OUT, "vb_frozen_probs.npz"), y=yte, **old)


# ------------------------------------------------------------------ linkage with a re-encoded gallery
def linkage(res):
    import s12_vb224 as s12

    if "linkage" in res:
        return
    m, f = s12.build("clip_l14_224")
    U = torch.from_numpy(np.load(f"{CACHE}/f7_clip_l14_224_test.npy")).float()  # stored uploads [785,16,1024]
    A = np.load(f"{CACHE}/full13_test.npy", mmap_mode="r")
    mR, sR = MEAN[RGB].view(1, 3, 1, 1).to(DEV), STD[RGB].view(1, 3, 1, 1).to(DEV)
    cm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
    cs = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)

    def enc(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return m(pixel_values=x).pooler_output.float()

    def same(x):  # exactly the extraction of the uploads (s12.extract)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return f(x).float()

    variants = {  # x: [B,13,158,158] float 0..255 on DEV
        "same preprocessing": lambda x: same(x),
        "bicubic resize": lambda x: enc(F.interpolate((x[:, RGB] - mR) / sR, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)),
        "8-bit resized image": lambda x: enc((F.interpolate(x[:, RGB], size=(224, 224), mode="bilinear", align_corners=False).round().clamp(0, 255) - mR) / sR),
        "CLIP preprocessing": lambda x: enc((F.interpolate(x[:, RGB] / 255, size=(224, 224), mode="bicubic", antialias=True, align_corners=False) - cm) / cs),
    }
    out = {"n_images": int(A.shape[0] * 16), "n_locations": int(A.shape[0])}
    Uf = F.normalize(U.flatten(0, 1), dim=1).to(DEV)
    dup = (Uf @ Uf.T > 1 - 1e-6).sum(1) - 1  # other stored uploads identical to this one
    out["duplicate_uploads"] = int((dup > 0).sum())
    del dup
    for vname, fn in variants.items():
        G = torch.zeros_like(U)
        for i in range(0, len(A), 32):
            xb = torch.from_numpy(np.ascontiguousarray(A[i : i + 32])).to(DEV).float()
            for p in range(16):
                G[i : i + len(xb), p] = fn(xb[:, p]).cpu()
        Gf = F.normalize(G.flatten(0, 1), dim=1).to(DEV)
        S = Uf @ Gf.T  # [12560, 12560]
        top2 = S.topk(2, dim=1)
        true = torch.arange(len(S), device=DEV)
        r_all = {"top1": float((top2.indices[:, 0] == true).float().mean() * 100),
                 "ties": int(((top2.values[:, 0] - top2.values[:, 1]) < 1e-5).sum()),
                 "cos_true_mean": float(S[true, true].mean()), "cos_true_min": float(S[true, true].min())}
        per = []
        for p in range(16):  # within one party: 785 candidates
            Sp = F.normalize(U[:, p], dim=1) @ F.normalize(G[:, p], dim=1).T
            per.append(float((Sp.argmax(1) == torch.arange(len(Sp))).float().mean() * 100))
        r_all["top1_within_party_mean"], r_all["top1_within_party_min"] = float(np.mean(per)), float(np.min(per))
        out[vname] = r_all
        print("linkage", vname, r_all, flush=True)
    res["linkage"] = out
    save_json("vb_audit.json", res)


# ------------------------------------------------------------------ nearest public location
def nnpublic(res):
    from transformers import CLIPVisionModelWithProjection

    if "nnpublic" in res:
        return
    W = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").visual_projection.to(DEV).float()
    with torch.no_grad():
        Etr = F.normalize(W(torch.from_numpy(np.load(f"{CACHE}/f7_clip_l14_224_train.npy")).to(DEV).float()), dim=-1)  # [3142,16,768]
        Ete = F.normalize(W(torch.from_numpy(np.load(f"{CACHE}/f7_clip_l14_224_test.npy")).to(DEV).float()), dim=-1)
        G = Etr.flatten(0, 1)
        loc, cos = np.zeros((len(Ete), 16), np.int16), np.zeros((len(Ete), 16), np.float16)
        for p in range(16):
            s = Ete[:, p] @ G.T
            v, j = s.max(1)
            loc[:, p], cos[:, p] = (j // 16).cpu().numpy(), v.cpu().numpy()
    np.savez_compressed(os.path.join(OUT, "vb_nnpublic.npz"), loc=loc, cos=cos)
    res["nnpublic"] = {"label_match_party0": float((ytr[loc[:, 0]] == yte).mean() * 100),
                       "label_match_all": float((ytr[loc] == yte[:, None]).mean() * 100)}
    print("nnpublic", res["nnpublic"], flush=True)
    save_json("vb_audit.json", res)


# ------------------------------------------------------------------ split learning with foundation bottoms
def split():
    from pe_split import train_split

    kinds = os.environ.get("PE_SPLIT_KINDS", "clip_b16_frozen,clip_b16_lora").split(",")
    seeds = [int(s) for s in os.environ.get("PE_SEEDS", "0,1,2").split(",")]
    epochs = int(os.environ.get("PE_SPLIT_EPOCHS", "10"))
    res = load("vb_split_clipb.json")
    pp = os.path.join(OUT, "vb_split_clipb_probs.npz")
    probs = dict(np.load(pp)) if os.path.exists(pp) else {}
    probs.pop("y", None)
    Atr = np.load(f"{CACHE}/full13_train.npy", mmap_mode="r")
    Ate = np.load(f"{CACHE}/full13_test.npy", mmap_mode="r")
    t0 = time.time()
    parties = [np.ascontiguousarray(np.concatenate([Atr[:, p][:, RGB], Ate[:, p][:, RGB]])) for p in range(16)]  # uint8 [N,3,158,158]
    print(f"party arrays ready {time.time() - t0:.0f}s", flush=True)
    y = np.concatenate([ytr, yte])
    te = np.arange(len(ytr), len(y))
    # the parties' own preprocessing (per-band standardisation of the raw composites), as for the frozen uploads
    stats = ((MEAN[RGB] / 255).tolist(), (STD[RGB] / 255).tolist())
    for kind in kinds:
        out = res.setdefault(kind, {"runs": [], "infos": []})
        for s in seeds:
            if f"{kind}__seed{s}" in probs:
                continue
            tr, va = split_val(len(ytr), s, 0.15, ytr)
            m, info = train_split(parties, y, tr, va, te, [kind] * 16, C, epochs=epochs, bs=int(os.environ.get("PE_SPLIT_BS", "32")),
                                  size=(224, 224), seed=s, log=lambda x: print(kind, s, x, flush=True), stats=stats)
            probs[f"{kind}__seed{s}"] = info.pop("test_proba").astype(np.float32)
            out["runs"].append({k: float(v) for k, v in m.items()})
            out["infos"].append({k: v for k, v in info.items()})
            out["summary"] = summarize(out["runs"])
            print(kind, "seed", s, {k: round(float(v) * 100, 2) for k, v in m.items()}, f"{info['secs']:.0f}s", flush=True)
            save_json("vb_split_clipb.json", res)
            np.savez_compressed(pp, y=yte, **probs)


# ------------------------------------------------------------------ image fingerprints (VertiBench <-> WorldStrat)
def fingerprint(A):
    """A: uint8 [N,16,13,158,158] -> float16 [N,16,29]: 13 band means and a 4x4 thumbnail of B4 (red)."""
    out = np.zeros((len(A), A.shape[1], 29), np.float16)
    for i in range(0, len(A), 64):
        x = torch.from_numpy(np.ascontiguousarray(A[i : i + 64])).float()
        out[i : i + len(x), :, :13] = x.mean((-1, -2)).numpy()
        out[i : i + len(x), :, 13:] = F.adaptive_avg_pool2d(x[:, :, 3], 4).flatten(2).numpy()
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    seed_all(0)
    res = load("vb_audit.json")
    if stage in ("fingerprint", "all"):
        fp = {s: fingerprint(np.load(f"{CACHE}/full13_{s}.npy", mmap_mode="r")) for s in ("train", "test")}
        np.savez_compressed(os.path.join(OUT, "vb_fingerprints.npz"), y_train=ytr, y_test=yte, **fp)
    for name, fn in (("preds", preds), ("linkage", linkage), ("nnpublic", nnpublic)):
        if stage in (name, "all"):
            try:
                fn(res)
            except Exception:
                import traceback

                traceback.print_exc()
    if stage == "split":
        split()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
