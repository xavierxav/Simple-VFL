"""Stage 20 - inverting the uploads with a public, generic decoder, and party-side transforms against it.

The frozen encoders are public, so a curious label owner needs no in-domain image/feature pairs: a decoder trained by
others on web images can be applied to the uploads.  For OpenAI CLIP ViT-L/14 such a decoder is public: Stable unCLIP
2.1 "small" (stabilityai/stable-diffusion-2-1-unclip-small) generates an image from the projected CLIP image embedding,
and the projection (visual_projection) is part of the public encoder.  We know of no such decoder for the pooled DINOv2
token.

  python s20_generic_inversion.py vb        E1, VertiBench: the CLIP-L uploads of the paper (band-standardised input)
                                            and the uploads a party would send with CLIP's own preprocessing
  python s20_generic_inversion.py defences  E3, VertiBench CLIP-L: party-side transforms of the upload (secret rotation,
                                            private PCA-64, autoencoder to 100 dimensions, noise as targets in 100
                                            dimensions, one-layer split learning) against an attacker who knows k
                                            (image, upload) pairs of the party, or none (spectral alignment with public
                                            in-domain features), then the public decoder
  python s20_generic_inversion.py milk      E1, MILK10k: clinical and dermoscopic CLIP-L uploads
Metrics: an independent encoder (DINOv2-B) re-identifies the source of each reconstruction among the test images
(top-1/top-10; chance 1/N), a classifier trained on the DINOv2-B features of real images predicts the label from the
reconstruction, PSNR/SSIM on 64x64 thumbnails; for E3 also the cosine between the recovered and the true CLIP
embedding and the re-identification of the source in CLIP space by an attacker who holds candidate images.
Outputs: generic_inversion.json, generic_inversion_examples.npz.  PE_STUB=1: random decoder/judge (smoke test).
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pe_common import DEV, OUT, seed_all, split_val, train_head, predict_proba

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
STUB = bool(os.environ.get("PE_STUB"))
N_VB = int(os.environ.get("PE_GI_NVB", "160"))
N_MILK = int(os.environ.get("PE_GI_NMILK", "120"))
N_DEF = int(os.environ.get("PE_GI_NDEF", "32"))
STEPS = int(os.environ.get("PE_GI_STEPS", "25"))
KS = [int(k) for k in os.environ.get("PE_GI_KS", "10,100,1000,3142").split(",")]
PATH = os.path.join(OUT, "generic_inversion.json")
NPZ = os.path.join(OUT, "generic_inversion_examples.npz")
CLIPN = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
NEX = 8  # examples kept per configuration (128x128)


def load_res():
    return json.load(open(PATH)) if os.path.exists(PATH) else {}


def save(res, ex=None):
    with open(PATH, "w") as f:
        json.dump(res, f, indent=1)
    if ex is not None:
        old = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
        old.update(ex)
        np.savez_compressed(NPZ, **old)
    print("saved", PATH, flush=True)


def resize_u8(imgs, size):
    x = torch.as_tensor(np.ascontiguousarray(imgs)).float()
    return F.interpolate(x, size=(size, size), mode="area" if x.shape[-1] > size else "bilinear").clamp(0, 255).round().byte().numpy()


def psnr(x, y):
    return 10 * torch.log10(1.0 / F.mse_loss(x, y, reduction="none").mean((1, 2, 3)).clamp_min(1e-10))


def ssim(x, y, win=11, sigma=1.5):  # same as stage 4
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    k = (g @ g.T)[None, None].repeat(3, 1, 1, 1)
    f = lambda t: F.conv2d(t, k, groups=3)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def pix(rec, orig):
    """PSNR/SSIM on 64x64 thumbnails (uint8 [N,3,H,W] each)."""
    a, b = (torch.as_tensor(resize_u8(t, 64)).float() / 255 for t in (rec, orig))
    return {"psnr": float(psnr(a, b).mean()), "ssim": float(ssim(a, b).mean())}


def stretch(x):  # per image and channel, 2nd-98th percentile -> [0,1] (dark Sentinel-2 L1C composites)
    f = x.flatten(2)
    lo, hi = f.quantile(0.02, dim=2)[..., None, None], f.quantile(0.98, dim=2)[..., None, None]
    return ((x - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1)


class Decoder:
    """The public decoder and the public CLIP-L encoder that comes with it."""

    def __init__(self):
        if STUB:
            torch.manual_seed(0)
            self.W = nn.Linear(1024, 768, bias=False).to(DEV)
            return
        try:
            from diffusers import StableUnCLIPImg2ImgPipeline
        except Exception as e:  # diffusers missing or incompatible with the installed transformers
            print("!! diffusers import failed, installing a pinned version:", repr(e), flush=True)
            import subprocess

            subprocess.run(f"{sys.executable} -m pip install -q 'diffusers==0.31.0'", shell=True)
            from diffusers import StableUnCLIPImg2ImgPipeline
        for repo in ("stabilityai/stable-diffusion-2-1-unclip-small", "sd2-community/stable-diffusion-2-1-unclip-small"):  # original, mirror
            try:
                self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(repo, torch_dtype=torch.float16).to(DEV)
                self.repo = repo
                break
            except Exception as e:
                print("!! unavailable", repo, repr(e)[:300], flush=True)
        print("decoder", self.repo, flush=True)
        self.pipe.set_progress_bar_config(disable=True)
        self.enc = self.pipe.image_encoder.eval()  # CLIPVisionModelWithProjection (OpenAI ViT-L/14)
        self.W = self.enc.visual_projection

    @torch.no_grad()
    def project(self, pooled):
        """Pooled CLIP-L output (what a party uploads) -> CLIP image embedding, with the public projection."""
        p = torch.as_tensor(np.asarray(pooled, dtype=np.float32)).to(DEV)
        return torch.cat([self.W(p[i : i + 1024].to(self.W.weight.dtype)).float() for i in range(0, len(p), 1024)])

    @torch.no_grad()
    def pooled(self, pixel_values):
        if STUB:
            return torch.randn(len(pixel_values), 1024, device=DEV)
        return self.enc.vision_model(pixel_values=pixel_values.to(DEV, torch.float16)).pooler_output.float()

    @torch.no_grad()
    def decode(self, emb, seed=0, bs=4):
        """CLIP image embeddings [B,768] -> uint8 images [B,3,256,256]."""
        if STUB:
            return (torch.rand(len(emb), 3, 96, 96) * 255).byte().numpy()
        out = []
        for i in range(0, len(emb), bs):
            e = emb[i : i + bs].to(DEV, torch.float16)
            g = torch.Generator(device=DEV).manual_seed(seed + i)
            im = self.pipe(prompt=[""] * len(e), image_embeds=e, num_inference_steps=STEPS, generator=g, output_type="np").images
            out.append(resize_u8((np.clip(im, 0, 1) * 255).round().astype(np.uint8).transpose(0, 3, 1, 2), 256))
        return np.concatenate(out)


class Judge:
    """Independent encoder (DINOv2-B) for re-identification and label inference."""

    def __init__(self):
        if STUB:
            self.fn = None
            return
        from pe_encoders import encoder

        self.fn, _ = encoder("dinov2_b")

    @torch.no_grad()
    def feats(self, imgs, stretched=False):
        out = []
        for i in range(0, len(imgs), 64):
            x = torch.as_tensor(np.ascontiguousarray(imgs[i : i + 64])).to(DEV).float() / 255
            x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            if stretched:
                x = stretch(x)
            if self.fn is None:
                f = x.mean((2, 3)) @ torch.randn(3, 64, generator=torch.Generator().manual_seed(0)).to(DEV)
            else:
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
                    f = self.fn(x).float()
            out.append(F.normalize(f, dim=1).cpu())
        return torch.cat(out)


def reid(Q, G, true):
    """Rank of the true gallery item for every query (cosine); top-1/top-10 rates (%) and median rank."""
    s = F.normalize(torch.as_tensor(Q).float(), dim=1) @ F.normalize(torch.as_tensor(G).float(), dim=1).T
    own = s[torch.arange(len(s)), torch.as_tensor(true)]
    r = (s > own[:, None]).sum(1).float() + 1
    return {"top1": float((r <= 1).float().mean() * 100), "top10": float((r <= 10).float().mean() * 100), "median_rank": float(r.median()), "n_gallery": int(G.shape[0])}


def label_inference(Gtr, ytr, Q, yq, Greal=None):
    """Classifier trained on independent-encoder features of real images, applied to reconstructions (and to the real
    images for reference): accuracy and macro F1 (%)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(np.asarray(Gtr), ytr)
    out = {}
    for k, X in (("recon", Q), ("real", Greal)):
        if X is None:
            continue
        p = clf.predict(np.asarray(X))
        out[k] = {"acc": float((p == yq).mean() * 100), "f1": float(f1_score(yq, p, average="macro", labels=np.unique(ytr), zero_division=0) * 100)}
    return out


def mean_colour(imgs):
    m = np.asarray(imgs, dtype=np.float32).mean((2, 3), keepdims=True)
    return np.broadcast_to(m, np.asarray(imgs).shape).round().astype(np.uint8)


def thumbs16(imgs):  # stage-4 pixel re-identification space: 64x64 thumbnails average-pooled to 16x16
    return torch.cat([F.avg_pool2d(torch.as_tensor(resize_u8(imgs[i : i + 512], 64)).float() / 255, 4).flatten(1) for i in range(0, len(imgs), 512)])


def evaluate(recon, orig, G, true, judge, stretched, clf_data=None, T16=None):
    """recon/orig uint8 [n,3,h,w]; G: judge features of the whole gallery, true: gallery index of each source;
    T16: 16x16 thumbnails of the whole gallery for the pixel re-identification of stage 4 (MSE)."""
    out = pix(recon, orig)
    if T16 is not None:
        R16 = thumbs16(recon)
        d = torch.cdist(R16, T16)
        own = d[torch.arange(len(d)), torch.as_tensor(true)]
        r = (d < own[:, None]).sum(1).float() + 1
        out["reid_pix"] = {"top1": float((r <= 1).float().mean() * 100), "top10": float((r <= 10).float().mean() * 100), "median_rank": float(r.median())}
    if judge is None:
        return out
    R = judge.feats(recon, stretched)
    out.update({"reid": reid(R, G, true), "cos_to_source": float((R * G[torch.as_tensor(true)]).sum(1).mean())})
    if clf_data is not None:
        out["labels"] = label_inference(*clf_data[:2], R, clf_data[2], G[torch.as_tensor(true)])
    return out


# ------------------------------------------------------------------ E1, VertiBench
def vb_data():
    A = {s: np.load(f"{CACHE}/full13_{s}.npy", mmap_mode="r") for s in ("train", "test")}  # [N,16,13,158,158] uint8
    Fc = {s: np.load(f"{CACHE}/f7_clip_l14_224_{s}.npy").astype(np.float32) for s in ("train", "test")}  # [N,16,1024]
    y = {s: np.load(f"{CACHE}/labels_{s}.npy") for s in ("train", "test")}
    return A, Fc, y


def vb():
    from s7_vb_encoders import prep

    res = load_res()
    out = res.setdefault("vb", {})
    dec, judge = Decoder(), Judge()
    A, Fc, y = vb_data()
    p = 0  # party 0 (one revisit); every party gives the same picture
    rgb = {s: np.ascontiguousarray(A[s][:, p][:, [3, 2, 1]]) for s in ("train", "test")}  # uint8 [N,3,158,158]
    G = {s: judge.feats(rgb[s], stretched=True) for s in ("train", "test")}
    sel = np.linspace(0, len(y["test"]) - 1, N_VB).round().astype(int)
    out["n_test"], out["n_attacked"] = int(len(y["test"])), int(len(sel))
    # the attacker's projection of a stored upload is the public encoder's embedding of the same input
    x = torch.as_tensor(np.ascontiguousarray(A["test"][sel[:16], p])).to(DEV).float()
    out["check_cos_pooled"] = float(F.cosine_similarity(dec.pooled(prep(x, 224)), torch.as_tensor(Fc["test"][sel[:16], p]).to(DEV)).mean())
    Eall = dec.project(Fc["test"][:, p])
    out["linkage_clip"] = reid(Eall[sel].cpu(), Eall.cpu(), sel)  # attacker holding the candidate images: exact match
    xs = torch.as_tensor(rgb["test"][sel]).to(DEV).float() / 255
    xs = F.interpolate(xs, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    m, s_ = (torch.tensor(v, device=DEV)[None, :, None, None] for v in CLIPN)
    variants = {"paper upload": Eall[sel], "CLIP preprocessing": dec.W(dec.pooled((xs - m) / s_).to(dec.W.weight.dtype)).float()}
    ex = {"vb__original": resize_u8(rgb["test"][sel[:NEX]], 128)}
    clf = (G["train"], y["train"], y["test"][sel])
    mc = mean_colour(rgb["test"][sel])
    T16 = thumbs16(rgb["test"])
    out["mean_colour"] = evaluate(mc, rgb["test"][sel], None, sel, None, True, None, T16)
    out["chance_top1"] = 100 / len(y["test"])
    for name, E in variants.items():
        t0 = time.time()
        recon = dec.decode(E)
        out[name] = evaluate(recon, rgb["test"][sel], G["test"], sel, judge, True, clf, T16)
        out[name]["secs"] = round(time.time() - t0)
        ex[f"vb__{name.replace(' ', '_')}"] = resize_u8(recon[:NEX], 128)
        print("vb", name, json.dumps(out[name])[:400], flush=True)
        save(res, ex)
    out["real_label_inference"] = label_inference(G["train"], y["train"], G["test"][torch.as_tensor(sel)], y["test"][sel])
    save(res, ex)


# ------------------------------------------------------------------ E3, party-side transforms (VertiBench, CLIP-L)
def mlp(d_in, d_out, hidden=512, bn=False):
    return nn.Sequential(nn.Linear(d_in, hidden), *([nn.BatchNorm1d(hidden)] if bn else []), nn.ReLU(), nn.Linear(hidden, d_out))


def fit_ae(X, d=100, epochs=60, seed=0):
    seed_all(seed)
    enc, dec_ = mlp(X.shape[1], d).to(DEV), mlp(d, X.shape[1]).to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec_.parameters()], lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=DEV).split(256):
            loss = F.mse_loss(dec_(enc(X[idx])), X[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    enc.eval()
    return lambda Z: enc(Z).detach()


def fit_nat(X, d=100, epochs=60, seed=0):
    """Noise as targets (Bojanowski & Joulin 2017) on top of the frozen features, as FedOnce's parties do on images:
    each sample is assigned a fixed random point of the unit sphere, re-assigned within each batch (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    seed_all(seed)
    net = mlp(X.shape[1], d, bn=True).to(DEV)
    T = F.normalize(torch.randn(len(X), d, device=DEV), dim=1)
    assign = torch.arange(len(X), device=DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(X), device=DEV).split(256):
            if len(idx) < 2:
                continue
            o = F.normalize(net(X[idx]), dim=1)
            if ep % 3 == 0:  # re-assign the batch's targets among its samples
                with torch.no_grad():
                    cost = -(o @ T[assign[idx]].T).cpu().numpy()
                r, c = linear_sum_assignment(cost)
                assign[idx[torch.as_tensor(r, device=DEV)]] = assign[idx][torch.as_tensor(c, device=DEV)].clone()
            loss = F.mse_loss(o, T[assign[idx]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return lambda Z: F.normalize(net(Z), dim=1).detach()


def zs(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def split_one_layer(Ftr, ytr, Fte, yte, d=100, seed=0, epochs=50, patience=10):
    """One-layer split learning: each party trains Linear(1024 -> d) with the label owner's gradients, the label owner an
    MLP head on the concatenation.  Returns test metrics, epochs run, the trained party layers' outputs."""
    import copy

    seed_all(seed)
    K, C = Ftr.shape[1], int(ytr.max()) + 1
    Xtr, Xte = zs(Ftr, Fte)
    layers = nn.ModuleList([nn.Linear(Ftr.shape[2], d) for _ in range(K)]).to(DEV)
    head = nn.Sequential(nn.Linear(K * d, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.Linear(1000, C)).to(DEV)
    params = [*layers.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=4)
    tr, va = split_val(len(ytr), seed, 0.15, ytr.cpu().numpy())
    tr, va = torch.as_tensor(tr, device=DEV), torch.as_tensor(va, device=DEV)
    fwd = lambda X: head(torch.cat([layers[k](X[:, k]) for k in range(K)], 1))  # noqa: E731
    best, state, bad, ep_run = 1e9, None, 0, 0
    for ep in range(epochs):
        ep_run = ep + 1
        head.train()
        for idx in tr[torch.randperm(len(tr), device=DEV)].split(512):
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(fwd(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.cross_entropy(fwd(Xtr[va]), ytr[va]).item()
        sched.step(vl)
        if vl < best:
            best, state, bad = vl, (copy.deepcopy(layers.state_dict()), copy.deepcopy(head.state_dict())), 0
        else:
            bad += 1
            if bad >= patience:
                break
    layers.load_state_dict(state[0])
    head.load_state_dict(state[1])
    head.eval()
    with torch.no_grad():
        acc = float((fwd(Xte).argmax(1) == yte).float().mean() * 100)
        codes = (torch.stack([layers[k](Xtr[:, k]) for k in range(K)], 1), torch.stack([layers[k](Xte[:, k]) for k in range(K)], 1))
    return acc, ep_run, codes


def ridge_map(X, Y, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Closed-form ridge X -> Y (standardised X, relative lambda chosen by 5-fold CV); returns a predictor."""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6) if len(X) > 1 else torch.ones_like(X[0])
    Z = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=X.device)], 1).double()
    Yd = Y.double()

    def fit(Zf, Yf, lam):
        G = Zf.T @ Zf
        R = lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype) * G.diagonal().mean().clamp_min(1e-9)
        R[-1, -1] = 0
        return torch.linalg.solve(G + R, Zf.T @ Yf)

    n = len(Z)
    if n >= 10:
        f = torch.arange(n, device=Z.device) % 5
        errs = []
        for lam in lams:
            e = 0.0
            for i in range(5):
                W = fit(Z[f != i], Yd[f != i], lam)
                e += (1 - F.cosine_similarity(Z[f == i] @ W, Yd[f == i], dim=1)).sum().item()
            errs.append(e)
        lam = lams[int(np.argmin(errs))]
    else:
        lam = 1.0
    W = fit(Z, Yd, lam)
    return lambda Xn: (torch.cat([(Xn - mu) / sd, torch.ones(len(Xn), 1, device=Xn.device)], 1).double() @ W).float()


def mlp_map(X, Y, seed=0, epochs=300):
    seed_all(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    net = mlp(X.shape[1], Y.shape[1], hidden=1024).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Xs = (X - mu) / sd
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=DEV).split(256):
            loss = (1 - F.cosine_similarity(net(Xs[idx]), Y[idx], dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return lambda Xn: net((Xn - mu) / sd).detach()


def spectral_alignment(C, PUB, d):
    """No pairs: match the eigenvectors of the party's uploads (any linear map of its features, e.g. a secret rotation or
    its own PCA basis) to those of public in-domain features, in order of eigenvalue, signs by the skewness of the
    projections.  Returns the estimated features [N,1024]."""
    def eig(X, k):
        Xc = X - X.mean(0)
        val, vec = torch.linalg.eigh((Xc.T @ Xc / len(X)).double())
        o = torch.argsort(val, descending=True)[:k]
        return vec[:, o].float(), Xc

    Eu, Cc = eig(C, d)
    Ea, Ac = eig(PUB, d)
    sk = lambda Z: ((Z - Z.mean(0)) ** 3).mean(0)  # noqa: E731
    s = torch.sign(sk(Cc @ Eu)) * torch.sign(sk(Ac @ Ea))
    return (Cc @ Eu) * s @ Ea.T + PUB.mean(0)


def defences():
    res = load_res()
    out = res.setdefault("defences", {})
    dec, judge = Decoder(), Judge()
    A, Fc, y = vb_data()
    Ftr, Fte = (torch.as_tensor(Fc[s]).to(DEV) for s in ("train", "test"))
    ytr, yte = (torch.as_tensor(y[s]).to(DEV) for s in ("train", "test"))
    K, N = Ftr.shape[1], len(Ftr)
    Etr0, Ete0 = dec.project(Ftr[:, 0].cpu().numpy()), dec.project(Fte[:, 0].cpu().numpy())
    PUB = Ftr[:, 1:].reshape(-1, Ftr.shape[2])  # the attacker's public in-domain features: other revisits, public encoder
    codes, util = {}, {}
    # ---- party-side transforms, each fitted by the party on its own training features only
    for name in ("plain", "rotation", "PCA-64", "autoencoder-100", "noise-as-targets-100"):
        Ctr, Cte = [], []
        for k in range(K):
            g = torch.Generator().manual_seed(100 + k)
            if name == "plain":
                f = lambda Z: Z  # noqa: E731
            elif name == "rotation":
                Q, _ = torch.linalg.qr(torch.randn(Ftr.shape[2], Ftr.shape[2], generator=g))
                f = lambda Z, Q=Q.to(DEV): Z @ Q  # noqa: E731
            elif name == "PCA-64":
                mu = Ftr[:, k].mean(0)
                _, _, Vh = torch.linalg.svd(Ftr[:, k] - mu, full_matrices=False)
                f = lambda Z, mu=mu, V=Vh[:64]: (Z - mu) @ V.T  # noqa: E731
            else:
                mu, sd = Ftr[:, k].mean(0), Ftr[:, k].std(0).clamp_min(1e-6)
                fit = fit_ae if name.startswith("auto") else fit_nat
                g_ = fit((Ftr[:, k] - mu) / sd, seed=k)
                f = lambda Z, g_=g_, mu=mu, sd=sd: g_((Z - mu) / sd)  # noqa: E731
            with torch.no_grad():
                Ctr.append(f(Ftr[:, k]))
                Cte.append(f(Fte[:, k]))
        codes[name] = (torch.stack(Ctr, 1), torch.stack(Cte, 1))
        accs = []
        for s in range(3):  # utility: the usual MLP head on the 16 transformed uploads
            Xtr, Xte = zs(*codes[name])
            tr, va = split_val(N, s, 0.15, y["train"])
            m = train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], int(ytr.max()) + 1, arch="mlp2", seed=s)
            accs.append(float((predict_proba(m, Xte).argmax(1) == yte).float().mean() * 100))
        util[name] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)), "dim": int(codes[name][0].shape[2])}
        print("utility", name, util[name], flush=True)
    accs, eps = [], []
    for s in range(3):
        a, e, c = split_one_layer(Ftr, ytr, Fte, yte, seed=s)
        accs.append(a)
        eps.append(e)
        if s == 0:
            codes["one-layer split learning-100"] = c
    util["one-layer split learning-100"] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)), "dim": 100, "epochs": eps,
                                            "bytes_per_train_sample_per_party": float(np.mean(eps) * 100 * 2 * 2)}
    print("utility split", util["one-layer split learning-100"], flush=True)
    out["utility"] = util
    save(res)
    # ---- attacks on party 0
    att = out.setdefault("attacks", {})
    mu_e = Etr0.mean(0, keepdim=True)  # an attacker who outputs the mean training embedding
    out["cos_mean_embedding"] = float(F.cosine_similarity(mu_e.expand_as(Ete0), Ete0).mean())
    sel = np.linspace(0, len(y["test"]) - 1, N_VB).round().astype(int)[:: max(1, N_VB // N_DEF)][:N_DEF]
    recov = {}
    for name, (Ctr, Cte) in codes.items():
        a = att.setdefault(name, {})
        C0tr, C0te = Ctr[:, 0].float(), Cte[:, 0].float()
        if name in ("plain", "rotation", "PCA-64"):  # no pairs: spectral alignment of the linear transform
            Fh = spectral_alignment(torch.cat([C0tr, C0te]), PUB, C0tr.shape[1])[N:]
            Eh = dec.project(Fh.cpu().numpy())
            a["k=0"] = {"cos": float(F.cosine_similarity(Eh, Ete0).mean()), "linkage": reid(Eh.cpu(), Ete0.cpu(), np.arange(len(Ete0)))}
            recov[(name, 0)] = Eh
            print("attack", name, "k=0", a["k=0"], flush=True)
        for k in KS:
            k = min(k, N)
            idx = torch.as_tensor(np.random.default_rng(k).choice(N, k, replace=False), device=DEV)
            Eh = ridge_map(C0tr[idx], Etr0[idx])(C0te)
            kind = "ridge"
            if k >= 1000:  # a non-linear map when pairs are plentiful; keep the better on 10 % held-out pairs
                h = idx[: k // 10]
                rest = idx[k // 10 :]
                cand = {"ridge": ridge_map(C0tr[rest], Etr0[rest]), "mlp": mlp_map(C0tr[rest], Etr0[rest])}
                sc = {n_: float(F.cosine_similarity(fn(C0tr[h]), Etr0[h]).mean()) for n_, fn in cand.items()}
                if sc["mlp"] > sc["ridge"]:
                    Eh, kind = mlp_map(C0tr[idx], Etr0[idx])(C0te), "mlp"
            a[f"k={k}"] = {"cos": float(F.cosine_similarity(Eh, Ete0).mean()), "linkage": reid(Eh.cpu(), Ete0.cpu(), np.arange(len(Ete0))), "map": kind}
            recov[(name, k)] = Eh
            print("attack", name, f"k={k}", a[f"k={k}"], flush=True)
        save(res)
    # ---- decode a few configurations with the public decoder
    rgb = np.ascontiguousarray(A["test"][:, 0][:, [3, 2, 1]])
    Gte = judge.feats(rgb, stretched=True)
    Gtr = judge.feats(np.ascontiguousarray(A["train"][:, 0][:, [3, 2, 1]]), stretched=True)
    clf = (Gtr, y["train"], y["test"][sel])
    T16 = thumbs16(rgb)
    dc = out.setdefault("decoded", {"n": int(len(sel))})
    ex = {"def__original": resize_u8(rgb[sel[:NEX]], 128)}
    recov[("plain", "direct")] = Ete0  # undefended upload, public projection only
    todo = [("plain", "direct"), ("rotation", 0), ("PCA-64", 0), ("PCA-64", 1000), ("autoencoder-100", 100), ("autoencoder-100", 1000),
            ("noise-as-targets-100", 1000), ("one-layer split learning-100", 1000)]
    for name, k in todo:
        Eh = recov.get((name, k))
        if Eh is None:
            continue
        recon = dec.decode(Eh[torch.as_tensor(sel, device=DEV)])
        key = f"{name}|k={k}"
        dc[key] = evaluate(recon, rgb[sel], Gte, sel, judge, True, clf, T16)
        ex[f"def__{name.replace(' ', '_')}__k{k}"] = resize_u8(recon[:NEX], 128)
        print("decoded", key, json.dumps(dc[key])[:300], flush=True)
        save(res, ex)


# ------------------------------------------------------------------ E1, MILK10k
def milk():
    from s11_milk10k import ROOT, folds, lesions

    res = load_res()
    out = res.setdefault("milk", {})
    dec, judge = Decoder(), Judge()
    les = lesions()
    y = les["dx7"].to_numpy()
    te = folds(y)[0][2]
    sel = np.sort(np.random.default_rng(0).choice(te, min(N_MILK, len(te)), replace=False))
    trn = np.setdiff1d(np.arange(len(y)), te)
    m, s_ = (torch.tensor(v, device=DEV)[None, :, None, None] for v in CLIPN)
    ex = {}
    for kind in ("clin", "derm"):
        imgs = np.load(f"{ROOT}/img_{kind}.npy", mmap_mode="r")  # [N,3,224,224] uint8, as encoded in the paper
        G = judge.feats(imgs)
        x = torch.as_tensor(np.ascontiguousarray(imgs[sel])).to(DEV).float() / 255
        pooled = dec.pooled((x - m) / s_)
        o = out.setdefault(kind, {"n_attacked": int(len(sel)), "n_gallery": int(len(y))})
        if kind == "clin" and not STUB:  # the decoder's encoder is the paper's CLIP-L (openai/clip-vit-large-patch14)
            from pe_encoders import encoder

            fn, _ = encoder("clip_l14")
            with torch.no_grad():
                o["check_cos_pooled"] = float(F.cosine_similarity(pooled[:16], fn(x[:16]).float()).mean())
            del fn
            torch.cuda.empty_cache()
        E = dec.W(pooled.to(dec.W.weight.dtype)).float()
        clf = (G[torch.as_tensor(trn)], y[trn], y[sel])
        T16 = thumbs16(imgs)
        o["mean_colour"] = evaluate(mean_colour(imgs[sel]), imgs[sel], G, sel, judge, False, clf, T16)
        t0 = time.time()
        recon = dec.decode(E)
        o["upload"] = evaluate(recon, imgs[sel], G, sel, judge, False, clf, T16)
        o["upload"]["secs"] = round(time.time() - t0)
        o["real_label_inference"] = label_inference(G[torch.as_tensor(trn)], y[trn], G[torch.as_tensor(sel)], y[sel])
        ex[f"milk__{kind}__original"] = resize_u8(imgs[sel[:NEX]], 128)
        ex[f"milk__{kind}__upload"] = resize_u8(recon[:NEX], 128)
        print("milk", kind, json.dumps(o["upload"])[:400], flush=True)
        save(res, ex)


if __name__ == "__main__":
    {"vb": vb, "defences": defences, "milk": milk}[sys.argv[1]]()
