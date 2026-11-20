"""Stage 4 - feature-inversion attack on the uploaded DINOv2-g class tokens.

An honest-but-curious coordinator (the label owner) holds auxiliary image/feature pairs from the same distribution
(here: the training pairs of every party, a strong assumption) and knows which party sent which vector. It trains a
decoder  upload -> 64x64 RGB thumbnail  and applies it to the uploads of held-out test samples.

Decoder: a learned per-party affine alignment (initialised at the identity) followed by a shared deconvolutional trunk.
The alignment matters for per-party PCA, whose bases differ between parties: a decoder blind to the sender would have
to decode 16 incompatible coordinate systems at once.
Attacks:  single = one upload -> the image of that party;  joint = the 16 uploads of a sample (+ target party) -> the
image of the target party (the coordinator receives all of them).
Upload formats: 16-bit float, scalar quantisation, per-party PCA (same compressor as stage 2), Gaussian noise
(rho x per-coordinate std, re-drawn at every training step), and the clipped isotropic Gaussian mechanism of a
per-upload (eps, delta)-DP release (clip to the median norm, sigma = 2c/mu, mu from eps at delta = 1e-5).
Reference points: mean training image; nearest training image in feature space.
Outputs: $PE_OUT/vb_inversion.json and vb_inversion_examples.npz (8 test locations, party 0).
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pe_common import DEV, OUT, save_json, seed_all

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
EPOCHS = int(os.environ.get("PE_INV_EPOCHS", "15"))
EX = [0, 97, 211, 333, 452, 571, 690, 777]  # test samples shown in the figure (party 0)

import s2_vertibench_heads as s2  # noqa: E402  (re-use the exact compressor of stage 2)


def load_feature(split, backbone="dinov2_g"):
    cands = {"dinov2_g": ("feat_dinov2_g", "f7_dinov2_g_168"), "clip_l14": ("feat_clip_l14", "f7_clip_l14_224")}[backbone]
    for f in (f"{CACHE}/{c}_{split}.npy" for c in cands):
        if os.path.exists(f):
            return torch.from_numpy(np.load(f)).to(DEV).float()
    raise FileNotFoundError(backbone)


def load_thumbs(split):
    path = f"{CACHE}/thumb64_{split}.npy"
    if not os.path.exists(path):  # derive from the packed full-resolution arrays of stage 3
        A = np.load(f"{CACHE}/full13_{split}.npy", mmap_mode="r")
        T = np.zeros((len(A), 16, 3, 64, 64), np.uint8)
        for i in range(0, len(A), 256):
            x = torch.from_numpy(np.ascontiguousarray(A[i : i + 256][:, :, [3, 2, 1]])).to(DEV).float().flatten(0, 1)
            T[i : i + 256] = F.interpolate(x, size=64, mode="area").clamp(0, 255).round().byte().view(-1, 16, 3, 64, 64).cpu().numpy()
        np.save(path, T)
    return torch.from_numpy(np.load(path)).to(DEV)


def gdp_mu(eps, delta=1e-5):
    """mu such that a mu-GDP mechanism is (eps, delta)-DP (inverse of the Dong et al. conversion, log space)."""
    from scipy.special import log_ndtr, ndtr

    def eps_of(mu):
        f = lambda e: ndtr(-e / mu + mu / 2) - np.exp(e + log_ndtr(-e / mu - mu / 2)) - delta  # noqa: E731
        lo, hi = 0.0, mu * mu + 20 * mu + 50
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if f(m) > 0 else (lo, m)
        return hi

    lo, hi = 1e-4, 1e3
    for _ in range(200):
        m = (lo * hi) ** 0.5
        lo, hi = (m, hi) if eps_of(m) < eps else (lo, m)
    return lo


class Decoder(nn.Module):
    def __init__(self, d, joint=False, n_parties=16):
        super().__init__()
        self.joint, self.d = joint, d
        self.A = nn.Parameter(torch.eye(d).repeat(n_parties, 1, 1))  # per-party alignment
        self.b = nn.Parameter(torch.zeros(n_parties, d))
        self.fc = nn.Sequential(nn.Linear(n_parties * d + n_parties if joint else d, 512 * 4 * 4), nn.ReLU())
        blocks, c = [], 512
        for c_out in (256, 128, 64, 32):  # 4 -> 8 -> 16 -> 32 -> 64
            blocks += [nn.ConvTranspose2d(c, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU()]
            c = c_out
        self.up = nn.Sequential(*blocks, nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, z, party):
        if self.joint:  # z [B,16,d]; party = target party [B]
            z = torch.einsum("bkd,kde->bke", z, self.A) + self.b
            z = torch.cat([z.flatten(1), F.one_hot(party, 16).float()], 1)
        else:  # z [B,d]; party = sender [B]
            out = torch.empty_like(z)
            for p in party.unique():
                m = party == p
                out[m] = z[m] @ self.A[p] + self.b[p]
            z = out
        return self.up(self.fc(z).view(-1, 512, 4, 4))


def ssim(x, y, win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    k = (g @ g.T)[None, None].repeat(3, 1, 1, 1)
    f = lambda t: F.conv2d(t, k, groups=3)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def reid(rec, tgt, chunk=1024):
    """Rank of the true test image among all test images (MSE on 16x16 thumbnails): top-1 / top-10 rate, median rank."""
    R = rec.float().flatten(1)  # inputs are 16x16 thumbnails (64x64 average-pooled by 4)
    T = tgt.float().flatten(1)
    tn = (T * T).sum(1)
    ranks = []
    for i in range(0, len(R), chunk):
        r = R[i : i + chunk]
        d = (r * r).sum(1, keepdim=True) - 2 * r @ T.T + tn[None]
        own = d[torch.arange(len(r)), torch.arange(i, i + len(r))]
        ranks.append((d < own[:, None]).sum(1) + 1)
    ranks = torch.cat(ranks).float()
    return {"top1": (ranks <= 1).float().mean().item() * 100, "top10": (ranks <= 10).float().mean().item() * 100, "median_rank": ranks.median().item(), "n": len(ranks)}


def psnr(x, y):
    return 10 * torch.log10(1.0 / F.mse_loss(x, y, reduction="none").mean((1, 2, 3)).clamp_min(1e-10))


def conditions(Ftr):
    """name -> (deterministic transform prep(Xtr, Xte), per-step training perturbation or None, test perturbation)."""
    def det(d, bits):
        comp = s2.make_compressor(d, bits)
        return lambda Xtr, Xte: comp(Xtr, Xtr[:1], Xte, 0)[0::2]

    sd = Ftr.std(0, keepdim=True)  # [1,16,D] per party/coordinate, training rows only

    def noise(r):  # X: [B,16,D] (p None) or [B,D] uploads of parties p
        return lambda X, p=None, g=None: X + r * (sd if p is None else sd[0, p]) * torch.randn(X.shape, device=DEV, generator=g)

    C = Ftr.flatten(0, 1).norm(dim=1).median().item()

    def dp(eps):
        sigma = 2 * C / gdp_mu(eps)
        clip = lambda X: X * (C / X.norm(dim=-1, keepdim=True).clamp_min(C))  # noqa: E731
        return (lambda Xtr, Xte: (clip(Xtr), clip(Xte))), (lambda X, p=None, g=None: X + sigma * torch.randn(X.shape, device=DEV, generator=g)), sigma

    ident = lambda Xtr, Xte: (Xtr, Xte)  # noqa: E731
    out = {
        "full, 16-bit": (det(1536, 16), None),
        "full, 8-bit": (det(1536, 8), None),
        "full, 4-bit": (det(1536, 4), None),
        "full, 1-bit": (det(1536, 1), None),
        "PCA-256": (det(256, 32), None),
        "PCA-64": (det(64, 32), None),
        "PCA-16": (det(16, 32), None),
        "PCA-64, 4-bit": (det(64, 4), None),
        "PCA-16, 4-bit": (det(16, 4), None),
        "noise rho=1": (ident, noise(1.0)),
        "noise rho=3": (ident, noise(3.0)),
    }
    for eps in (64, 8):
        p, pert, sigma = dp(eps)
        out[f"DP eps={eps}"] = (p, pert)
    return out, C


def train_eval(Ztr, Zte, Ttr, Tte, pert, joint, seed=0):
    """Ztr [N,16,d], Ttr uint8 [N,16,3,64,64]; returns mean PSNR/SSIM on the test uploads + 8 example reconstructions."""
    seed_all(seed)
    d = Ztr.shape[-1]
    mu, sd = Ztr.mean((0, 1), keepdim=True), Ztr.std((0, 1), keepdim=True).clamp_min(1e-6)  # monotone rescaling only
    g_te = torch.Generator(device=DEV).manual_seed(4242)
    Zte_in = pert(Zte, None, g_te) if pert is not None else Zte  # one fixed release for the test uploads
    dec = Decoder(d, joint).to(DEV)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    N = len(Ztr)
    pairs = torch.stack(torch.meshgrid(torch.arange(N, device=DEV), torch.arange(16, device=DEV), indexing="ij"), -1).flatten(0, 1)
    steps = EPOCHS * ((len(pairs) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for _ in range(EPOCHS):
        dec.train()
        for idx in pairs[torch.randperm(len(pairs), device=DEV)].split(256):
            if len(idx) < 2:
                continue
            i, p = idx[:, 0], idx[:, 1]
            z = Ztr[i] if joint else Ztr[i, p]
            if pert is not None:  # a fresh draw at every step: the attacker knows the mechanism, not the noise
                z = pert(z, None if joint else p)
            loss = F.mse_loss(dec((z - mu.squeeze(0)) / sd.squeeze(0) if not joint else (z - mu) / sd, p), Ttr[i, p].float() / 255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    dec.eval()
    P, S, RECS = [], [], []
    te_pairs = torch.stack(torch.meshgrid(torch.arange(len(Zte), device=DEV), torch.arange(16, device=DEV), indexing="ij"), -1).flatten(0, 1)
    with torch.no_grad():
        for idx in te_pairs.split(1024):
            i, p = idx[:, 0], idx[:, 1]
            z = Zte_in[i] if joint else Zte_in[i, p]
            rec = dec((z - mu.squeeze(0)) / sd.squeeze(0) if not joint else (z - mu) / sd, p)
            tgt = Tte[i, p].float() / 255
            P.append(psnr(rec, tgt))
            S.append(ssim(rec, tgt))
            RECS.append(F.avg_pool2d(rec, 4).half())
        ex_i = torch.tensor([e % len(Zte) for e in EX], device=DEV)
        ex_p = torch.zeros_like(ex_i)
        z = Zte_in[ex_i] if joint else Zte_in[ex_i, ex_p]
        ex = (dec((z - mu.squeeze(0)) / sd.squeeze(0) if not joint else (z - mu) / sd, ex_p) * 255).byte().cpu().numpy()
    tgt_all = F.avg_pool2d(Tte.flatten(0, 1).float() / 255, 4)
    rr = reid(torch.cat(RECS).float(), tgt_all, chunk=1024) if not joint else None
    out = {"psnr": torch.cat(P).mean().item(), "ssim": torch.cat(S).mean().item(), "train_loss_last": loss.item()}
    if rr:
        out["reid"] = rr
    return out, ex


def main():
    Ftr, Fte = load_feature("train"), load_feature("test")
    Ttr, Tte = load_thumbs("train"), load_thumbs("test")
    res = {"backbone": "dinov2_g", "n_train_pairs": len(Ftr) * 16, "n_test_images": len(Fte) * 16, "epochs": EPOCHS, "single": {}, "joint": {}}
    ex_idx = torch.tensor([e % len(Fte) for e in EX], device=DEV)
    examples = {"original": Tte[ex_idx, 0].cpu().numpy()}

    # reference points
    Tf = Tte.flatten(0, 1).float() / 255
    Tmean = (Ttr.flatten(0, 1).float() / 255).mean(0, keepdim=True)
    res["mean_image"] = {"psnr": psnr(Tmean.expand_as(Tf), Tf).mean().item(), "ssim": ssim(Tmean.expand_as(Tf), Tf).mean().item()}
    gal = F.normalize(Ftr.flatten(0, 1), dim=1)
    q = F.normalize(Fte.flatten(0, 1), dim=1)
    P, S, J = [], [], []
    for i in range(0, len(q), 2048):
        j = (q[i : i + 2048] @ gal.T).argmax(1)
        J.append(j)
        rec = Ttr.flatten(0, 1)[j].float() / 255
        P.append(psnr(rec, Tf[i : i + 2048]))
        S.append(ssim(rec, Tf[i : i + 2048]))
    res["nearest_neighbour"] = {"psnr": torch.cat(P).mean().item(), "ssim": torch.cat(S).mean().item()}
    mc = Tf.mean((2, 3), keepdim=True).expand_as(Tf)  # oracle: each image replaced by its own mean colour
    res["mean_colour_oracle"] = {"psnr": psnr(mc, Tf).mean().item(), "ssim": ssim(mc, Tf).mean().item(),
                                 "reid": reid(F.avg_pool2d(mc, 4), F.avg_pool2d(Tf, 4))}
    print("mean-colour oracle", res["mean_colour_oracle"], flush=True)
    J = torch.cat(J)
    examples["nearest_neighbour"] = Ttr.flatten(0, 1)[J[ex_idx * 16]].cpu().numpy()
    print("mean image", res["mean_image"], "| NN retrieval", res["nearest_neighbour"], flush=True)

    conds, C = conditions(Ftr)
    res["clip_norm"] = C
    KEY = ("full, 16-bit", "PCA-64", "DP eps=8")  # three seeds for the conditions discussed in the text
    jobs = [(n, m) for n in conds for m in ("single", "joint") if m == "single" or n in ("full, 16-bit", "full, 1-bit", "PCA-64", "PCA-16", "noise rho=3", "DP eps=8")]
    try:  # the best encoder of the paper is attacked too (full upload)
        Ctr, Cte = load_feature("train", "clip_l14"), load_feature("test", "clip_l14")
        jobs += [("CLIP-L, full, 16-bit", "single"), ("CLIP-L, full, 16-bit", "joint")]
    except FileNotFoundError:
        Ctr = None
    for name, mode in jobs:
        if name.startswith("CLIP"):
            Ztr, Zte, pert = Ctr, Cte, None
        else:
            prep, pert = conds[name]
            Ztr, Zte = prep(Ftr, Fte)
        if True:
            t0 = time.time()
            seeds = (0, 1, 2) if (name in KEY and mode == "single") else (0,)
            rs = []
            for sd_ in seeds:
                r_, ex_ = train_eval(Ztr, Zte, Ttr, Tte, pert, mode == "joint", seed=sd_)
                rs.append(r_)
                if sd_ == 0:
                    r, ex = dict(r_), ex_
            r["psnr_seeds"] = [x["psnr"] for x in rs]
            r["psnr"] = float(np.mean(r["psnr_seeds"]))
            r["ssim"] = float(np.mean([x["ssim"] for x in rs]))
            r["secs"] = round(time.time() - t0)
            res[mode][name] = r
            examples[f"{mode}:{name}"] = ex
            print(f"{mode:6s} {name:16s} PSNR {r['psnr']:.2f} dB  SSIM {r['ssim']:.3f}  ({r['secs']}s)", flush=True)
            save_json("vb_inversion.json", res)
            np.savez_compressed(os.path.join(OUT, "vb_inversion_examples.npz"), **{k.replace(" ", "_").replace(",", "").replace(":", "__").replace("=", ""): v for k, v in examples.items()})


if __name__ == "__main__":
    main()
