"""Stage 4 - feature-inversion attack on the uploaded DINOv2-g class tokens.

An honest-but-curious coordinator (or eavesdropper) holding image/feature pairs from the same distribution trains a
decoder  feature -> 64x64 RGB thumbnail  and applies it to the features of held-out test samples.
We repeat it for each "upload format" (full float32, PCA, scalar quantisation, Gaussian noise) so the privacy
leakage can be read next to the accuracy / bytes trade-off of stage 2 (same compressor, same noise definition).
Reference points: the mean training image, and nearest-neighbour retrieval in the training gallery.
Outputs: $PE_OUT/vb_inversion.json and vb_inversion_examples.npz (8 test thumbnails per condition).
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pe_common import DEV, save_json, seed_all

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
BACKBONE = os.environ.get("PE_INV_BACKBONE", "dinov2_g")
EPOCHS = int(os.environ.get("PE_INV_EPOCHS", "15"))

import s2_vertibench_heads as s2  # noqa: E402  (re-use the exact compressor of stage 2)


class Decoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, 512 * 4 * 4), nn.ReLU())
        blocks, c = [], 512
        for c_out in (256, 128, 64, 32):  # 4 -> 8 -> 16 -> 32 -> 64
            blocks += [nn.ConvTranspose2d(c, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU()]
            c = c_out
        self.up = nn.Sequential(*blocks, nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, f):
        return self.up(self.fc(f).view(-1, 512, 4, 4))


def ssim(x, y, win=11, sigma=1.5):
    """Mean SSIM over a batch of [B,3,H,W] images in [0,1] (Gaussian window, standard constants)."""
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None]
    k = (g @ g.T)[None, None].repeat(3, 1, 1, 1)
    f = lambda t: F.conv2d(t, k, groups=3)  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def psnr(x, y):
    return 10 * torch.log10(1.0 / F.mse_loss(x, y, reduction="none").mean((1, 2, 3)).clamp_min(1e-10))


def conditions():
    def noise(r):
        def prep(Xtr, Xva, Xt, seed):
            sd = Xtr.std(0, keepdim=True)
            g = torch.Generator(device=DEV).manual_seed(1000 + seed)
            return tuple(X + r * sd * torch.randn(X.shape, device=DEV, generator=g) for X in (Xtr, Xva, Xt))

        return prep

    return [
        ("float32 (full 1536-d)", s2.make_compressor(1536, 32)),
        ("int8", s2.make_compressor(1536, 8)),
        ("int4", s2.make_compressor(1536, 4)),
        ("1-bit", s2.make_compressor(1536, 1)),
        ("PCA-256 float32", s2.make_compressor(256, 32)),
        ("PCA-64 float32", s2.make_compressor(64, 32)),
        ("PCA-16 float32", s2.make_compressor(16, 32)),
        ("PCA-64 int8", s2.make_compressor(64, 8)),
        ("PCA-16 int4", s2.make_compressor(16, 4)),
        ("noise 0.3 x std", noise(0.3)),
        ("noise 1 x std", noise(1.0)),
        ("noise 3 x std", noise(3.0)),
    ]


def main():
    Ftr = torch.from_numpy(np.load(f"{CACHE}/feat_{BACKBONE}_train.npy")).to(DEV).float()  # [N,16,D]
    Fte = torch.from_numpy(np.load(f"{CACHE}/feat_{BACKBONE}_test.npy")).to(DEV).float()
    Ttr = torch.from_numpy(np.load(f"{CACHE}/thumb64_train.npy")).to(DEV)  # uint8 [N,16,3,64,64]
    Tte = torch.from_numpy(np.load(f"{CACHE}/thumb64_test.npy")).to(DEV).flatten(0, 1).float() / 255
    Ntr, Nte = len(Ftr), len(Fte)
    res = {"backbone": BACKBONE, "n_train_pairs": Ntr * 16, "n_test_images": Nte * 16, "conditions": {}}
    examples = {"original": (Tte[:8] * 255).byte().cpu().numpy()}

    # reference 1: mean training image;  reference 2: nearest neighbour in the training gallery (cosine, same party slot)
    Tmean = (Ttr.flatten(0, 1).float() / 255).mean(0, keepdim=True)
    res["mean_image"] = {"psnr": psnr(Tmean.expand_as(Tte), Tte).mean().item(), "ssim": ssim(Tmean.expand_as(Tte), Tte).mean().item()}
    P, S, nn_rec = [], [], []
    gal = F.normalize(Ftr.flatten(0, 1), dim=1)
    gal_img = Ttr.flatten(0, 1)
    q = F.normalize(Fte.flatten(0, 1), dim=1)
    for i in range(0, len(q), 2048):
        j = (q[i : i + 2048] @ gal.T).argmax(1)
        rec = gal_img[j].float() / 255
        P.append(psnr(rec, Tte[i : i + 2048]))
        S.append(ssim(rec, Tte[i : i + 2048]))
        if i == 0:
            nn_rec = (rec[:8] * 255).byte().cpu().numpy()
    res["nearest_neighbour"] = {"psnr": torch.cat(P).mean().item(), "ssim": torch.cat(S).mean().item()}
    examples["nearest_neighbour"] = nn_rec
    print("mean image", res["mean_image"], "| NN retrieval", res["nearest_neighbour"], flush=True)

    for name, prep in conditions():
        t0 = time.time()
        Ztr, Zte, _ = prep(Ftr, Fte, Fte[:1], 0)  # statistics from the training rows only
        Ztr, Zte = Ztr.flatten(0, 1), Zte.flatten(0, 1)
        # standardise only for the decoder's numerical comfort (monotone, does not add information)
        mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True).clamp_min(1e-6)
        Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
        Yt = Ttr.flatten(0, 1)
        seed_all(0)
        dec = Decoder(Ztr.shape[1]).to(DEV)
        opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        steps = EPOCHS * ((len(Ztr) + 255) // 256)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        for ep in range(EPOCHS):
            dec.train()
            perm = torch.randperm(len(Ztr), device=DEV)
            for i in range(0, len(perm), 256):
                idx = perm[i : i + 256]
                if len(idx) < 2:
                    continue
                loss = F.mse_loss(dec(Ztr[idx]), Yt[idx].float() / 255)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        dec.eval()
        P, S, ex = [], [], None
        with torch.no_grad():
            for i in range(0, len(Zte), 1024):
                rec = dec(Zte[i : i + 1024])
                P.append(psnr(rec, Tte[i : i + 1024]))
                S.append(ssim(rec, Tte[i : i + 1024]))
                if i == 0:
                    ex = (rec[:8] * 255).byte().cpu().numpy()
        res["conditions"][name] = {"psnr": torch.cat(P).mean().item(), "ssim": torch.cat(S).mean().item(), "train_loss_last": loss.item(), "secs": round(time.time() - t0)}
        examples[name] = ex
        print(f"{name:24s} PSNR {res['conditions'][name]['psnr']:.2f} dB  SSIM {res['conditions'][name]['ssim']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        save_json("vb_inversion.json", res)
        np.savez_compressed(os.path.join(os.environ.get("PE_OUT", "/content/results"), "vb_inversion_examples.npz"), **{k.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-"): v for k, v in examples.items()})


if __name__ == "__main__":
    main()
