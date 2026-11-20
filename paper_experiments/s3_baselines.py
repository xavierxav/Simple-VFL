"""Stage 3 - end-to-end baselines on the raw VertiBench images, at FULL resolution (13 x 158 x 158) by default.

  python s3_baselines.py pack
      pack the 62,832 raw per-party .npy files into two uint8 arrays full13_{train,test}.npy [N,16,13,158,158]
  python s3_baselines.py splitnn --bands rgb|all [--seeds 0,1,2,3,4] [--epochs 40] [--bs 64]
      16 ResNet18 bottom models + MLP top, trained end-to-end (classic SplitNN, many communication rounds).
  python s3_baselines.py ssl --bands rgb|all [--epochs 100]
      FedOnce-style one-shot VFL: each party trains a SimCLR ResNet18 on its own unlabeled images, uploads the
      512-d features once, the active party trains the MLP head (same head/protocol as the foundation-model rows).
  --res 80 reproduces the cheaper variant on the 80x80 thumbnails of stage 1 (3x3 stem, no max-pool).
At full resolution the bottom models are standard ResNet18s (7x7 stride-2 stem + max-pool, only the input
channel count changed), trained from scratch.
Results: $PE_OUT/baseline_<mode>_<bands>[_r80].json (includes wall-clock seconds, for the cost comparison).
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align

from pe_common import DEV, auc_macro_ovr, evaluate, fmt, save_json, seed_all, split_val, summarize, train_head

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
RAW = os.environ.get("PE_RAW", "/content/vb_raw/data/Vertibench")
MEAN = torch.tensor([81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384])
STD = torch.tensor([37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326])
RGB = [3, 2, 1]


def pack_full(split):
    """Raw per-party files -> one uint8 array [N,16,13,H,W] on disk (values are already in the 0..255 range)."""
    path = f"{CACHE}/full13_{split}.npy"
    if os.path.exists(path + ".ok"):
        return path
    N = len(np.load(f"{CACHE}/labels_{split}.npy"))
    probe = np.load(f"{RAW}/satellite_party0_{split}_0.npy")
    print(split, "raw file", probe.shape, probe.dtype, "min/max", probe.min(), probe.max(), flush=True)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(N, 16) + probe.shape)

    def load(job):
        p, i = job
        a = np.load(f"{RAW}/satellite_party{p}_{split}_{i}.npy")
        return p, i, (a if a.dtype == np.uint8 else np.clip(np.rint(a), 0, 255).astype(np.uint8))

    jobs = [(p, i) for p in range(16) for i in range(N)]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for k in range(0, len(jobs), 2048):
            for p, i, a in ex.map(load, jobs[k : k + 2048]):
                out[i, p] = a
            print(f"pack {split} {min(k + 2048, len(jobs))}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    del out
    open(path + ".ok", "w").close()
    return path


def load_images(split, bands, res):
    """uint8 images [N,16,C,H,W]; on the GPU when small enough (RGB), otherwise kept in host RAM (13 bands)."""
    ch = RGB if bands == "rgb" else list(range(13))
    src = f"{CACHE}/thumb13_{split}.npy" if res == 80 else pack_full(split)
    t0 = time.time()
    a = np.load(src, mmap_mode="r")
    X = torch.from_numpy(np.ascontiguousarray(a[:, :, ch]))
    if X.numel() < 6e9:
        X = X.to(DEV)
    print(f"loaded {split} {tuple(X.shape)} on {X.device} in {time.time() - t0:.0f}s", flush=True)
    return X, ch


def take(X, idx):
    """Gather a batch of uint8 images wherever X lives and move it to the GPU."""
    return X[idx.to(X.device)].to(DEV, non_blocking=True)


def normalise(xb, ch):
    m, s = MEAN[ch].to(DEV)[None, :, None, None], STD[ch].to(DEV)[None, :, None, None]
    return (xb.float() - m) / s


def resnet18(c_in, small_stem=False, pretrained=False):
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    if pretrained:  # ImageNet weights: keep the RGB stem
        assert c_in == 3 and not small_stem
    elif small_stem:  # 80x80 thumbnails: 3x3 stem, no max-pool
        m.conv1 = nn.Conv2d(c_in, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
    else:  # full resolution: standard ImageNet stem, only the number of input channels changes
        m.conv1 = nn.Conv2d(c_in, 64, 7, 2, 3, bias=False)
    m.fc = nn.Identity()
    return m


def flip_aug(x):
    f = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(3), x)
    f = torch.rand(len(x), device=x.device) < 0.5
    return torch.where(f[:, None, None, None], x.flip(2), x)


# ------------------------------------------------------------------ SplitNN
class SplitNN(nn.Module):
    def __init__(self, c_in, n_cls, small_stem=False, pretrained=False):
        super().__init__()
        self.bottoms = nn.ModuleList([resnet18(c_in, small_stem, pretrained) for _ in range(16)])
        self.top = nn.Sequential(nn.Linear(16 * 512, 1000), nn.BatchNorm1d(1000), nn.ReLU(), nn.Linear(1000, n_cls))

    def forward(self, xs):  # xs: list of 16 tensors [B,C,H,W]
        return self.top(torch.cat([b(x) for b, x in zip(self.bottoms, xs)], 1))


def run_splitnn(a):
    ytr_all = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    Xtr_all, ch = load_images("train", a.bands, a.res)
    Xte, _ = load_images("test", a.bands, a.res)
    ncls = int(ytr_all.max()) + 1
    runs, secs, probs = [], [], []
    for seed in [int(s) for s in a.seeds.split(",")]:
        seed_all(seed)
        tr, va = split_val(len(ytr_all), seed, 0.15, ytr_all.cpu().numpy())
        tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
        model = SplitNN(len(ch), ncls, a.res == 80, a.pretrained).to(DEV).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4 if a.pretrained else 2e-3, total_steps=steps)
        scaler = torch.amp.GradScaler()
        best, best_state = 1e9, None
        t0 = time.time()

        def forward(X, idx, train):
            xb = take(X, idx)
            xs = [normalise(xb[:, p], ch) for p in range(16)]
            if train:
                xs = [flip_aug(x) for x in xs]
            xs = [x.contiguous(memory_format=torch.channels_last) for x in xs]
            with torch.autocast("cuda", dtype=torch.float16):
                return model(xs)

        for ep in range(a.epochs):
            model.train()
            perm = tr[torch.randperm(len(tr), device=DEV)]
            for i in range(0, len(perm), a.bs):
                idx = perm[i : i + a.bs]
                if len(idx) < 2:
                    continue
                loss = F.cross_entropy(forward(Xtr_all, idx, True).float(), ytr_all[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            model.eval()
            with torch.no_grad():
                vl = sum(F.cross_entropy(forward(Xtr_all, va[i : i + 256], False).float(), ytr_all[va[i : i + 256]], reduction="sum").item() for i in range(0, len(va), 256)) / len(va)
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"seed {seed} ep {ep + 1}/{a.epochs} val_loss {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(yte), 256):
                idx = torch.arange(i, min(len(yte), i + 256), device=DEV)
                probs.append(F.softmax(forward(Xte, idx, False).float(), 1))
        p = torch.cat(probs)
        runs.append({"acc": (p.argmax(1) == yte).float().mean().item(), "auc": auc_macro_ovr(yte.cpu().numpy(), p.cpu().numpy())})
        probs.append(p.cpu().numpy())
        secs.append(time.time() - t0)
        print("SEED", seed, runs[-1], flush=True)
        save_json(f"baseline_splitnn_{a.bands}{a.tag}.json", {"summary": summarize(runs), "secs_per_seed": secs, "epochs": a.epochs, "batch_size": a.bs,
                                                              "pretrained": a.pretrained, "test_probs_seed_mean": np.mean(probs, 0).round(5).tolist(), "input_hw": int(Xte.shape[-1]), "gpu": torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu", "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0})
    print("SplitNN", a.bands, fmt(summarize(runs)), flush=True)


# ------------------------------------------------------------------ SimCLR per party (FedOnce-style)
def crop_resize(x, out, scale=(0.4, 1.0)):
    B, _, H, W = x.shape
    s = torch.empty(B, device=x.device).uniform_(*scale)
    r = torch.exp(torch.empty(B, device=x.device).uniform_(np.log(3 / 4), np.log(4 / 3)))
    w = (H * torch.sqrt(s * r)).clamp(16, W)
    h = (H * torch.sqrt(s / r)).clamp(16, H)
    x0 = torch.rand(B, device=x.device) * (W - w)
    y0 = torch.rand(B, device=x.device) * (H - h)
    boxes = torch.stack([torch.arange(B, device=x.device).float(), x0, y0, x0 + w, y0 + h], 1)
    return roi_align(x, boxes, output_size=out, aligned=True)


def simclr_view(x, out):
    x = flip_aug(crop_resize(x, out))
    k = torch.randint(0, 4, (1,)).item()
    x = torch.rot90(x, k, (2, 3))
    gain = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    bias = torch.empty(len(x), x.shape[1], 1, 1, device=x.device).uniform_(-0.3, 0.3)
    return x * gain + bias + 0.05 * torch.randn_like(x)


def nt_xent(z1, z2, t=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / t
    n = len(z1)
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def run_ssl(a):
    ytr_all = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    ncls = int(ytr_all.max()) + 1
    Xtr, ch = load_images("train", a.bands, a.res)
    Xte, _ = load_images("test", a.bands, a.res)
    crop = 64 if a.res == 80 else 128  # SimCLR views: random resized crops; features: whole image
    feats_tr = torch.zeros(len(ytr_all), 16, 512, device=DEV)
    feats_te = torch.zeros(len(yte), 16, 512, device=DEV)
    t0 = time.time()
    for p in range(16):
        seed_all(100 + p)
        enc = resnet18(len(ch), a.res == 80).to(DEV).to(memory_format=torch.channels_last)
        proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).to(DEV)
        opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-5)
        n = len(Xtr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * (n // a.bs))
        scaler = torch.amp.GradScaler()
        Xp_tr, Xp_te = Xtr[:, p].to(DEV), Xte[:, p].to(DEV)  # this party's images only (<= 1.3 GB uint8)
        for ep in range(a.epochs):
            enc.train()
            perm = torch.randperm(n, device=DEV)
            tot = 0.0
            for i in range(0, n - a.bs + 1, a.bs):  # all unlabeled TRAIN-split images of this party (test never seen)
                x = normalise(Xp_tr[perm[i : i + a.bs]], ch)
                with torch.autocast("cuda", dtype=torch.float16):
                    z1, z2 = proj(enc(simclr_view(x, crop))), proj(enc(simclr_view(x, crop)))
                loss = nt_xent(z1.float(), z2.float())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            if ep % 10 == 9 or ep == 0:
                print(f"party {p} ep {ep + 1} loss {tot / (n // a.bs):.3f} ({time.time() - t0:.0f}s)", flush=True)
        enc.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for X, out in ((Xp_tr, feats_tr), (Xp_te, feats_te)):
                for i in range(0, len(X), 256):
                    out[i : i + 256, p] = enc(normalise(X[i : i + 256], ch)).float()
        del Xp_tr, Xp_te
    ssl_secs = time.time() - t0
    runs = []
    for seed in range(5):
        tr, va = split_val(len(ytr_all), seed, 0.15, ytr_all.cpu().numpy())
        tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
        m = train_head(feats_tr[tr], ytr_all[tr], feats_tr[va], ytr_all[va], ncls, "mlp2", seed=seed)
        acc, auc = evaluate(m, feats_te, yte)
        runs.append({"acc": acc, "auc": auc})
    s = summarize(runs)
    print("FedOnce-SSL", a.bands, fmt(s), f"SSL pre-training {ssl_secs:.0f}s", flush=True)
    save_json(f"baseline_ssl_{a.bands}{a.tag}.json", {"summary": s, "ssl_pretrain_secs_all_parties": ssl_secs, "epochs": a.epochs, "batch_size": a.bs, "feature_dim": 512, "input_hw": int(Xte.shape[-1]), "crop_hw": crop, "gpu": torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "splitnn", "ssl"])
    ap.add_argument("--bands", default="rgb", choices=["rgb", "all"])
    ap.add_argument("--res", default="full", choices=["full", "80"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-initialised ResNet-18 bottoms (RGB only)")
    a = ap.parse_args()
    a.res = 80 if a.res == "80" else "full"
    a.tag = ("_r80" if a.res == 80 else "") + ("_pt" if a.pretrained else "")
    if a.mode == "pack":
        from s1_vertibench_features import ensure_data  # download check + 7z extraction of the raw files

        RAW = ensure_data()
        os.makedirs(CACHE, exist_ok=True)
        for sp in ("train", "test"):
            if not os.path.exists(f"{CACHE}/labels_{sp}.npy"):
                np.save(f"{CACHE}/labels_{sp}.npy", np.load(f"{RAW}/label_{sp}.npy").astype(np.int64))
            pack_full(sp)
    elif a.mode == "splitnn":
        a.epochs, a.bs = a.epochs or 40, a.bs or 64
        run_splitnn(a)
    else:
        a.epochs, a.bs = a.epochs or (40 if a.res == 80 else 100), a.bs or 256
        run_ssl(a)
