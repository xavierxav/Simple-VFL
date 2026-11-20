"""Stage 8 - reproduction of the published VertiBench Satellite SplitNN result (81.17 +- 0.35 %).

Follows the task description of the VertiBench repository (Satellite.md, section 4):
  SplitResNet: one torchvision resnet18 (weights=None) per party, conv1 = Conv2d(13, 64, 9, stride=2, padding=3),
  fc = Identity; aggregation MLP [1000, 4] on the concatenated 16 x 512 outputs (torchvision.ops.MLP, ReLU, no norm);
  Adam lr 1e-5, weight decay 1e-5, StepLR(step_size=10, gamma=0.5), batch 32, 50 epochs, cross-entropy,
  no early stopping, accuracy on the official test set.
Not specified there, so chosen here: inputs scaled to [0, 1] (x / 255), no augmentation, training on the whole
official training set (no validation split), test accuracy after the last epoch.  The test accuracy of every epoch
is also logged, to show how much picking the best epoch on the test set would add.
The seed changes the initialisation and the batch order only: the train/test split is fixed by the benchmark files.

  python s8_vertibench_repro.py [--seeds 0,1,2,3,4] [--scale 255|std]
Output: $PE_OUT/vb_repro_splitresnet[_std].json
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pe_common import DEV, auc_macro_ovr, save_json, seed_all, summarize

CACHE = os.environ.get("PE_CACHE", "/content/cache/vb")
MEAN = torch.tensor([81.22718, 71.60518, 68.65638, 71.84505, 80.67904, 107.35618, 121.81858, 117.40219, 131.74495, 39.24643, 1.0040737, 111.348335, 78.19384])
STD = torch.tensor([37.995075, 38.487247, 37.658237, 46.64616, 44.534966, 42.94063, 46.846764, 46.01008, 50.308132, 24.336931, 4.5140524, 60.380325, 51.48326])


class SplitResNet(nn.Module):
    def __init__(self, n_parties=16, channels=13, kernel_size=9, agg_hidden=(1000, 4)):
        super().__init__()
        self.local = nn.ModuleList()
        for _ in range(n_parties):
            r = torchvision.models.resnet18(weights=None)
            r.fc = nn.Identity()
            r.conv1 = nn.Conv2d(channels, 64, kernel_size, stride=2, padding=3, bias=False)
            self.local.append(r)
        self.agg = torchvision.ops.MLP(512 * n_parties, list(agg_hidden))

    def forward(self, xs):
        return self.agg(torch.cat([r(x) for r, x in zip(self.local, xs)], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--scale", default="255", choices=["255", "std"])
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    Xtr = torch.from_numpy(np.load(f"{CACHE}/full13_train.npy"))  # uint8 [N,16,13,158,158] in host RAM (16 GB)
    Xte = torch.from_numpy(np.load(f"{CACHE}/full13_test.npy"))
    ytr = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    yte = torch.from_numpy(np.load(f"{CACHE}/labels_test.npy")).to(DEV)
    m, s = MEAN.to(DEV)[None, :, None, None], STD.to(DEV)[None, :, None, None]

    def norm(x):  # x uint8 [B,13,H,W] on DEV
        return x.float() / 255 if a.scale == "255" else (x.float() - m) / s

    def forward(model, X, idx):
        xb = X[idx.cpu()].to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            return model([norm(xb[:, p]).contiguous(memory_format=torch.channels_last) for p in range(16)])

    @torch.no_grad()
    def test(model):
        model.eval()
        P = []
        for i in range(0, len(yte), 128):
            P.append(F.softmax(forward(model, Xte, torch.arange(i, min(len(yte), i + 128))).float(), 1))
        P = torch.cat(P)
        return (P.argmax(1) == yte).float().mean().item(), auc_macro_ovr(yte.cpu().numpy(), P.cpu().numpy())

    runs, curves, secs = [], [], []
    tag = "" if a.scale == "255" else "_std"
    for seed in [int(x) for x in a.seeds.split(",")]:
        seed_all(seed)
        model = SplitResNet().to(DEV).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        scaler = torch.amp.GradScaler()
        g = torch.Generator().manual_seed(seed)
        curve, t0 = [], time.time()
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(len(ytr), generator=g)
            tot = 0.0
            for i in range(0, len(perm), 32):
                idx = perm[i : i + 32]
                loss = F.cross_entropy(forward(model, Xtr, idx).float(), ytr[idx.to(DEV)])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(idx)
            sched.step()
            acc, auc = test(model)
            curve.append({"epoch": ep + 1, "train_loss": tot / len(ytr), "test_acc": acc * 100, "test_auc": auc * 100})
            print(f"seed {seed} ep {ep + 1}/{a.epochs} loss {tot / len(ytr):.4f} test acc {acc * 100:.2f} ({time.time() - t0:.0f}s)", flush=True)
        runs.append({"acc": curve[-1]["test_acc"] / 100, "auc": curve[-1]["test_auc"] / 100})
        curves.append(curve)
        secs.append(time.time() - t0)
        best = summarize([{"acc": max(c["test_acc"] for c in cv) / 100, "auc": max(c["test_auc"] for c in cv) / 100} for cv in curves])
        save_json(f"vb_repro_splitresnet{tag}.json", {"last_epoch": summarize(runs), "best_epoch_on_test": best, "curves": curves, "secs_per_seed": secs,
                                                      "gpu": torch.cuda.get_device_name(), "scale": a.scale, "published": {"acc": 81.17, "acc_std": 0.35}})
        print("SEED", seed, runs[-1], flush=True)


if __name__ == "__main__":
    main()
