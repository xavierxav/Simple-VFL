"""Split learning (SplitNN) on natural-colour image parties, with conventional or foundation backbones.

One bottom model per party, a top model identical to the frozen-feature head (concat -> Linear(K*d,1000)-BN-ReLU-
Linear), trained end to end; the checkpoint with the lowest validation loss is tested.  Backbones:
  resnet18        ResNet-18 from scratch               (AdamW, one-cycle, peak lr 2e-3)
  resnet18_in     ResNet-18 ImageNet-initialised        (peak lr 3e-4)
  <vit>_lora      DINOv2 / CLIP ViT with LoRA (r=16) on the attention query/value projections, rest frozen (lr 5e-4)
  <vit>_ft        DINOv2 / CLIP ViT fully fine-tuned (backbone lr 2e-5, top 1e-3)
  <vit>_frozen    the same ViT frozen (sanity check: should match the one-shot frozen-feature head)
with <vit> in {dinov2_s, dinov2_b, clip_b16}.  Communication is counted as in split learning: every training step
sends the K party embeddings up and their gradients down (half precision), every evaluation pass the embeddings.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pe_common import DEV, metrics_from_proba, seed_all
from pe_encoders import CLIPN, HF, IMAGENET

_norm_cache = {}


def _norm(x, stats):
    key = (stats[0][0], x.device)
    if key not in _norm_cache:
        _norm_cache[key] = (torch.tensor(stats[0], device=x.device)[None, :, None, None], torch.tensor(stats[1], device=x.device)[None, :, None, None])
    m, s = _norm_cache[key]
    return (x - m) / s


class Bottom(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        base = kind.rsplit("_", 1)[0] if kind.endswith(("_lora", "_ft", "_frozen")) else kind
        self.stats = CLIPN if base.startswith("clip") else IMAGENET
        if base in ("resnet18", "resnet18_in"):
            import torchvision

            m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if base == "resnet18_in" else None)
            m.fc = nn.Identity()
            self.net, self.dim, self.vit = m, 512, False
            return
        from transformers import AutoModel, CLIPVisionModel

        m = CLIPVisionModel.from_pretrained(HF[base]) if base.startswith("clip") else AutoModel.from_pretrained(HF[base])
        self.dim, self.vit, self.clip = m.config.hidden_size, True, base.startswith("clip")
        if kind.endswith("_lora"):
            from peft import LoraConfig, get_peft_model

            tm = ["q_proj", "v_proj"] if self.clip else ["query", "value"]
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=tm))
        elif kind.endswith("_frozen"):
            for p in m.parameters():
                p.requires_grad_(False)
        self.net = m

    def forward(self, x):  # x float [B,3,H,W] in [0,1]
        x = _norm(x, self.stats)
        if not self.vit:
            return self.net(x)
        if self.clip:
            sq = x.shape[-1] == x.shape[-2] == 224
            return self.net(pixel_values=x, interpolate_pos_encoding=not sq).pooler_output
        return self.net(pixel_values=x).pooler_output


class SplitModel(nn.Module):
    def __init__(self, kinds, n_cls, hidden=1000):
        super().__init__()
        self.bottoms = nn.ModuleList([Bottom(k) for k in kinds])
        d = sum(b.dim for b in self.bottoms)
        self.top = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, n_cls))

    def forward(self, xs):
        return self.top(torch.cat([b(x) for b, x in zip(self.bottoms, xs)], 1))


def _augment(x, g):
    """GPU augmentation: random horizontal flip + random resized crop (scale 0.6-1) back to the input size."""
    B, _, H, W = x.shape
    flip = torch.rand(B, generator=g) < 0.5
    x = torch.where(flip.to(x.device)[:, None, None, None], x.flip(-1), x)
    s = torch.empty(B).uniform_(0.6, 1.0, generator=g).sqrt()
    cx = (torch.rand(B, generator=g) * 2 - 1) * (1 - s)
    cy = (torch.rand(B, generator=g) * 2 - 1) * (1 - s)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, cx, cy
    grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _batch(parties, idx, size):
    xs = []
    for P in parties:
        x = torch.from_numpy(np.ascontiguousarray(P[idx])).to(DEV, non_blocking=True).float() / 255
        if size is not None and tuple(x.shape[-2:]) != tuple(size):
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        xs.append(x)
    return xs


@torch.no_grad()
def _predict(model, parties, idx, size, bs):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            out.append(model(_batch(parties, idx[i : i + bs], size)).float())
    return torch.cat(out)


def train_split(parties, y, tr, va, te, kinds, n_cls, epochs=20, bs=64, size=None, seed=0, class_weight=None, log=print):
    """parties: list of uint8 arrays [N,3,h,w] (CPU, indexable); y: int numpy [N]; tr/va/te: index arrays.
    kinds: one backbone name per party.  size: (H,W) fed to the bottoms (None = stored size).
    Returns (test metrics dict, info dict)."""
    seed_all(seed)
    model = SplitModel(kinds, n_cls).to(DEV).to(memory_format=torch.channels_last)
    base = kinds[0]
    groups = []
    for b in model.bottoms:
        ps = [p for p in b.parameters() if p.requires_grad]
        if not ps:
            continue
        lr = 2e-3 if b.kind == "resnet18" else 3e-4 if b.kind == "resnet18_in" else 5e-4 if b.kind.endswith("_lora") else 2e-5
        groups.append({"params": ps, "lr": lr})
    groups.append({"params": model.top.parameters(), "lr": 2e-3 if base == "resnet18" else 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEV)
    yt = torch.as_tensor(y, device=DEV)
    g = torch.Generator().manual_seed(seed)
    best, best_state, t0 = float("inf"), None, time.time()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            if len(idx) < 2:
                continue
            xs = [_augment(x, g) for x in _batch(parties, idx, size)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                loss = F.cross_entropy(model(xs).float(), yt[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        lv = _predict(model, parties, va, size, 256)
        vl = F.cross_entropy(lv, yt[va]).item()
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in trainable or "running" in k or "num_batches" in k}
        log(f"  ep {ep + 1}/{epochs} val loss {vl:.4f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state, strict=False)
    p = F.softmax(_predict(model, parties, te, size, 256), 1).cpu().numpy()
    secs = time.time() - t0
    d = sum(b.dim for b in model.bottoms)
    n_steps_samples = epochs * len(tr)
    comm = (2 * n_steps_samples + epochs * len(va) + len(te)) * d * 2  # bytes, fp16 activations (+ gradients in training)
    info = {"secs": secs, "epochs": epochs, "bytes_total": int(comm), "bytes_per_train_sample": float(comm / len(tr)),
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0, "test_proba": p}
    m = metrics_from_proba(y[te], p)
    del model, opt
    torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return m, info
