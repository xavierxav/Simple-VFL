"""Stage 6 - wall-clock cost of the one-shot pipeline on one GPU, for the cost comparison with the stage-3 baselines.

(i) frozen-encoder forward pass, timed on all training images of party 0 and extrapolated to the 16 parties x
(train + test); data loading is timed separately; (ii) training of the MLP head on the cached DINOv2-g features.
Run it on an otherwise idle GPU.  Output: $PE_OUT/vb_timing_<GPU>.json (e.g. A100, L4), so runs on several GPUs coexist.
"""

import os
import time

import numpy as np
import torch

from pe_common import DEV, save_json, split_val, train_head
from s1_vertibench_features import CACHE, build, ensure_data

N_TOTAL = 3142 + 785


def time_encoder(name, data, n=3142, chunk=256):
    fn, _ = build(name)
    load = lambda idx: torch.from_numpy(np.stack([np.load(f"{data}/satellite_party0_train_{i}.npy") for i in idx]).astype(np.float32))  # noqa: E731
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        fn(load(range(64)).to(DEV))  # warm-up
    torch.cuda.synchronize()
    t_io = t_gpu = 0.0
    for s in range(0, n, chunk):
        t0 = time.time()
        x = load(range(s, min(n, s + chunk))).to(DEV)
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            fn(x)
        torch.cuda.synchronize()
        t_io, t_gpu = t_io + t1 - t0, t_gpu + time.time() - t1
    try:
        params = sum(p.numel() for p in fn.__closure__[0].cell_contents.parameters())
    except Exception:
        params = None
    r = {"gpu_ms_per_image": 1000 * t_gpu / n, "io_ms_per_image": 1000 * t_io / n, "gpu_min_all_parties": t_gpu / n * 16 * N_TOTAL / 60, "params_m": params / 1e6 if params else None}
    print(name, r, flush=True)
    return r


def main():
    data = ensure_data()
    res = {"gpu": torch.cuda.get_device_name()}
    for name in ("dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_l14"):
        res[name] = time_encoder(name, data)
        torch.cuda.empty_cache()
    tag = next((g for g in ("A100", "H100", "L4", "T4", "G4") if g in res["gpu"]), "gpu")
    if not os.path.exists(f"{CACHE}/feat_dinov2_g_train.npy"):  # VM without cached features: encoders only
        save_json(f"vb_timing_{tag}.json", res)
        return
    y = torch.from_numpy(np.load(f"{CACHE}/labels_train.npy")).to(DEV)
    X = torch.from_numpy(np.load(f"{CACHE}/feat_dinov2_g_train.npy")).to(DEV).float()
    tr, va = split_val(len(y), 0, 0.15, y.cpu().numpy())
    tr, va = torch.from_numpy(tr).to(DEV), torch.from_numpy(va).to(DEV)
    torch.cuda.synchronize()
    t0 = time.time()
    train_head(X[tr], y[tr], X[va], y[va], int(y.max()) + 1, "mlp2", seed=0)
    torch.cuda.synchronize()
    res["head_train_secs_dinov2_g_mlp2"] = time.time() - t0
    print(res, flush=True)
    save_json(f"vb_timing_{tag}.json", res)


if __name__ == "__main__":
    main()
