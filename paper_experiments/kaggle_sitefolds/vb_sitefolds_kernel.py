"""Kaggle T4 kernel for stage 26 (VertiBench Satellite on site-grouped folds).

download (HF mirror of the VertiBench archive) -> pack (s26 data) -> features (s26 features) -> frozen heads (s26 frozen)
-> SplitNN per fold (s26 splitnn) -> paired analysis (s26 analysis).  Everything large lives under /tmp (node-local
disk); results and logs go to /kaggle/working (the kernel output).  With two GPUs, GPU 1 runs SplitNN folds 1,3,4 while
GPU 0 extracts the features, trains the frozen heads, then runs SplitNN folds 0,2.
"""

import glob
import os
import subprocess
import sys
import time

BR = "claude/simple-vfl-paper-publication-qx73nw"
REPO = "/tmp/repo"
WORK = "/tmp/vertibench"
ARCHIVE = os.path.join(WORK, "Vertibench.7z")
SAT = os.path.join(WORK, "satellite_dataset")
RAW = os.path.join(SAT, "Vertibench")
CACHE = "/tmp/vbcache"
OUT = "/kaggle/working/results"
LOGS = "/kaggle/working/logs"
URL = "https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z"
for d in (WORK, CACHE, OUT, LOGS):
    os.makedirs(d, exist_ok=True)
T0 = time.time()


def run(cmd, **kw):
    kw.setdefault("check", True)
    print(f"[{time.time() - T0:.0f}s] +", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


run(["git", "clone", "-q", "--depth", "1", "--branch", BR, "https://github.com/xavierxav/Simple-VFL.git", REPO])
print(run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout, flush=True)
run([sys.executable, "-m", "pip", "install", "-q", "py7zr"])
run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; df -h /tmp | tail -1", shell=True)

if not os.path.exists(ARCHIVE) or os.path.getsize(ARCHIVE) < 8_000_000_000:
    run(["curl", "-sSL", "--retry", "8", "--retry-delay", "10", "-o", ARCHIVE, URL])
print("archive", round(os.path.getsize(ARCHIVE) / 2**30, 2), "GiB", flush=True)
if len(glob.glob(os.path.join(RAW, "satellite_party*.npy"))) < 62832:
    import py7zr

    t = time.time()
    with py7zr.SevenZipFile(ARCHIVE, "r") as a:
        a.extractall(path=SAT)
    print("extracted in", round(time.time() - t), "s", flush=True)
assert len(glob.glob(os.path.join(RAW, "satellite_party*.npy"))) == 62832
os.remove(ARCHIVE)

ENV = dict(os.environ, PE_RAW=RAW, PE_CACHE=CACHE, PE_OUT=OUT, PYTHONUNBUFFERED="1")
S26 = os.path.join(REPO, "paper_experiments", "s26_vb_sitefolds.py")
CWD = os.path.join(REPO, "paper_experiments")


def stage(name, args, gpu=None):
    env = dict(ENV) if gpu is None else dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(os.path.join(LOGS, f"{name}.log"), "w") as f:
        return subprocess.Popen([sys.executable, "-u", S26] + args, cwd=CWD, env=env, stdout=f, stderr=subprocess.STDOUT)


p = stage("data", ["data"])
assert p.wait() == 0, "data stage failed"
print(f"[{time.time() - T0:.0f}s] packed", flush=True)
run(f"rm -rf {RAW}", shell=True)  # the packed arrays are all that is needed now

import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)


def chain(name, stages, gpu):
    """Run s26 stages one after the other on one GPU (a shell chain, so both GPUs can work at once)."""
    cmd = " && ".join(f"{sys.executable} -u {S26} {' '.join(a)} > {LOGS}/{n}.log 2>&1" for n, a in stages)
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen(["bash", "-c", cmd], cwd=CWD, env=env)


A = [("features", ["features"]), ("frozen", ["frozen"])]
if n_gpu >= 2:
    procs = [chain("gpu0", A + [("splitnn_02", ["splitnn", "--folds", "0,2"])], 0),
             chain("gpu1", [("splitnn_134", ["splitnn", "--folds", "1,3,4"])], 1)]
else:
    procs = [chain("gpu0", A + [("splitnn_all", ["splitnn", "--folds", "0,1,2,3,4"])], 0)]
while any(q.poll() is None for q in procs):
    time.sleep(300)
    for lg in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        lines = open(lg, errors="ignore").read().strip().splitlines()
        print(f"[{time.time() - T0:.0f}s] {os.path.basename(lg)}: {lines[-1][:160] if lines else ''}", flush=True)
print("exit codes", [q.returncode for q in procs], flush=True)
r = subprocess.run([sys.executable, "-u", S26, "analysis", OUT], cwd=CWD, env=ENV, capture_output=True, text=True)
open(os.path.join(LOGS, "analysis.log"), "w").write(r.stdout + r.stderr)
print(r.stdout[-6000:], r.stderr[-3000:], flush=True)
print(f"done in {time.time() - T0:.0f}s", flush=True)
