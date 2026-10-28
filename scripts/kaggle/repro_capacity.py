"""Capacity-matched one-view control for the 16-view Sentinel experiment.

One random Sentinel view per sample, but the central head is widened to 15,932
hidden units so it has ~24.58M parameters, matching the 16-view model's
(24,576 -> 1000 -> 4) head. Runs on sources and IPCC.
"""

import glob
import os
import re
import statistics
import subprocess
import sys

REPO = "/kaggle/working/Simple-VFL"
DATA = "/kaggle/temp/data"
CANDIDATES = [
    "/kaggle/input/simple-vfl-ipcc-data-bin/satellite_dataset.bin",
    "/kaggle/input/datasets/xavierdesprez/simple-vfl-ipcc-data-bin/satellite_dataset.bin",
]
CONFIG = "sentinel_horizontal_capacity"
LABELS = [("sources", 4), ("ipcc", 7)]
SEEDS = "0,1,2,3,4"

print("=== clone ===", flush=True)
if not os.path.isdir(REPO):
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "main", "https://github.com/xavierxav/Simple-VFL.git", REPO],
        check=True,
    )
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout, flush=True)

print("=== deps ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "addict"], check=True)

print("=== data ===", flush=True)
print("input:", os.listdir("/kaggle/input"), flush=True)
archive = next((path for path in CANDIDATES if os.path.exists(path)), None)
if archive is None:
    raise FileNotFoundError(f"feature archive not found in {CANDIDATES}")
if not os.path.isdir(f"{DATA}/satellite_dataset"):
    os.makedirs(DATA, exist_ok=True)
    print(f"extracting {archive}", flush=True)
    subprocess.run(["tar", "-xf", archive, "-C", DATA], check=True)
print("data ready", flush=True)

print("=== cache patch ===", flush=True)
datasets_py = f"{REPO}/simple_vfl/dataset/datasets.py"
with open(datasets_py, encoding="utf-8") as handle:
    source = handle.read()
source = source.replace("@lru_cache(maxsize=8192)", "@lru_cache(maxsize=None)")
with open(datasets_py, "w", encoding="utf-8") as handle:
    handle.write(source)
print("lru_cache(maxsize=None)", flush=True)

import torch  # noqa: E402

print("=== gpu ===", flush=True)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), flush=True)

env = dict(os.environ, DATA_ROOT=DATA, PYTHONPATH=REPO, PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")

for label, num_classes in LABELS:
    print(f"\n===== START capacity_1view {label} =====", flush=True)
    command = [
        sys.executable,
        "-m",
        "simple_vfl",
        "-m",
        "--config-name=satellite",
        f"+experiment_satellite_features={CONFIG}",
        f"dataset.label_type={label}",
        f"dataset.num_classes={num_classes}",
        "runtime.device=auto",
        "runtime.num_workers=0",
        f"hydra.sweep.dir=outputs/sweeps/capacity_1view_{label}",
        f"runtime.seed={SEEDS}",
    ]
    result = subprocess.run(command, cwd=REPO, env=env, check=True)
    print(f"===== END capacity_1view {label} exit={result.returncode} =====", flush=True)

print("\n===== RESULTS (capacity-matched 1-view) =====", flush=True)
pattern = re.compile(r"Test accuracy:\s*([0-9.]+),\s*Test AUC:\s*([0-9.]+)")
for label, _ in LABELS:
    accuracies = []
    aucs = []
    for log_path in glob.glob(f"{REPO}/outputs/sweeps/capacity_1view_{label}/**/*.log", recursive=True):
        with open(log_path, encoding="utf-8", errors="ignore") as handle:
            for match in pattern.finditer(handle.read()):
                accuracies.append(float(match.group(1)))
                aucs.append(float(match.group(2)))
    if len(accuracies) != 5:
        raise RuntimeError(f"{label}: expected five results, got {len(accuracies)}")
    print(
        f"{label}: n={len(accuracies)} "
        f"acc={statistics.mean(accuracies) * 100:.2f} +/- {statistics.pstdev(accuracies) * 100:.2f} "
        f"auc={statistics.mean(aucs) * 100:.2f} +/- {statistics.pstdev(aucs) * 100:.2f}",
        flush=True,
    )
