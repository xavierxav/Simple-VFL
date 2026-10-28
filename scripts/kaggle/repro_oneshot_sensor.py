"""One-shot two-sensor VFL on a Kaggle T4: SPOT + averaged Sentinel embedding.

Each party uploads one 1536-d feature per sample (the Sentinel party averages its
16 revisits locally), then the coordinator trains a 2x1000 -> 4/7 head. This is
the two-party, single-upload counterpart of the late-fusion model.
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
CONFIG = "spot_and_sentinel_mean_embedding"
LABELS = [("ipcc", 7), ("simplified_ipcc", 4), ("sources", 4)]
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
    print(f"\n===== START oneshot_sensor {label} =====", flush=True)
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
        f"hydra.sweep.dir=outputs/sweeps/oneshot_sensor_{label}",
        f"runtime.seed={SEEDS}",
    ]
    result = subprocess.run(command, cwd=REPO, env=env, check=True)
    print(f"===== END oneshot_sensor {label} exit={result.returncode} =====", flush=True)

print("\n===== RESULTS (one-shot two-sensor) =====", flush=True)
pattern = re.compile(r"Test accuracy:\s*([0-9.]+),\s*Test AUC:\s*([0-9.]+)")
for label, _ in LABELS:
    accuracies = []
    aucs = []
    for log_path in glob.glob(f"{REPO}/outputs/sweeps/oneshot_sensor_{label}/**/*.log", recursive=True):
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
