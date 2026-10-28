"""VertiBench replication on a Kaggle T4 (download -> DINOv2 features -> training).

The archive is 8.2 GB compressed / 19 GiB on disk, so everything happens under
/tmp (node-local, ~1 TB free in this image) instead of the 20 GiB /kaggle quota.
"""

import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

REPO = "/kaggle/working/Simple-VFL"
WORK = "/tmp/vertibench"
ARCHIVE = os.path.join(WORK, "Vertibench.7z")
DATA_ROOT = os.path.join(WORK, "data")
SATELLITE = os.path.join(DATA_ROOT, "satellite_dataset")
FEATURES = "extracted_features_giant_rgb_Vertibench"
URL = "https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z"
EXPECTED_ARCHIVE_BYTES = 8_222_228_129


def run(command, **kwargs):
    kwargs.setdefault("check", True)
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, **kwargs)


print("=== clone ===", flush=True)
if not os.path.isdir(REPO):
    run(["git", "clone", "--depth", "1", "--branch", "main", "https://github.com/xavierxav/Simple-VFL.git", REPO])
print(run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout, flush=True)

print("=== deps ===", flush=True)
run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "addict", "py7zr"], check=True)

print("=== download ===", flush=True)
os.makedirs(WORK, exist_ok=True)
if not os.path.exists(ARCHIVE) or os.path.getsize(ARCHIVE) < 8_000_000_000:
    run(["curl", "-L", "--retry", "5", "--retry-delay", "10", "-o", ARCHIVE, URL])
if os.path.getsize(ARCHIVE) != EXPECTED_ARCHIVE_BYTES:
    raise RuntimeError(f"unexpected archive size: {os.path.getsize(ARCHIVE)} bytes")
print("archive", round(os.path.getsize(ARCHIVE) / 2**30, 2), "GiB", flush=True)

print("=== extract ===", flush=True)
if not os.path.isdir(os.path.join(SATELLITE, "Vertibench")):
    os.makedirs(SATELLITE, exist_ok=True)
    import py7zr

    start = time.time()
    with py7zr.SevenZipFile(ARCHIVE, "r") as archive:
        archive.extractall(path=SATELLITE)
    print("extracted in", round(time.time() - start, 1), "s", flush=True)
files = glob.glob(os.path.join(SATELLITE, "Vertibench", "satellite_party*.npy"))
print("raw files", len(files), flush=True)
assert len(files) == 62832, len(files)

print("=== features ===", flush=True)
feature_dir = os.path.join(SATELLITE, FEATURES)
if len(glob.glob(os.path.join(feature_dir, "satellite_party*.npy"))) < 60000:
    run(
        [
            sys.executable,
            os.path.join(REPO, "worldstrat_downloading_and_features_extraction", "extract_features_vertibench.py"),
            "--data-root",
            SATELLITE,
            "--output-dir",
            FEATURES,
            "--batch-size",
            "256",
        ],
        cwd=os.path.join(REPO, "worldstrat_downloading_and_features_extraction"),
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )

for name in ("label_train.npy", "label_test.npy"):
    shutil.copyfile(os.path.join(SATELLITE, "Vertibench", name), os.path.join(feature_dir, name))
features = glob.glob(os.path.join(feature_dir, "satellite_party*.npy"))
print("feature files", len(features), flush=True)
assert len(features) == 62832, len(features)

print("=== gpu ===", flush=True)
import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), flush=True)

env = dict(os.environ, DATA_ROOT=DATA_ROOT, PYTHONPATH=REPO, PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")

RUNS = [
    ("vertical_2layers", ["+experiment_satellite_features=vertibench"]),
    ("vertical_linear", ["+experiment_satellite_features=vertibench", "model=SatelliteFeaturesCentralized"]),
    ("horizontal_1view", ["+experiment_satellite_features=vertibench_horizontal"]),
]

for name, overrides in RUNS:
    print(f"\n===== START vertibench {name} =====", flush=True)
    command = [
        sys.executable,
        "-m",
        "simple_vfl",
        "-m",
        "--config-name=satellite",
        *overrides,
        "runtime.device=auto",
        "runtime.num_workers=0",
        f"hydra.sweep.dir=outputs/sweeps/vertibench_{name}",
        "runtime.seed=0,1,2,3,4",
    ]
    result = subprocess.run(command, cwd=REPO, env=env, check=True)
    print(f"===== END vertibench {name} exit={result.returncode} =====", flush=True)

print("\n===== RESULTS (VertiBench) =====", flush=True)
pattern = re.compile(r"Test accuracy:\s*([0-9.]+),\s*Test AUC:\s*([0-9.]+)")
for name, _ in RUNS:
    accuracies = []
    aucs = []
    for log_path in glob.glob(f"{REPO}/outputs/sweeps/vertibench_{name}/**/*.log", recursive=True):
        with open(log_path, encoding="utf-8", errors="ignore") as handle:
            for match in pattern.finditer(handle.read()):
                accuracies.append(float(match.group(1)))
                aucs.append(float(match.group(2)))
    if len(accuracies) != 5:
        raise RuntimeError(f"{name}: expected five test results, got {len(accuracies)}")
    print(
        f"{name}: n={len(accuracies)} "
        f"acc={statistics.mean(accuracies) * 100:.2f} +/- {statistics.pstdev(accuracies) * 100:.2f} "
        f"auc={statistics.mean(aucs) * 100:.2f} +/- {statistics.pstdev(aucs) * 100:.2f}",
        flush=True,
    )
