"""Turn the result JSONs (paper_results/) into LaTeX tables and figures (paper/src/gen/).

usage: python make_assets.py [results_dir] [out_dir]
Everything is generated from the JSON so the numbers in the paper cannot drift from the logs.
Missing inputs are skipped, so the script can be run on partial results.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "paper_results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/src/gen"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
C = {"blue": "#2a6fbb", "orange": "#d9772b", "green": "#3a9a5b", "red": "#c0392b", "grey": "#7f8c8d", "purple": "#8e5bb5"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pm(s, key="acc", d=2):
    return f"{s[key]:.{d}f}\\,{{\\scriptsize$\\pm${s[key + '_std']:.{d}f}}}"


def kib(b):
    return f"{b / 1024:.1f}" if b < 1024 * 1024 else f"{b / 1024 / 1024:.2f}\\,Mi"


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)
    print("wrote", name)


NAMES = {
    "dinov2_g": "DINOv2 ViT-g/14", "dinov2_l": "DINOv2 ViT-L/14", "dinov2_b": "DINOv2 ViT-B/14", "dinov2_s": "DINOv2 ViT-S/14",
    "clip_l14": "CLIP ViT-L/14", "resnet50_in1k": "ResNet-50 (ImageNet-1k, supervised)", "dinov2_b_random": "DINOv2 ViT-B/14, random init",
    "dinov2_g_4comp": "DINOv2 ViT-g/14, 4 band composites", "ssl4eo_vits16": "SSL4EO-S12 ViT-S/16 (13 bands)",
}

# ------------------------------------------------------------------ backbones table
bb = load("vb_backbones.json")
if bb:
    rows = []
    for k in ["dinov2_g", "dinov2_l", "dinov2_b", "dinov2_s", "clip_l14", "resnet50_in1k", "ssl4eo_vits16", "dinov2_b_random", "dinov2_g_4comp"]:
        if k in bb:
            r = bb[k]
            rows.append(f"{NAMES[k]} & {r['dim']} & {pm(r['vertical_mlp2'])} & {pm(r['vertical_mlp2'], 'auc')} & {pm(r['vertical_linear'])} & {pm(r['horizontal_1view_mlp2'])} \\\\")
    write("tab_backbones.tex", "\n".join(rows))
    if "dinov2_g_solo_parties" in bb:
        solo = [s["acc"] for s in bb["dinov2_g_solo_parties"]]
        write("solo.tex", f"\\newcommand{{\\soloMin}}{{{min(solo):.2f}}}\\newcommand{{\\soloMax}}{{{max(solo):.2f}}}\\newcommand{{\\soloMean}}{{{np.mean(solo):.2f}}}\n")

# ------------------------------------------------------------------ pareto figure
par = load("vb_pareto.json")
if par:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    cols = {32: C["blue"], 16: C["purple"], 8: C["green"], 4: C["orange"], 2: C["red"], 1: C["grey"]}
    for bits in [32, 8, 4, 2, 1]:
        pts = sorted([(p["bytes_per_sample"], p["acc"], p["acc_std"], p["d"]) for p in par if p["bits"] == bits])
        if pts:
            x, y, e = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
            ax.errorbar(x, y, yerr=e, marker="o", ms=3, lw=1, capsize=1.5, color=cols[bits], label=f"{bits}-bit" if bits < 32 else "float32")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("one-time upload per sample, all 16 parties (bytes)")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(title="precision", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
    plt.close(fig)
    rows = []
    for p in sorted(par, key=lambda p: -p["bytes_per_sample"]):
        if p["bits"] in (32, 8, 4, 1) and p["d"] in (1536, 256, 64, 16):
            rows.append(f"{p['d']} & {p['bits']} & {p['bytes_per_sample'] / 1024:.2f} & {pm(p)} & {pm(p, 'auc')} \\\\")
    write("tab_pareto.tex", "\n".join(rows))

# ------------------------------------------------------------------ parties figure
pt = load("vb_parties.json")
if pt:
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ks = [s["k"] for s in pt["subsets"]]
    axs[0].errorbar(ks, [s["acc"] for s in pt["subsets"]], yerr=[s["acc_std"] for s in pt["subsets"]], marker="o", ms=3, capsize=2, color=C["blue"])
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(ks)
    axs[0].set_xticklabels(ks)
    axs[0].set_xlabel("number of parties $k$ (random subsets)")
    axs[0].set_ylabel("test accuracy (%)")
    for p_drop, col in zip([0.0, 0.25, 0.5], [C["grey"], C["orange"], C["green"]]):
        pts = sorted([(m["missing_at_test"], m["acc"], m["acc_std"]) for m in pt["missing"] if m["train_party_dropout"] == p_drop])
        if pts:
            axs[1].errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=3, capsize=2, color=col, label=f"train party-dropout {p_drop:g}")
    axs[1].set_xlabel("parties missing at inference (of 16)")
    axs[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_parties.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------ privacy: accuracy vs reconstruction
inv, noise = load("vb_inversion.json"), load("vb_noise.json")
if inv and par:
    acc = {}
    lookup = {(p["d"], p["bits"]): p for p in par}
    mapping = {"float32 (full 1536-d)": (1536, 32), "int8": (1536, 8), "int4": (1536, 4), "1-bit": (1536, 1), "PCA-256 float32": (256, 32),
               "PCA-64 float32": (64, 32), "PCA-16 float32": (16, 32), "PCA-64 int8": (64, 8), "PCA-16 int4": (16, 4)}
    for k, v in mapping.items():
        if v in lookup:
            acc[k] = lookup[v]
    if noise:
        for k, r in (("noise 0.3 x std", 0.3), ("noise 1 x std", 1.0), ("noise 3 x std", 3.0)):
            for n in noise:
                if abs(n["noise_ratio"] - r) < 1e-9:
                    acc[k] = n
    rows, fig, ax = [], *plt.subplots(figsize=(4.6, 3.1))
    for name, c in inv["conditions"].items():
        if name in acc:
            a = acc[name]
            ax.scatter(c["psnr"], a["acc"], s=18, color=C["red"] if "noise" in name else C["blue"])
            ax.annotate(name.replace(" float32", "").replace("float32 (full 1536-d)", "full"), (c["psnr"], a["acc"]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
            rows.append(f"{name} & {a['acc']:.2f} & {c['psnr']:.2f} & {c['ssim']:.3f} \\\\")
    ax.axvline(inv["mean_image"]["psnr"], color=C["grey"], ls="--", lw=0.8)
    ax.text(inv["mean_image"]["psnr"], ax.get_ylim()[0], " mean image", fontsize=6, color=C["grey"], va="bottom")
    ax.set_xlabel("reconstruction PSNR of the inversion attack (dB, lower = more private)")
    ax.set_ylabel("VFL test accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_privacy.pdf"))
    plt.close(fig)
    rows.append(f"\\midrule mean training image & -- & {inv['mean_image']['psnr']:.2f} & {inv['mean_image']['ssim']:.3f} \\\\")
    rows.append(f"nearest neighbour in gallery & -- & {inv['nearest_neighbour']['psnr']:.2f} & {inv['nearest_neighbour']['ssim']:.3f} \\\\")
    write("tab_privacy.tex", "\n".join(rows))

npz = os.path.join(RES, "vb_inversion_examples.npz")
if os.path.exists(npz):
    z = np.load(npz)
    keys = [k for k in z.files]
    order = ["original", "nearest_neighbour", "float32_full_1536-d", "int8", "1-bit", "PCA-64_float32", "PCA-16_float32", "noise_0.3_x_std", "noise_1_x_std", "noise_3_x_std"]
    keys = [k for k in order if k in keys]
    fig, axs = plt.subplots(len(keys), 8, figsize=(6.6, 0.82 * len(keys)))
    for i, k in enumerate(keys):
        for j in range(8):
            axs[i, j].imshow(np.transpose(z[k][j], (1, 2, 0)))
            axs[i, j].axis("off")
        axs[i, 0].text(-0.1, 0.5, k.replace("_", " "), transform=axs[i, 0].transAxes, ha="right", va="center", fontsize=6)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.995, bottom=0.005, wspace=0.03, hspace=0.05)
    fig.savefig(os.path.join(OUT, "fig_inversion.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------ baselines table
base = {k: load(f"baseline_{k}.json") for k in ["splitnn_rgb", "splitnn_all", "ssl_rgb", "ssl_all"]}
if any(base.values()):
    rows = []
    for k, label in (("splitnn_rgb", "SplitNN, ResNet18, 3 bands"), ("splitnn_all", "SplitNN, ResNet18, 13 bands"), ("ssl_rgb", "FedOnce-style SimCLR, 3 bands"), ("ssl_all", "FedOnce-style SimCLR, 13 bands")):
        b = base[k]
        if b:
            secs = b.get("secs_per_seed", [None])[0] or b.get("ssl_pretrain_secs_all_parties")
            rows.append(f"{label} & {pm(b['summary'])} & {pm(b['summary'], 'auc')} & {secs / 60:.0f} \\\\")
    write("tab_baselines.tex", "\n".join(rows))

# ------------------------------------------------------------------ WorldStrat
ws = load("ws_results.json")
if ws:
    order = [("location_rf", "Location only (RF on lon/lat)"), ("spot", "SPOT"), ("spot_wide2000", "SPOT, wider head (capacity control)"), ("sent1", "Sentinel, 1 random revisit"),
             ("sent16_concat", "Sentinel, 16 revisits (one-shot)"), ("spot_sent_mean2", "SPOT + mean Sentinel (one-shot)"), ("spot_sent16_concat", "SPOT + 16 Sentinel (one-shot)")]
    for label in ("ipcc", "simplified_ipcc", "sources"):
        if label not in ws:
            continue
        rows = []
        for key, name in order:
            cells = []
            for reg in ("random", "spatial"):
                s = ws[label].get(reg, {}).get(key)
                cells += [pm(s), pm(s, "auc")] if s else ["--", "--"]
            rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        write(f"tab_ws_{label}.tex", "\n".join(rows))
    # figure: random vs spatial AUC for ipcc
    if "ipcc" in ws:
        fig, ax = plt.subplots(figsize=(4.8, 2.9))
        keys = [k for k, _ in order]
        names = [n.split(" (")[0] for _, n in order]
        x = np.arange(len(keys))
        for off, reg, col in ((-0.18, "random", C["blue"]), (0.18, "spatial", C["orange"])):
            v = [ws["ipcc"][reg][k]["auc"] if k in ws["ipcc"][reg] else np.nan for k in keys]
            e = [ws["ipcc"][reg][k]["auc_std"] if k in ws["ipcc"][reg] else 0 for k in keys]
            ax.bar(x + off, v, 0.34, yerr=e, color=col, label=f"{reg} split", capsize=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("macro AUC (%), IPCC 7 classes")
        ax.set_ylim(50, 100)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_ws_split.pdf"))
        plt.close(fig)
print("done")
