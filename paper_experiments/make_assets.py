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


def fmin(m):
    return f"{m:.1f}" if m < 10 else f"{m:.0f}"


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
    refs = [(81.17, "SplitNN, published (13 bands)")]
    sn = load("baseline_splitnn_all.json")
    if sn:
        refs.append((sn["summary"]["acc"], "SplitNN, ours (13 bands)"))
    for yv, lab in refs:
        ax.axhline(yv, color=C["grey"], ls="--" if "published" in lab else ":", lw=0.8)
        ax.text(ax.get_xlim()[0] if False else 40, yv + 0.12, lab, fontsize=6, color=C["grey"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("one-time upload per sample, all 16 parties (bytes)")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(title="precision", ncol=3, frameon=False, loc="lower right", fontsize=6, title_fontsize=6)
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

# ------------------------------------------------------------------ Gaussian-DP reading of the noise runs
def gdp_eps(mu, delta=1e-5):
    """(eps, delta) of a mu-GDP mechanism (Dong et al.), solved in log space (no overflow for large mu)."""
    from scipy.special import log_ndtr, ndtr

    f = lambda e: ndtr(-e / mu + mu / 2) - np.exp(e + log_ndtr(-e / mu - mu / 2)) - delta  # noqa: E731
    lo, hi = 0.0, mu * mu + 20 * mu + 50
    for _ in range(300):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) > 0 else (lo, m)
    return hi


noise = load("vb_noise.json")
if noise:
    rows = []
    for n in noise:
        if n["noise_ratio"] > 0:
            mu = 2 * n["C"] / n["sigma_per_coord"]
            e = gdp_eps(mu)
            ex = int(np.floor(np.log10(e)))
            es = f"{e:.0f}" if e < 1000 else f"${e / 10**ex:.1f}\\times10^{{{ex}}}$"
            rows.append(f"{n['noise_ratio']:g} & {pm(n)} & {pm(n, 'auc')} & {mu:.1f} & {es} \\\\")
    write("tab_noise.tex", "\n".join(rows))
    # noise multiplier (in units of the mean per-coordinate std) needed for eps = 1 and eps = 8 at delta = 1e-5
    C0, sd0 = noise[1]["C"], noise[1]["sigma_per_coord"] / noise[1]["noise_ratio"]
    need = {}
    for target in (1, 8):
        lo, hi = 1e-3, 100.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gdp_eps(mid) < target else (lo, mid)
        need[target] = 2 * C0 / lo / sd0
    write("noise.tex", f"\\newcommand{{\\noiseForEpsOne}}{{{need[1]:.0f}}}\\newcommand{{\\noiseForEpsEight}}{{{need[8]:.0f}}}\\newcommand{{\\clipC}}{{{C0:.1f}}}\n")

# ------------------------------------------------------------------ privacy: accuracy vs reconstruction
inv = load("vb_inversion.json")
if inv and par:
    lookup = {(p["d"], p["bits"]): p for p in par}
    mapping = {"float32 (full 1536-d)": (1536, 32), "int8": (1536, 8), "int4": (1536, 4), "1-bit": (1536, 1), "PCA-256 float32": (256, 32),
               "PCA-64 float32": (64, 32), "PCA-16 float32": (16, 32), "PCA-64 int8": (64, 8), "PCA-16 int4": (16, 4)}
    pretty = {"float32 (full 1536-d)": "full, float32", "int8": "full, 8-bit", "int4": "full, 4-bit", "1-bit": "full, 1-bit",
              "PCA-256 float32": "PCA-256, float32", "PCA-64 float32": "PCA-64, float32", "PCA-16 float32": "PCA-16, float32",
              "PCA-64 int8": "PCA-64, 8-bit", "PCA-16 int4": "PCA-16, 4-bit"}
    size = lambda b: f"{b / 1024:.0f}\\,KiB" if b >= 1024 else f"{b:.0f}\\,B"  # noqa: E731
    rows = []
    for name, c in inv["conditions"].items():
        if name in mapping and mapping[name] in lookup:
            a_ = lookup[mapping[name]]
            rows.append(f"{pretty[name]} & {size(a_['bytes_per_sample'])} & {pm(a_)} & {c['psnr']:.1f} & {c['ssim']:.3f} & -- \\\\")
    rows.append("\\midrule")
    for name, c in inv["conditions"].items():
        if name.startswith("noise") and noise:
            r = float(name.split()[1])
            n = next(n for n in noise if abs(n["noise_ratio"] - r) < 1e-9)
            mu = 2 * n["C"] / n["sigma_per_coord"]
            e = gdp_eps(mu)
            ex = int(np.floor(np.log10(e)))
            es = f"{e:.0f}" if e < 1000 else f"${e / 10**ex:.1f}\\times10^{{{ex}}}$"
            rows.append(f"full + noise, $\\rho={r:g}$ & 96\\,KiB & {pm(n)} & {c['psnr']:.1f} & {c['ssim']:.3f} & {es} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\emph{{reference:}} mean training image & -- & -- & {inv['mean_image']['psnr']:.1f} & {inv['mean_image']['ssim']:.3f} & -- \\\\")
    rows.append(f"\\emph{{reference:}} nearest training neighbour & -- & -- & {inv['nearest_neighbour']['psnr']:.1f} & {inv['nearest_neighbour']['ssim']:.3f} & -- \\\\")
    write("tab_privacy.tex", "\n".join(rows))

npz = os.path.join(RES, "vb_inversion_examples.npz")
if os.path.exists(npz):
    z = np.load(npz)
    keys = [k for k in z.files]
    order = ["original", "nearest_neighbour", "float32_full_1536-d", "1-bit", "PCA-64_float32", "PCA-16_float32", "noise_3_x_std"]
    label = {"original": "original", "nearest_neighbour": "nearest neighbour", "float32_full_1536-d": "full, float32", "1-bit": "full, 1-bit",
             "PCA-64_float32": "PCA-64", "PCA-16_float32": "PCA-16", "noise_3_x_std": "noise, $\\rho=3$"}
    GAIN = 3.0  # display only: the 8-bit reflectances are dark; same gain for every row
    keys = [k for k in order if k in keys]
    ex = [0, 3, 4, 6]  # four revisits (parties) of the first test location, with different illumination
    fig, axs = plt.subplots(len(ex), len(keys), figsize=(6.6, 0.98 * len(ex)))
    for j, k in enumerate(keys):
        for i, e in enumerate(ex):
            axs[i, j].imshow(np.clip(np.transpose(z[k][e], (1, 2, 0)).astype(float) * GAIN / 255, 0, 1))
            axs[i, j].axis("off")
        axs[0, j].set_title(label[k], fontsize=7)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(OUT, "fig_inversion.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------ main comparison table (VertiBench)
base = {k: load(f"baseline_{k}.json") for k in ["splitnn_all", "splitnn_rgb", "ssl_all", "ssl_rgb"]}
tim = load("vb_timing_A100.json") or load("vb_timing_L4.json")  # same GPU as the baselines when available
tim_gpu = (tim or {}).get("gpu", "")
if bb or any(base.values()):
    MiB = 1024 * 1024
    grp = lambda t: f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{t}}}}} \\\\"  # noqa: E731
    rows = [grp("Iterative: split learning, end-to-end"), "SplitNN, published~\\cite{wu2024vertibench} & 13 & -- & $81.17$\\,{\\scriptsize$\\pm$0.35} & -- & -- \\\\"]
    rep = load("vb_repro_splitresnet.json")
    if rep:
        ep = len(rep["curves"][0])
        rows.append(f"SplitResNet, VertiBench recipe (ours) & 13 & {2 * 16 * 512 * 4 * ep / MiB:.1f}\\,MiB ({ep} ep.) & {pm(rep['last_epoch'])} & {pm(rep['last_epoch'], 'auc')} & {fmin(np.mean(rep['secs_per_seed']) / 60)} \\\\")
    for k, bands in (("splitnn_all", 13), ("splitnn_rgb", 3)):
        b = base[k]
        if b:
            ep = b["epochs"]
            comm = 2 * 16 * 512 * 4 * ep / MiB  # activations forward + gradients backward, float32, every epoch
            gpu = fmin(np.mean(b["secs_per_seed"]) / 60)
            rows.append(f"SplitNN, ResNet-18 & {bands} & {comm:.1f}\\,MiB ({ep} ep.) & {pm(b['summary'])} & {pm(b['summary'], 'auc')} & {gpu} \\\\")
    rows += ["\\midrule", grp("One-shot: self-supervised encoder trained by each party (FedOnce-style)")]
    for k, bands in (("ssl_all", 13), ("ssl_rgb", 3)):
        b = base[k]
        if b:
            gpu = fmin(b["ssl_pretrain_secs_all_parties"] / 60)
            rows.append(f"SimCLR, ResNet-18 & {bands} & {16 * 512 * 4 / 1024:.0f}\\,KiB & {pm(b['summary'])} & {pm(b['summary'], 'auc')} & {gpu} \\\\")
    if bb:
        rows += ["\\midrule", grp("One-shot: frozen foundation encoder, no party-side training (this work)")]
        for k in ("clip_l14", "dinov2_g", "dinov2_b", "dinov2_s"):
            if k in bb:
                r = bb[k]
                gpu = fmin(tim[k]["gpu_min_all_parties"]) if tim and k in tim else "--"
                rows.append(f"{NAMES[k]} & 3 & {16 * r['dim'] * 4 / 1024:.0f}\\,KiB & {pm(r['vertical_mlp2'])} & {pm(r['vertical_mlp2'], 'auc')} & {gpu} \\\\")
    write("tab_main.tex", "\n".join(rows))
    if tim:
        write("timing.tex", "".join(f"\\newcommand{{\\gpu{k.replace('dinov2_', '').upper()}}}{{{tim[k]['gpu_min_all_parties']:.1f}}}" for k in ("dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g") if k in tim)
              + "".join(f"\\newcommand{{\\headSecs}}{{{t['head_train_secs_dinov2_g_mlp2']:.0f}}}" for t in [load("vb_timing_L4.json") or tim][:1] if "head_train_secs_dinov2_g_mlp2" in t) + "\n")

# ------------------------------------------------------------------ why CLIP: encoder ablations x geography
enc, geo = load("vb_encoders.json"), load("vb_geo.json")
if enc:
    label7 = {"dinov2_g_168": "DINOv2 ViT-g/14, 168\\,px, CLS", "dinov2_g_224": "DINOv2 ViT-g/14, 224\\,px, CLS",
              "dinov2_g_168_clsmean": "DINOv2 ViT-g/14, 168\\,px, CLS + mean patch", "clip_l14_224": "CLIP ViT-L/14, 224\\,px",
              "clip_l14_168": "CLIP ViT-L/14, 168\\,px", "clip_b16_224": "CLIP ViT-B/16, 224\\,px", "siglip_b16_224": "SigLIP ViT-B/16, 224\\,px",
              "dinov2_s_168": "DINOv2 ViT-S/14, 168\\,px", "dinov2_b_random_168": "ViT-B/14, random weights"}
    rows = []
    for k, name in label7.items():
        if k not in enc:
            continue
        cells = [name, pm(enc[k]["mlp2"])]
        if geo and "nn_geolocation" in geo and k in geo["nn_geolocation"]:
            g = geo["nn_geolocation"][k]
            cells += [f"{g['median_km']:.0f}", f"{g['within_100km']:.0f}"]
        else:
            cells += ["--", "--"]
        for reg in ("random", "spatial"):
            c = (geo or {}).get("cv", {}).get(reg, {}).get(k)
            cells.append(pm(c) if c else "--")
        rows.append(" & ".join(cells) + " \\\\")
    if geo and "official_location_rf" in geo:
        r = geo["official_location_rf"]
        cells = ["\\emph{location only} (RF on lon/lat)", f"{r['acc']:.2f}", "--", "--"]
        for reg in ("random", "spatial"):
            c = geo.get("cv", {}).get(reg, {}).get("location_rf")
            cells.append(pm(c) if c else "--")
        rows += ["\\midrule", " & ".join(cells) + " \\\\"]
    write("tab_encoders.tex", "\n".join(rows))
    if geo:
        mr = geo.get("cv", {}).get("majority_rate", {})
        write("geo.tex", "".join(f"\\newcommand{{\\geo{k.replace('_', '')}}}{{{v}}}" for k, v in
              (("nmatched", geo["n_matched"]), ("nconfident", geo["n_confident"]), ("nsamples", geo["n_samples"]),
               ("align", f"{geo['revisit_alignment_mean'] * 100:.0f}"), ("pois", geo["distinct_pois"]),
               ("majrandom", f"{np.mean(mr.get('random', [0])):.1f}"), ("majspatial", f"{np.mean(mr.get('spatial', [0])):.1f}"))) + "\n")

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
            for reg in ("random_cv" if "random_cv" in ws[label] else "random", "spatial"):
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
