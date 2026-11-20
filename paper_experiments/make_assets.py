"""Turn the result JSONs (paper_results/) into LaTeX tables, macros and figures (paper/src/gen/).

usage: python make_assets.py [results_dir] [out_dir]
Every number of the paper comes from these files, so the text cannot drift from the logs.
Missing inputs are skipped, so the script can be run on partial results.
Spreads are sample standard deviations (ddof=1) recomputed from the per-seed / per-fold values.
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
KiB, MiB = 1024, 1024 * 1024
MACROS = {}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def sd(s, key):
    vals = s.get(key + "s")
    if vals is not None and len(vals) > 1:
        return float(np.std(vals, ddof=1))
    return s.get(key + "_std", 0.0)


def pm(s, key="acc", d=1):
    return f"{s[key]:.{d}f}\\,{{\\scriptsize$\\pm${sd(s, key):.{d}f}}}"


def size(b):
    return f"{b / MiB:.2f}\\,MiB" if b >= MiB else (f"{b / KiB:.0f}\\,KiB" if b >= KiB else f"{b:.0f}\\,B")


def fmin(m):
    return f"{m:.1f}" if m < 10 else f"{m:.0f}"


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)
    print("wrote", name)


def macro(name, value):
    MACROS[name] = value


NAMES = {
    "dinov2_g": "DINOv2 ViT-g/14", "dinov2_l": "DINOv2 ViT-L/14", "dinov2_b": "DINOv2 ViT-B/14", "dinov2_s": "DINOv2 ViT-S/14",
    "clip_l14": "CLIP ViT-L/14", "resnet50_in1k": "ResNet-50 (ImageNet-1k, supervised)", "dinov2_b_random": "ViT-B/14, random weights",
    "dinov2_g_4comp": "DINOv2 ViT-g/14, 4 composites (6 bands)", "ssl4eo_vits16": "SSL4EO-S12 ViT-S/16 (13 bands)",
}
bb, enc, geo, ctl = load("vb_backbones.json"), load("vb_encoders.json"), load("vb_geo.json"), load("vb_controls.json")
tim = load("vb_timing_A100.json") or load("vb_timing_L4.json")
timL4 = load("vb_timing_L4.json")
inv, rep, ws = load("vb_inversion.json"), load("vb_repro_splitresnet.json"), load("ws_results.json")
base = {k: load(f"baseline_{k}.json") for k in ["splitnn_all", "splitnn_rgb", "splitnn_rgb_pt", "ssl_all", "ssl_rgb"]}
E = (ctl or {})

# ------------------------------------------------------------------ main comparison table (VertiBench, official split)
if bb or enc or any(base.values()):
    grp = lambda t: f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{t}}}}} \\\\"  # noqa: E731
    split_comm = lambda ep: 2 * 16 * 512 * 2 * ep  # noqa: E731  activations + gradients, half precision, every epoch
    rows = [grp("Iterative: split learning, end-to-end"), "SplitNN, published~\\cite{wu2024vertibench} & 13 & -- & 81.2\\,{\\scriptsize$\\pm$0.4} & -- & -- \\\\"]
    if rep:
        ep = len(rep["curves"][0])
        rows.append(f"SplitNN, VertiBench recipe (ours) & 13 & {size(split_comm(ep))} & {pm(rep['last_epoch'])} & {pm(rep['last_epoch'], 'auc')} & {fmin(np.mean(rep['secs_per_seed']) / 60)} \\\\")
        macro("reproAcc", f"{rep['last_epoch']['acc']:.1f}")
        macro("reproStd", f"{sd(rep['last_epoch'], 'acc'):.1f}")
        macro("reproBest", f"{rep['best_epoch_on_test']['acc']:.1f}")
        macro("reproMin", fmin(np.mean(rep["secs_per_seed"]) / 60))
    for k, bands, lab in (("splitnn_all", 13, "SplitNN, our recipe, scratch"), ("splitnn_rgb", 3, "SplitNN, our recipe, scratch"), ("splitnn_rgb_pt", 3, "SplitNN, our recipe, ImageNet init.")):
        b = base[k]
        if b:
            rows.append(f"{lab} & {bands} & {size(split_comm(b['epochs']))} & {pm(b['summary'])} & {pm(b['summary'], 'auc')} & {fmin(np.mean(b['secs_per_seed']) / 60)} \\\\")
    rows += ["\\midrule", grp("One-shot: self-supervised encoder trained by each party (SimCLR variant of FedOnce)")]
    for k, bands in (("ssl_all", 13), ("ssl_rgb", 3)):
        b = base[k]
        if b:
            rows.append(f"SimCLR ResNet-18, scratch & {bands} & {size(16 * 512 * 2)} & {pm(b['summary'])} & {pm(b['summary'], 'auc')} & {fmin(b['ssl_pretrain_secs_all_parties'] / 60)} \\\\")
    rows += ["\\midrule", grp("One-shot: frozen public encoder, no party-side training (this work)")]

    def frozen_time(key):
        if tim and key in tim:
            t = tim[key]
            return fmin(t["gpu_min_all_parties"] * (1 + t["io_ms_per_image"] / t["gpu_ms_per_image"]))
        return "--"

    frozen = []
    if enc:
        frozen += [("CLIP ViT-L/14, 224\\,px", enc.get("clip_l14_224"), 1024, "clip_l14"), ("DINOv2 ViT-g/14, 224\\,px", enc.get("dinov2_g_224"), 1536, None),
                   ("DINOv2 ViT-g/14, 168\\,px", enc.get("dinov2_g_168"), 1536, "dinov2_g"), ("DINOv2 ViT-S/14, 168\\,px", enc.get("dinov2_s_168"), 384, "dinov2_s")]
    e6 = E.get("E6_ssl4eo", {})
    if e6 and "mlp2" in e6:
        frozen.append(("SSL4EO-S12 ViT-S/16, 13 bands", e6, e6["dim"], None))
    for lab, r, d, tk in frozen:
        if r:
            s = r["mlp2"]
            bands = 13 if "13 bands" in lab else 3
            rows.append(f"{lab} & {bands} & {size(16 * d * 2)} & {pm(s)} & {pm(s, 'auc')} & {frozen_time(tk) if tk else '--'} \\\\")
    for k, m in (("splitnn_all", "splitAll"), ("splitnn_rgb", "splitRGB"), ("splitnn_rgb_pt", "splitPT"), ("ssl_all", "sslAll"), ("ssl_rgb", "sslRGB")):
        b = base[k]
        if b:
            macro(m + "Acc", f"{b['summary']['acc']:.1f}")
            macro(m + "Min", fmin((np.mean(b["secs_per_seed"]) if "secs_per_seed" in b else b["ssl_pretrain_secs_all_parties"]) / 60))
    for k, m in (("dinov2_g", "timeG"), ("clip_l14", "timeClip"), ("dinov2_s", "timeS")):
        if tim and k in tim:
            macro(m, frozen_time(k))
    rows += ["\\midrule", grp("References")]
    e5 = E.get("E5_paired", {})
    if e5:
        rows.append(f"majority class & -- & -- & {e5['majority_acc']:.1f} & 50.0 & -- \\\\")
        macro("majVB", f"{e5['majority_acc']:.1f}")
    if geo and "official_location_rf" in geo:
        r = geo["official_location_rf"]
        rows.append(f"location only (RF on recovered lon/lat) & -- & -- & {r['acc']:.1f} & {r['auc']:.1f} & -- \\\\")
        macro("locRFofficial", f"{r['acc']:.1f}")
    write("tab_main.tex", "\n".join(rows))

# ------------------------------------------------------------------ encoders x geography (why CLIP)
if enc:
    label7 = [("dinov2_g_168", "DINOv2 ViT-g/14, 168\\,px"), ("dinov2_g_224", "DINOv2 ViT-g/14, 224\\,px"), ("dinov2_g_168_clsmean", "DINOv2 ViT-g/14, 168\\,px, CLS+mean"),
              ("clip_l14_224", "CLIP ViT-L/14, 224\\,px"), ("clip_l14_168", "CLIP ViT-L/14, 168\\,px"), ("clip_b16_224", "CLIP ViT-B/16, 224\\,px"),
              ("siglip_b16_224", "SigLIP ViT-B/16, 224\\,px"), ("dinov2_s_168", "DINOv2 ViT-S/14, 168\\,px"), ("dinov2_b_random_168", "ViT-B/14, random weights")]
    rows = []
    for k, name in label7:
        if k not in enc:
            continue
        cells = [name, pm(enc[k]["mlp2"])]
        g = (geo or {}).get("nn_geolocation", {}).get(k)
        cells += [f"{g['median_km']:.0f}", f"{g['within_100km']:.0f}"] if g else ["--", "--"]
        for reg in ("random", "spatial"):
            c = (geo or {}).get("cv", {}).get(reg, {}).get(k)
            cells += [pm(c), pm(c, "auc")] if c else ["--", "--"]
        rows.append(" & ".join(cells) + " \\\\")
    if geo and "cv" in geo:
        rows.append("\\midrule")
        for k, name in (("location_rf", "location only (RF on lon/lat)"), ("train_majority", "majority class of the training folds")):
            cells = [name, f"{geo['official_location_rf']['acc']:.1f}" if k == "location_rf" and "official_location_rf" in geo else "--", "--", "--"]
            for reg in ("random", "spatial"):
                c = geo["cv"].get(reg, {}).get(k)
                cells += [pm(c), pm(c, "auc") if k == "location_rf" else "50.0"] if c else ["--", "--"]
            rows.append(" & ".join(cells) + " \\\\")
    write("tab_encoders.tex", "\n".join(rows))
    if geo:
        macro("geoagree", f"{geo.get('provider_agreement', float('nan')) * 100:.1f}")
        macro("geonconfident", f"{geo['n_confident']:,}".replace(",", "{,}"))
        macro("geonsamples", f"{geo['n_samples']:,}".replace(",", "{,}"))
        macro("geoalign", f"{geo['revisit_alignment_mean'] * 100:.0f}")
        # rank correlation between geolocation ability and accuracy (official split)
        ks = [k for k, _ in label7 if k in enc and k in geo.get("nn_geolocation", {})]
        if len(ks) > 3:
            from scipy.stats import spearmanr

            rho, p = spearmanr([enc[k]["mlp2"]["acc"] for k in ks], [-geo["nn_geolocation"][k]["median_km"] for k in ks])
            macro("geoSpearman", f"{rho:.2f}")
            macro("geoSpearmanP", f"{p:.1g}" if p >= 0.001 else "<0.001")
            macro("geoSpearmanN", str(len(ks)))

# ------------------------------------------------------------------ backbones table (official split, stage 2)
if bb:
    rows = []
    for k in ["dinov2_g", "dinov2_l", "dinov2_b", "dinov2_s", "clip_l14", "resnet50_in1k", "dinov2_b_random", "dinov2_g_4comp"]:
        if k in bb:
            r = bb[k]
            rows.append(f"{NAMES[k]} & {r['dim']} & {pm(r['vertical_mlp2'])} & {pm(r['vertical_mlp2'], 'auc')} & {pm(r['vertical_linear'])} & {pm(r['horizontal_1view_mlp2'])} \\\\")
    write("tab_backbones.tex", "\n".join(rows))

# ------------------------------------------------------------------ Pareto (validation-selected operating points)
par = E.get("E7_pareto_val") if isinstance(E.get("E7_pareto_val"), list) else None
if par:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    cols = {16: C["blue"], 8: C["green"], 4: C["orange"], 2: C["red"], 1: C["grey"]}
    for bits in [16, 8, 4, 2, 1]:
        pts = sorted([(p["bytes_per_sample"], p["acc"], sd(p, "acc"), p["d"]) for p in par if p["bits"] == bits])
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=3, lw=1, capsize=1.5, color=cols[bits], label=f"{bits}-bit" + (" float" if bits == 16 else ""))
            if bits == 16:
                for p in pts:
                    ax.annotate(f"{p[3]}", (p[0], p[1]), fontsize=5, xytext=(0, 4), textcoords="offset points", ha="center", color=C["blue"])
    refs = [(81.17, "SplitNN, published (13 bands)", "--")]
    if rep:
        refs.append((rep["last_epoch"]["acc"], "SplitNN, VertiBench recipe (ours)", ":"))
    for yv, lab, ls in refs:
        ax.axhline(yv, color=C["grey"], ls=ls, lw=0.8)
        ax.text(1.2 * min(p["bytes_per_sample"] for p in par), yv + 0.12, lab, fontsize=5.5, color=C["grey"])
    # best validation accuracy under each byte budget -> its test accuracy
    budgets = [256, 512, 2048, 49152]
    sel = {}
    for B in budgets:
        cand = [p for p in par if p["bytes_per_sample"] <= B]
        best = max(cand, key=lambda p: p["val_acc"])
        sel[B] = best
        ax.scatter([best["bytes_per_sample"]], [best["acc"]], s=40, facecolors="none", edgecolors="k", lw=0.8, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048, 8192, 49152])
    ax.set_xticklabels(["32 B", "128 B", "512 B", "2 KiB", "8 KiB", "48 KiB"])
    ax.set_xlabel("one-time upload per sample, all 16 parties")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(title="precision", ncol=3, frameon=False, loc="lower right", fontsize=6, title_fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
    plt.close(fig)
    full = next(p for p in par if p["d"] == 1536 and p["bits"] == 16)
    macro("fullAcc", f"{full['acc']:.1f}")
    for B, tag in zip(budgets, ("Two", "Five", "Twok", "Full")):
        p = sel[B]
        macro(f"sel{tag}Acc", f"{p['acc']:.1f}")
        macro(f"sel{tag}Cfg", f"PCA-{p['d']}, {p['bits']}-bit" if p["d"] < 1536 else f"{p['bits']}-bit")
        macro(f"sel{tag}Bytes", size(p["bytes_per_sample"]))
        macro(f"sel{tag}Wins", str(int(sum(a > b for a, b in zip(p["accs"], full["accs"])))))
    rows = []
    for p in sorted(par, key=lambda p: (-p["d"], -p["bits"])):
        rows.append(f"{p['d']} & {p['bits']} & {size(p['bytes_per_sample'])} & {p['val_acc']:.1f} & {pm(p)} & {pm(p, 'auc')} \\\\")
    write("tab_pareto.tex", "\n".join(rows))

# ------------------------------------------------------------------ regularisation controls (is the PCA gain a regularisation effect?)
e1 = E.get("E1_regularisation")
if isinstance(e1, dict) and "full_wd0" in e1:
    lab = {"full_wd0": "full 1536-d, no regularisation", "full_wd0.0001": "full, weight decay $10^{-4}$", "full_wd0.001": "full, weight decay $10^{-3}$",
           "full_wd0.01": "full, weight decay $10^{-2}$", "full_dropout0.5": "full, dropout 0.5", "full_dropout0.5_wd1e-3": "full, dropout 0.5 + wd $10^{-3}$",
           "pca1536_rotation": "per-party PCA rotation, no truncation", "randproj256": "random projection to 256-d", "randproj64": "random projection to 64-d",
           "pca256": "PCA-256", "pca256_wd1e-3": "PCA-256, wd $10^{-3}$", "pca64": "PCA-64", "pca64_wd1e-3": "PCA-64, wd $10^{-3}$"}
    rows = [f"{lab[k]} & {e1[k]['val_acc']:.1f} & {pm(e1[k])} & {pm(e1[k], 'auc')} \\\\" for k in lab if k in e1]
    write("tab_regularisation.tex", "\n".join(rows))
    best_full = max((k for k in e1 if k.startswith("full")), key=lambda k: e1[k]["val_acc"])
    macro("regBestFull", lab[best_full])
    macro("regBestFullAcc", f"{e1[best_full]['acc']:.1f}")
    for k, m in (("pca256", "regPCA"), ("pca1536_rotation", "regRot"), ("randproj256", "regRP"), ("full_wd0", "regNone")):
        if k in e1:
            macro(m + "Acc", f"{e1[k]['acc']:.1f}")

# ------------------------------------------------------------------ parties: number and missing parties (5 seeds, imputation)
e3, e2 = E.get("E3_parties"), E.get("E2_missing")
if isinstance(e3, dict) and isinstance(e2, list):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ks = [s["k"] for s in e3["subsets"]]
    axs[0].errorbar(ks, [s["acc"] for s in e3["subsets"]], yerr=[s["acc_std"] for s in e3["subsets"]], marker="o", ms=3, capsize=2, color=C["blue"])
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(ks)
    axs[0].set_xticklabels(ks)
    axs[0].set_xlabel("number of parties $k$ (random subsets)")
    axs[0].set_ylabel("test accuracy (%)")
    styles = {("0.0", "zero"): (C["grey"], "-", "no dropout, zeros"), ("0.0", "train_mean"): (C["grey"], "--", "no dropout, mean imputation"),
              ("0.25", "zero"): (C["orange"], "-", "dropout 0.25, zeros"), ("0.25", "train_mean"): (C["orange"], "--", "dropout 0.25, mean imputation"),
              ("0.5", "zero"): (C["green"], "-", "dropout 0.5, zeros")}
    for (pd_, imp), (col, ls, lab) in styles.items():
        pts = sorted([(m["missing_at_test"], m["acc"], sd(m, "acc")) for m in e2 if f"{m['train_party_dropout']:g}" == f"{float(pd_):g}" and (m["imputation"] == imp or m["missing_at_test"] == 0)])
        if pts:
            axs[1].errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=2.5, capsize=1.5, color=col, ls=ls, label=lab, lw=1)
    axs[1].set_xlabel("parties missing at inference (of 16)")
    axs[1].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_parties.pdf"))
    plt.close(fig)
    get = lambda p, m, imp: next(x for x in e2 if abs(x["train_party_dropout"] - p) < 1e-9 and x["missing_at_test"] == m and (x["imputation"] == imp or m == 0))  # noqa: E731
    for p, tp in ((0.0, "Zero"), (0.25, "Q"), (0.5, "H")):
        macro(f"miss{tp}None", f"{get(p, 0, 'zero')['acc']:.1f}")
        for m, tm in ((8, "Eight"), (12, "Twelve")):
            for imp, ti in (("zero", "Z"), ("zero_rescale", "R"), ("train_mean", "M")):
                macro(f"miss{tp}{tm}{ti}", f"{get(p, m, imp)['acc']:.1f}")
    for s in e3["subsets"]:
        macro(f"partiesK{ {1: 'One', 2: 'Two', 4: 'Four', 8: 'Eight', 16: 'All'}[s['k']] }".replace(" ", ""), f"{s['acc']:.1f}")
    solo = [s["acc"] for s in e3["solo"]]
    macro("soloMean", f"{np.mean(solo):.1f}")
    macro("soloMin", f"{min(solo):.1f}")
    macro("soloMax", f"{max(solo):.1f}")

# ------------------------------------------------------------------ privacy: leakage x accuracy of the same upload format
if inv and "single" in inv:
    accmap = {}
    if par:
        look = {(p["d"], p["bits"]): p for p in par}
        for name, key in (("full, 16-bit", (1536, 16)), ("full, 8-bit", (1536, 8)), ("full, 4-bit", (1536, 4)), ("full, 1-bit", (1536, 1)),
                          ("PCA-256", (256, 16)), ("PCA-64", (64, 16)), ("PCA-16", (16, 16)), ("PCA-64, 4-bit", (64, 4)), ("PCA-16, 4-bit", (16, 4))):
            if key in look:
                accmap[name] = (look[key], size(look[key]["bytes_per_sample"]))
    e4 = E.get("E4_dp")
    if isinstance(e4, list):
        for r in e4:
            if r["d"] == 1536 and r["eps_upload"]:
                accmap[f"DP eps={r['eps_upload']}"] = (r, size(16 * 1536 * 2))
    if enc and "clip_l14_224" in enc:
        accmap["CLIP-L, full, 16-bit"] = (enc["clip_l14_224"]["mlp2"], size(16 * 1024 * 2))
    noise_rows = {}
    rows = []
    order = ["full, 16-bit", "full, 8-bit", "full, 4-bit", "full, 1-bit", "PCA-256", "PCA-64", "PCA-16", "PCA-64, 4-bit", "PCA-16, 4-bit", "noise rho=1", "noise rho=3", "DP eps=64", "DP eps=8", "CLIP-L, full, 16-bit"]
    pretty = {"noise rho=1": "noise $\\rho=1$", "noise rho=3": "noise $\\rho=3$", "DP eps=64": "clip + noise, $\\varepsilon=64$", "DP eps=8": "clip + noise, $\\varepsilon=8$", "CLIP-L, full, 16-bit": "CLIP-L, full, 16-bit"}
    for name in order:
        r = inv["single"].get(name)
        if not r:
            continue
        a = accmap.get(name)
        j = inv.get("joint", {}).get(name)
        acc = pm(a[0]) if a else "--"
        up = a[1] if a else "--"
        ps = f"{r['psnr']:.1f}" + (f"\\,{{\\scriptsize$\\pm${np.std(r['psnr_seeds'], ddof=1):.1f}}}" if len(r.get("psnr_seeds", [])) > 1 else "")
        rows.append(f"{pretty.get(name, name)} & {up} & {acc} & {ps} & {r['ssim']:.2f} & {r['reid']['top1']:.1f} & {j['psnr']:.1f} \\\\" if j else
                    f"{pretty.get(name, name)} & {up} & {acc} & {ps} & {r['ssim']:.2f} & {r['reid']['top1']:.1f} & -- \\\\")
        if name in ("PCA-16", "DP eps=8", "noise rho=3", "full, 1-bit", "CLIP-L, full, 16-bit"):
            rows.append("\\midrule")
    rows.append("\\midrule")
    rows.append(f"\\emph{{reference:}} mean training image & -- & -- & {inv['mean_image']['psnr']:.1f} & {inv['mean_image']['ssim']:.2f} & 0.0 & -- \\\\")
    if "mean_colour_oracle" in inv:
        mo = inv["mean_colour_oracle"]
        rows.append(f"\\emph{{reference:}} each image's mean colour & -- & -- & {mo['psnr']:.1f} & {mo['ssim']:.2f} & {mo['reid']['top1']:.1f} & -- \\\\")
    rows.append(f"\\emph{{reference:}} nearest training image & -- & -- & {inv['nearest_neighbour']['psnr']:.1f} & {inv['nearest_neighbour']['ssim']:.2f} & -- & -- \\\\")
    rows = [r for i, r in enumerate(rows) if not (r == "\\midrule" and i + 1 < len(rows) and rows[i + 1] == "\\midrule")]
    write("tab_privacy.tex", "\n".join(rows))
    for name, tag in (("full, 16-bit", "Full"), ("PCA-64", "PCAsf"), ("PCA-16", "PCAst"), ("PCA-256", "PCAtf"), ("DP eps=8", "DPeight"), ("DP eps=64", "DPsf"),
                      ("full, 1-bit", "Onebit"), ("CLIP-L, full, 16-bit", "Clip"), ("noise rho=3", "Noisethree")):
        r = inv["single"].get(name)
        if r:
            macro(f"inv{tag}", f"{r['psnr']:.1f}")
            macro(f"inv{tag}Reid", f"{r['reid']['top1']:.1f}")
        j = inv.get("joint", {}).get(name)
        if j:
            macro(f"inv{tag}Joint", f"{j['psnr']:.1f}")
    macro("invMean", f"{inv['mean_image']['psnr']:.1f}")
    macro("invNN", f"{inv['nearest_neighbour']['psnr']:.1f}")
    if "mean_colour_oracle" in inv:
        macro("invColour", f"{inv['mean_colour_oracle']['psnr']:.1f}")
        macro("invColourReid", f"{inv['mean_colour_oracle']['reid']['top1']:.1f}")
    if isinstance(e4, list):
        rows = []
        for r in e4:
            eps = "--" if r["eps_upload"] is None else f"{r['eps_upload']:g}"
            mu = "--" if r["mu_upload"] is None else f"{r['mu_upload']:.2f}"
            rows.append(f"{'full 1536-d' if r['d'] == 1536 else 'PCA-' + str(r['d'])} & {eps} & {mu} & {pm(r)} & {pm(r, 'auc')} \\\\")
        write("tab_dp.tex", "\n".join(rows))
        for r in e4:
            tag = {1536: "Full", 64: "Sf", 16: "St"}[r["d"]] + {None: "None", 64: "Ssf", 8: "Eight", 1: "One"}[r["eps_upload"]]
            macro(f"dp{tag}", f"{r['acc']:.1f}")

# ------------------------------------------------------------------ inversion examples
npz = os.path.join(RES, "vb_inversion_examples.npz")
if os.path.exists(npz) and inv and "single" in inv:
    z = np.load(npz)
    cols = [("original", "original"), ("mean colour", None), ("nearest_neighbour", "nearest\ntraining image"), ("single__full_16-bit", "single upload,\nfull"),
            ("joint__full_16-bit", "16 uploads,\nfull"), ("single__PCA-64", "single upload,\nPCA-64"), ("single__DP_eps8", "single upload,\nclip+noise $\\varepsilon$=8"),
            ("single__CLIP-L_full_16-bit", "single upload,\nCLIP-L")]
    cols = [(k, t) for k, t in cols if k == "mean colour" or k in z.files]
    ex = [0, 1, 2, 3]
    GAIN = 3.0  # display only: the 8-bit reflectances are dark; same gain everywhere
    fig, axs = plt.subplots(len(ex), len(cols), figsize=(6.6, 0.95 * len(ex) + 0.35))
    for j, (k, title) in enumerate(cols):
        for i, e in enumerate(ex):
            img = z["original"][e] if k == "mean colour" else z[k][e]
            img = np.transpose(img, (1, 2, 0)).astype(float) / 255
            if k == "mean colour":
                img = np.broadcast_to(img.mean((0, 1), keepdims=True), img.shape)
            axs[i, j].imshow(np.clip(img * GAIN, 0, 1))
            axs[i, j].axis("off")
        axs[0, j].set_title(title or k, fontsize=6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(OUT, "fig_inversion.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------ paired tests
e5 = E.get("E5_paired")
if isinstance(e5, dict):
    for k, tag in (("clip_l14_224_vs_dinov2_g_168", "CvGs"), ("clip_l14_224_vs_dinov2_g_224", "CvGl"), ("dinov2_g_168_vs_dinov2_g_224", "GsvGl")):
        if k in e5:
            r = e5[k]
            macro(f"pair{tag}Diff", f"{r['diff']:+.1f}")
            macro(f"pair{tag}Lo", f"{r['ci95'][0]:+.1f}")
            macro(f"pair{tag}Hi", f"{r['ci95'][1]:+.1f}")
            macro(f"pair{tag}P", f"{r['mcnemar_p']:.2g}")

# ------------------------------------------------------------------ WorldStrat
if ws:
    order = [("location_rf", "Location only (RF on lon/lat)"), ("spot", "SPOT"), ("spot_wide2000", "SPOT, wider head"), ("sent1", "Sentinel, 1 random revisit"),
             ("sent16_concat", "Sentinel, 16 revisits"), ("spot_sent1", "SPOT + 1 Sentinel revisit"), ("spot_sent_mean2", "SPOT + mean Sentinel"), ("spot_sent16_concat", "SPOT + 16 Sentinel")]
    for label in ("ipcc", "simplified_ipcc", "sources"):
        if label not in ws:
            continue
        rnd = "random_cv" if "random_cv" in ws[label] else "random"
        rows = []
        maj = []
        for reg in (rnd, "spatial"):
            m = ws[label].get(reg, {}).get("train_majority_acc")
            maj.append(f"{m:.1f}" if m is not None else "--")
        rows.append(f"Majority class of the training set & {maj[0]} & 50.0 & {maj[1]} & 50.0 \\\\")
        for key, name in order:
            cells = []
            for reg in (rnd, "spatial"):
                s = ws[label].get(reg, {}).get(key)
                cells += [pm(s), pm(s, "auc")] if s else ["--", "--"]
            rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        write(f"tab_ws_{label}.tex", "\n".join(rows))
        for reg, t in ((rnd, "R"), ("spatial", "S")):
            d = ws[label].get(reg, {}).get("test_to_train_km")
            if d:
                macro(f"wsDist{t}{label.replace('_', '').replace('simplifiedipcc', 'simple')}", f"{np.median(d):.0f}")
    # spatial-split sensitivity to k
    rows = []
    for label in ("ipcc", "sources"):
        if label not in ws:
            continue
        for reg, name in (("spatial_k10", "$k=10$"), ("spatial", "$k=25$"), ("spatial_k50", "$k=50$"), ("random_cv", "random")):
            r = ws[label].get(reg)
            if not r:
                continue
            d = r.get("test_to_train_km")
            cells = [f"{label.replace('_', ' ')}", name, f"{np.median(d):.0f}" if d else "--"]
            for key in ("location_rf", "spot", "sent16_concat", "spot_sent_mean2"):
                cells.append(pm(r[key], "auc") if key in r else "--")
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    if rows:
        write("tab_ws_k.tex", "\n".join(rows[:-1]))

write("macros.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items()))
print("done")
