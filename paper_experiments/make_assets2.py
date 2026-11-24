"""Turn the result JSONs (paper_results/) into LaTeX tables, macros and figures (paper/src/gen/) - paper version 2.

usage: python make_assets2.py [results_dir] [out_dir]
Every number of the paper comes from these files.  Missing inputs are skipped (partial results are fine).
Spreads are sample standard deviations (ddof=1) recomputed from the per-seed / per-fold values.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    if s is None or key not in s:
        return "--"
    return f"{s[key]:.{d}f}\\,{{\\scriptsize$\\pm${sd(s, key):.{d}f}}}"


def size(b):
    return f"{b / MiB:.2f}\\,MiB" if b >= MiB else (f"{b / KiB:.0f}\\,KiB" if b >= KiB else f"{b:.0f}\\,B")


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)
    print("wrote", name)


DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def macro(name, value):
    """LaTeX macro names cannot contain digits or punctuation: spell digits out, drop the rest."""
    MACROS["".join(DIG.get(c, c) for c in name if c.isalnum())] = value


def f1(x, d=1):
    return f"{x:.{d}f}"


# ------------------------------------------------------------------ inputs
rep = load("vb_repro_splitresnet.json")
enc = load("vb224_encoders.json") or {}
heads224 = enc.get("heads", {})
cost = enc.get("cost", {})
Eg = load("vb224_controls_dinov2_g.json") or {}
if (load("vb224_controls_dinov2_g_dp.json") or {}).get("E4_dp"):  # third review: clipping norm per party
    Eg["E4_dp"] = load("vb224_controls_dinov2_g_dp.json")["E4_dp"]
Ec = load("vb224_controls_clip_l14.json") or {}
Eold = load("vb_controls.json") or {}
pool = load("vb224_pooling.json")
inv = load("vb224_inversion.json")
inv1 = inv or {}  # first attacker (15 epochs, Adam): kept as a third attacker, it generalises better on some formats
INV_NPZ = {}
_v2, _v2b = load("vb224_inversion_v2.json"), load("vb224_inversion_v2b.json")
if _v2:  # revised attacker (second review): shared deconvolutional decoder + linear decoder, per-location re-identification.
    # Per-party PCA bases differ between parties: the v2 rows that decoded them with a shared decoder are dropped and
    # replaced by the v2b re-run (per-party alignment / per-party linear map).
    for _part in ("single", "ridge"):
        for _k in list(_v2.get(_part, {})):
            if _k.startswith("PCA"):
                del _v2[_part][_k]
        for _k, _v in ((_v2b or {}).get(_part) or {}).items():
            _v2.setdefault(_part, {})[_k] = _v
    _dp = load("vb224_inversion_v2dp.json")  # third review: DP rows re-run with a clipping norm per party
    for _part in ("single", "ridge"):
        for _k, _v in ((_dp or {}).get(_part) or {}).items():
            _v2.setdefault(_part, {})[_k] = _v
            inv1.get(_part, {}).pop(_k, None)  # the first attacker ran on the old (pooled) clipping: drop its DP rows
    inv = _v2
    for _f, _pca in (("vb224_inversion_v2_examples.npz", False), ("vb224_inversion_v2b_examples.npz", True)):
        if os.path.exists(os.path.join(RES, _f)):
            _z = np.load(os.path.join(RES, _f))
            INV_NPZ.update({k: _z[k] for k in _z.files if ("PCA" in k) == _pca or k in ("original", "nearest_neighbour")})
    if _dp and os.path.exists(os.path.join(RES, "vb224_inversion_v2dp_examples.npz")):
        _z = np.load(os.path.join(RES, "vb224_inversion_v2dp_examples.npz"))
        INV_NPZ.update({k: _z[k] for k in _z.files if "DP" in k})
elif os.path.exists(os.path.join(RES, "vb224_inversion_examples.npz")):
    _z = np.load(os.path.join(RES, "vb224_inversion_examples.npz"))
    INV_NPZ = {k: _z[k] for k in _z.files}
he = {}
for _f in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
    if _f.startswith("he_heads") and _f.endswith(".json") and not (_f == "he_heads_ws.json" and os.path.exists(os.path.join(RES, "he_heads_ws_grouped.json"))):
        _d = load(_f)
        he.setdefault("he_cost", {}).update(_d.pop("he_cost", {}))
        for _k, _v in _d.items():  # deep merge: the he_heads_r_* re-runs only add heads (ridge_sel) to existing tasks
            if isinstance(_v, dict) and isinstance(he.get(_k), dict):
                for _c, _h in _v.items():
                    if isinstance(_h, dict) and isinstance(he[_k].get(_c), dict):
                        he[_k][_c].update(_h)
                    elif _h or _c not in he[_k]:
                        he[_k][_c] = _h
            else:
                he[_k] = _v
milk = load("milk_heads.json")
msplit = load("milk_split.json") or {}
msplit2 = load("milk_split_v2.json") or {}  # third review: frozen bottoms under the split-learning recipe, re-run of LoRA
aud = load("vb_audit.json") or {}  # kept probabilities, four-logit head, re-encoded linkage, nearest public location
ana = load("vb_audit_analysis.json") or {}  # stage 23: sibling subsets, paired tests against SplitNN
clipb = load("vb_split_clipb.json") or {}  # split learning with CLIP-B bottoms on VertiBench
cls = load("classic.json") or {}
ws = load("ws_sensors_grouped.json") or load("ws_sensors.json")  # third review: folds grouped by site
base = {k: load(f"baseline_{k}.json") for k in ["splitnn_all", "splitnn_rgb", "splitnn_rgb_pt", "ssl_all", "ssl_rgb", "nat_all"]}
K, N_TR = 16, 3142

# ------------------------------------------------------------------ Table: VertiBench main comparison
NAME = {"dinov2_g_224": "DINOv2 ViT-g/14", "dinov2_l_224": "DINOv2 ViT-L/14", "dinov2_b_224": "DINOv2 ViT-B/14", "dinov2_s_224": "DINOv2 ViT-S/14",
        "clip_l14_224": "CLIP ViT-L/14", "clip_b16_224": "CLIP ViT-B/16", "siglip_b16_224": "SigLIP ViT-B/16", "resnet50_in1k_224": "ResNet-50, ImageNet (supervised)",
        "dinov2_b_random_224": "ViT-B/14, random weights", "ssl4eo_s2_vits16": "SSL4EO-S12 ViT-S/16 (13 bands)"}
r18 = cost.get("vertibench_resnet18_bottom_158px", {}).get("flops_per_image")
rows = []


def row(label, bands, upload, gflops, minutes, s):
    g = "--" if gflops is None else (f"{gflops:.0f}" if gflops >= 10 else f"{gflops:.1f}")
    m = "--" if minutes is None else (f"{minutes:.1f}" if minutes < 10 else f"{minutes:.0f}")
    return f"{label} & {bands} & {upload} & {g} & {m} & {pm(s)} & {pm(s, 'f1')} & {pm(s, 'auc')} \\\\"


rows.append("\\multicolumn{8}{@{}l}{\\emph{Split learning (every epoch: activations up, gradients down)}}\\\\")
rows.append("SplitNN, published~\\cite{wu2024vertibench} & 13 & " + size(4 * K * 512 * 50) + " & -- & -- & 81.2\\,{\\scriptsize$\\pm$0.4} & -- & -- \\\\")
if rep:
    s = rep["last_epoch"]
    tr_gf = 3 * r18 * 50 / 1e9 if r18 else None  # per training image and party: 50 epochs x (forward + backward)
    rows.append(row("SplitNN, VertiBench recipe (ours)", 13, size(4 * K * 512 * 50), tr_gf, np.mean(rep["secs_per_seed"]) / 60, s))
    macro("reproAcc", f1(s["acc"]))
    macro("reproStd", f1(sd(s, "acc")))
    macro("reproBest", f1(rep["best_epoch_on_test"]["acc"]))
    macro("reproMin", f"{np.mean(rep['secs_per_seed']) / 60:.0f}")
for k, bands, lab in (("splitnn_all", 13, "SplitNN, our recipe, scratch"), ("splitnn_rgb", 3, "SplitNN, our recipe, scratch"), ("splitnn_rgb_pt", 3, "SplitNN, our recipe, ImageNet init.")):
    b = base[k]
    if b:
        rows.append(row(lab, bands, size(4 * K * 512 * b["epochs"]), None, np.mean(b["secs_per_seed"]) / 60, b["summary"]))
N_TR_H, N_VA_H, N_TE = int(N_TR * 0.85), N_TR - int(N_TR * 0.85), 785
for k, lab in (("clip_b16_lora", "CLIP ViT-B/16 bottoms, LoRA"), ("clip_b16_frozen", "CLIP ViT-B/16 bottoms, frozen")):
    s = clipb.get(k, {})
    if not s.get("summary"):
        continue
    E = s["infos"][0]["epochs"]
    tr_bytes = np.mean([i["bytes_per_train_sample"] for i in s["infos"]]) * 2 * E * N_TR_H / (2 * E * N_TR_H + E * N_VA_H + N_TE)
    if k.endswith("frozen"):  # frozen bottoms need no gradients: one augmented upload per epoch
        tr_bytes /= 2
    gf = cost.get("clip_b16_224", {}).get("flops_per_image")
    rows.append(row(lab + (", 3 seeds" if s["summary"]["n"] == 3 else f", {s['summary']['n']} seed" + ("s" if s["summary"]["n"] > 1 else "")), 3,
                    size(tr_bytes), (3 if "lora" in k else 1) * gf * E / 1e9 if gf else None, np.mean([i["secs"] for i in s["infos"]]) / 60, s["summary"]))
    t = "Lora" if "lora" in k else "Frozen"
    macro(f"vbSplit{t}Acc", f1(s["summary"]["acc"]))
    macro(f"vbSplit{t}Fone", f1(s["summary"]["f1"]))
    macro(f"vbSplit{t}Up", size(tr_bytes))
    macro(f"vbSplit{t}Min", f"{np.mean([i['secs'] for i in s['infos']]) / 60:.0f}")
    macro(f"vbSplit{t}N", str(s["summary"]["n"]))
lg = [(k.split("|")[0], v) for k, v in aud.get("heads", {}).items() if k.split("|")[1].startswith("logits")]
if lg:
    rows.append("\\midrule")
    rows.append("\\multicolumn{8}{@{}l}{\\emph{Few rounds, frozen features: each party trains an MLP to 4 logits (logits up, their gradients down)}}\\\\")
    for k, v in sorted(lg, key=lambda kv: ["clip_l14_224", "dinov2_g_224", "clip_b16_224"].index(kv[0]) if kv[0] in ("clip_l14_224", "dinov2_g_224", "clip_b16_224") else 9):
        c = cost.get(k, {})
        gf = c.get("flops_per_image")
        up = np.mean(v["epochs_run"]) * 2 * K * 4 * 2  # training passes only, as for split learning: 4 logits up, 4 gradients down, fp16
        rows.append(row(NAME[k], 3, size(up), gf / 1e9 if gf else None,
                        c.get("extract_secs_all_images") / 60 if c.get("extract_secs_all_images") else None, v))
        t = {"clip_l14_224": "Clip", "dinov2_g_224": "G", "clip_b16_224": "ClipB"}.get(k, k)
        macro(f"lg{t}Acc", f1(v["acc"]))
        macro(f"lg{t}Fone", f1(v["f1"]))
        macro(f"lg{t}Up", size(up))
        macro(f"lg{t}Ep", f"{np.median(v['epochs_run']):.0f}")
        m2 = aud["heads"].get(f"{k}|mlp2")
        if m2:
            macro(f"lg{t}Gap", f1(m2["acc"] - v["acc"]))
rows.append("\\midrule")
rows.append("\\multicolumn{8}{@{}l}{\\emph{One-shot, party-side self-supervised training (FedOnce)}}\\\\")
for k, bands, lab in (("nat_all", 13, "FedOnce (Noise As Targets)"), ("ssl_all", 13, "FedOnce, SimCLR variant"), ("ssl_rgb", 3, "FedOnce, SimCLR variant")):
    b = base[k]
    if b:
        rows.append(row(lab, bands, size(K * b.get("feature_dim", 512) * 2), None, b["ssl_pretrain_secs_all_parties"] / 60, b["summary"]))
rows.append("\\midrule")
rows.append("\\multicolumn{8}{@{}l}{\\emph{One-shot, frozen foundation features (this work), MLP head}}\\\\")
for k in ("dinov2_g_224", "clip_l14_224", "dinov2_b_224", "clip_b16_224", "ssl4eo_s2_vits16"):
    h = heads224.get(k)
    if k == "ssl4eo_s2_vits16" and not h and isinstance(Eold.get("E6_ssl4eo"), dict):
        h = {"dim": Eold["E6_ssl4eo"]["dim"], "mlp2": Eold["E6_ssl4eo"]["mlp2"]}
    if not h:
        continue
    c = cost.get(k, {})
    gf = c.get("flops_per_image")
    rows.append(row(NAME[k], 13 if "ssl4eo" in k else 3, size(K * h["dim"] * 2), gf / 1e9 if gf else None,
                    c.get("extract_secs_all_images") / 60 if c.get("extract_secs_all_images") else None, h["mlp2"]))
rows.append("\\midrule")
maj = (Eg.get("E5_paired") or {}).get("majority_acc") or (Eold.get("E5_paired") or {}).get("majority_acc")
if maj:
    rows.append(f"Majority class & -- & -- & -- & -- & {maj:.1f} & -- & -- \\\\")
    macro("majVB", f1(maj))
write("tab_main.tex", "\n".join(rows))
e5 = (Eg.get("E5_paired") or {}).get("clip_l14_224_vs_dinov2_g_224")
if e5:
    macro("pairDiff", f1(e5["diff"]))
    macro("pairLo", f1(e5["ci95"][0]))
    macro("pairHi", f1(e5["ci95"][1]))
    macro("pairP", f"{e5['mcnemar_p']:.2f}")
for k, m in (("dinov2_g_224", "G"), ("clip_l14_224", "Clip"), ("clip_b16_224", "ClipB"), ("dinov2_b_224", "DinoB"), ("dinov2_s_224", "DinoS")):
    h = heads224.get(k)
    if h:
        macro(f"vb{m}Acc", f1(h["mlp2"]["acc"]))
        macro(f"vb{m}Lin", f1(h["linear"]["acc"]))
        if k in cost:
            macro(f"vb{m}Min", f"{cost[k]['extract_secs_all_images'] / 60:.1f}")
            macro(f"vb{m}GF", f"{cost[k]['flops_per_image'] / 1e9:.0f}")
if r18:
    macro("rEighteenGF", f"{r18 / 1e9:.1f}")
    macro("rEighteenTrainGF", f"{3 * r18 * 50 / 1e9:.0f}")
for k, m in (("splitnn_rgb_pt", "splitPT"), ("nat_all", "nat"), ("ssl_all", "sslAll"), ("splitnn_all", "splitAll"), ("splitnn_rgb", "splitRGB")):
    if base[k]:
        macro(f"{m}Acc", f1(base[k]["summary"]["acc"]))
        secs = base[k].get("ssl_pretrain_secs_all_parties") or np.mean(base[k].get("secs_per_seed", [0]))
        macro(f"{m}Min", f"{secs / 60:.0f}")

# encoder table (VertiBench, 224 px)
rows = []
for k in ("dinov2_g_224", "dinov2_l_224", "dinov2_b_224", "dinov2_s_224", "clip_l14_224", "clip_b16_224", "siglip_b16_224", "resnet50_in1k_224", "dinov2_b_random_224"):
    h = heads224.get(k)
    if not h:
        continue
    c = cost.get(k, {})
    rows.append(f"{NAME[k]} & {c.get('params', 0) / 1e6:.0f} & {c.get('flops_per_image', 0) / 1e9:.0f} & {h['dim']} & {pm(h['mlp2'])} & {pm(h['mlp2'], 'f1')} & {pm(h['linear'])} \\\\")
write("tab_encoders.tex", "\n".join(rows))

combo = load("vb224_combo.json") or {}
CN = {"dinov2_g_224": "DINOv2-g", "clip_l14_224": "CLIP-L", "clip_b16_224": "CLIP-B", "dinov2_b_224": "DINOv2-B"}
rows = []
for k, v in combo.items():
    names = k.split("+")
    gf = sum(cost.get(n, {}).get("flops_per_image", 0) for n in names) / 1e9
    if len(names) < 2:
        continue
    pp = sum(cost.get(n, {}).get("params", 0) for n in names) / 1e6
    rows.append(f"{' + '.join(CN.get(n, n) for n in names)} & {pp:.0f} & {gf:.0f} & {v['dim']} & {pm(v['mlp2'])} & {pm(v['mlp2'], 'f1')} & -- \\\\")
    macro("combo" + "".join({"dinov2_g_224": "G", "clip_l14_224": "L", "clip_b16_224": "B", "dinov2_b_224": "D"}[n] for n in names), f1(v["mlp2"]["acc"]))
if rows:
    write("tab_combo_enc.tex", "\n".join(rows))

# ------------------------------------------------------------------ classic benchmarks vs published numbers
PUB = {  # (dataset, protocol): list of (method, number)
    ("cifar10", "K=2 top/bottom"): [("VFLAIR, splitVFL (ResNet-18, 30 ep.)", 79.8)],
    ("cifar10", "K=4 quadrants"): [("VFLAIR, splitVFL", 76.2), ("FedOnce, one-shot", 58.7), ("SplitNN (FedOnce paper)", 68.3)],
    ("cifar100", "K=2 top/bottom"): [("VFLAIR, splitVFL", 42.3)],
    ("cifar100", "K=4 quadrants"): [("VFLAIR, splitVFL", 38.2)],
    ("modelnet40", "K=4 (3 views/party, local mean)"): [("VIMADMM (ResNet-18)", 91.3), ("split learning (VIMADMM paper)", 90.0)],
}
DSN = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "modelnet40": "ModelNet40"}
rows = []
for (ds, proto), pubs in PUB.items():
    r = cls.get(ds, {})
    g, c = r.get("dinov2_g", {}).get(f"{proto}|mlp2"), r.get("clip_l14", {}).get(f"{proto}|mlp2")
    pub = "; ".join(f"{m} {v:.1f}" for m, v in pubs)
    rows.append(f"{DSN[ds]} & {proto.replace('K=', '$K{=}$')} & {pub} & {pm(g)} & {pm(c)} \\\\")
r = cls.get("modelnet40", {})
if r:
    for proto in ("K=12 (1 view/party)", "solo 1 view"):
        g, c = r.get("dinov2_g", {}).get(f"{proto}|mlp2"), r.get("clip_l14", {}).get(f"{proto}|mlp2")
        rows.append(f"ModelNet40 & {proto.replace('K=', '$K{=}$')} & -- & {pm(g)} & {pm(c)} \\\\")
for ds in ("cifar10", "cifar100"):
    r = cls.get(ds, {})
    for proto in ("solo bottom half", "solo quadrant", "centralised (full image)"):
        g, c = r.get("dinov2_g", {}).get(f"{proto}|mlp2"), r.get("clip_l14", {}).get(f"{proto}|mlp2")
        if g or c:
            rows.append(f"{DSN[ds]} & {proto} & -- & {pm(g)} & {pm(c)} \\\\")
write("tab_classic.tex", "\n".join(rows))
SUN = {256: (78.2, 80.4), 512: (81.1, 84.1), 1024: (85.3, 86.4), 2048: (86.1, 87.5)}
rows = []
for n, (one, few) in SUN.items():
    g = cls.get("cifar10", {}).get("dinov2_g", {}).get(f"aligned {n} left/right|mlp2")
    c = cls.get("cifar10", {}).get("clip_l14", {}).get(f"aligned {n} left/right|mlp2")
    rows.append(f"{n} & {one:.1f} & {few:.1f} & {pm(g)} & {pm(c)} \\\\")
    if g:
        macro(f"sun{ {256: 'A', 512: 'B', 1024: 'C', 2048: 'D'}[n] }".replace(" ", ""), f1(g["acc"]))
write("tab_sun.tex", "\n".join(rows))
for ds in ("cifar10", "cifar100"):
    for e in ("dinov2_g", "clip_l14"):
        for proto, t in (("K=2 top/bottom", "Two"), ("K=4 quadrants", "Four"), ("centralised (full image)", "Cent"), ("solo quadrant", "SoloQ"), ("solo bottom half", "SoloH")):
            s = cls.get(ds, {}).get(e, {}).get(f"{proto}|mlp2")
            if s:
                macro(f"{ {'cifar10': 'cTen', 'cifar100': 'cHun'}[ds] }{ {'dinov2_g': 'G', 'clip_l14': 'C'}[e] }{t}".replace(" ", ""), f1(s["acc"]))
mn = cls.get("modelnet40", {})
for e, t in (("dinov2_g", "G"), ("clip_l14", "C")):
    for proto, p in (("K=4 (3 views/party, local mean)", "Four"), ("K=12 (1 view/party)", "Twelve"), ("solo 1 view", "Solo")):
        s = mn.get(e, {}).get(f"{proto}|mlp2")
        if s:
            macro(f"mn{t}{p}", f1(s["acc"]))

# ------------------------------------------------------------------ MILK10k
MCFG = [("clin", "clinical photo (A)"), ("derm", "dermoscopy (B)"), ("meta", "metadata (C)"), ("clin+derm", "A + B"), ("derm+meta", "B + C"),
        ("clin+derm+meta", "A + B + C"), ("derm halves", "dermoscopy, left + right halves"), ("derm left", "dermoscopy, left half only"),
        ("clin halves", "clinical, left + right halves")]
MENC = [("dinov2_g", "DINOv2-g"), ("clip_l14", "CLIP-L"), ("biomedclip", "BiomedCLIP"), ("monet", "MONET")]
if milk:
    encs = [(e, n) for e, n in MENC if e in milk]
    rows = []
    for c, lab in MCFG:
        cells = []
        for e, _ in encs:
            s = milk[e].get(f"{c}|mlp2|bal") or (milk[MENC[0][0]].get(f"{c}|mlp2|bal") if c == "meta" else None)
            cells.append(pm(s, "f1"))
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if c in ("derm", "clin+derm+meta"):
            rows.append("\\midrule")
    write("tab_milk.tex", "\n".join(rows))
    write("tab_milk_head.tex", " & ".join(n for _, n in encs))
    macro("milkNcols", str(len(encs)))
    for e, t in (("dinov2_g", "G"), ("clip_l14", "C"), ("biomedclip", "B"), ("monet", "M")):
        if e not in milk:
            continue
        for c, tc in (("clin", "Clin"), ("derm", "Derm"), ("clin+derm", "Both"), ("clin+derm+meta", "All"), ("derm halves", "Halves"), ("derm+meta", "DermMeta")):
            s = milk[e].get(f"{c}|mlp2|bal")
            if s:
                macro(f"milk{t}{tc}", f1(s["f1"]))
                macro(f"milk{t}{tc}Acc", f1(s["acc"]))
                macro(f"milk{t}{tc}Bacc", f1(s["bacc"]))
        s0, s1 = milk[e].get("clin+derm|mlp2|plain"), milk[e].get("clin+derm|linear|bal")
        if s0:
            macro(f"milk{t}BothPlain", f1(s0["f1"]))
        if s1:
            macro(f"milk{t}BothLin", f1(s1["f1"]))
    s = milk.get("dinov2_g", {}).get("meta|mlp2|bal")
    if s:
        macro("milkMeta", f1(s["f1"]))
    macro("milkN", f"{milk['n_lesions']:,}".replace(",", "{,}"))
    cc = milk["class_counts"]
    macro("milkBCC", f1(100 * cc[0] / sum(cc)))
# MILK10k: how the label owner combines the parties (s17)
mf = load("milk_fusion.json")
if mf:
    ENCF = [("dinov2_g", "DINOv2-g"), ("clip_l14", "CLIP-L"), ("monet", "MONET"), ("dinov2_g+clip_l14", "DINOv2-g + CLIP-L"), ("dinov2_g+monet", "DINOv2-g + MONET")]
    rows = []
    for e, lab in ENCF:
        r = mf.get(e)
        if not r:
            continue
        d = r.get("diag clin vs derm", {})
        cells = [pm(r.get(k), "f1") for k in ("derm|concat", "clin+derm|concat", "clin+derm|late fusion", "clin+derm|branch aux md0.3", "clin+derm+meta|branch aux md0.3")]
        rows.append(f"{lab} & " + " & ".join(cells) + (f" & {d['both_wrong']:.1f} ({d['both_wrong_if_independent']:.1f})" if d else " & --") + " \\\\")
        t = {"dinov2_g": "G", "clip_l14": "C", "monet": "M", "dinov2_g+clip_l14": "GC", "dinov2_g+monet": "GM"}[e]
        for k, tag in (("derm|concat", "Derm"), ("clin+derm|concat", "Cat"), ("clin+derm|late fusion", "Late"), ("clin+derm|branch aux md0.3", "Branch"),
                       ("clin+derm+meta|branch aux md0.3", "BranchMeta"), ("clin+derm+meta|concat", "CatMeta")):
            if k in r:
                macro(f"mf{t}{tag}", f1(r[k]["f1"]))
        if d:
            for k2, tag in (("both_wrong", "Both"), ("both_wrong_if_independent", "Indep"), ("oracle_acc", "Oracle"), ("err_a", "ErrClin"), ("err_b", "ErrDerm")):
                macro(f"mf{t}{tag}", f1(d[k2]))
    write("tab_milkfusion.tex", "\n".join(rows))
# MILK10k: over-fitting or modality competition? (s19)
mr = load("milk_reg.json")
if mr:
    EN = [e for e in ("dinov2_g", "clip_l14", "monet") if e in mr]
    RR = [("concat", "concat MLP"), ("concat + input dropout 0.5", "\\quad + input dropout 0.5"), ("concat + feature noise 0.5", "\\quad + Gaussian feature noise"),
          ("concat + feature mixup 0.4", "\\quad + feature mixup"), ("concat + PCA-256 per party", "\\quad + PCA-256 per party"),
          ("late fusion (weight on val)", "late fusion, weight tuned on validation"), ("late fusion (fixed 0.5)", "late fusion, fixed weight 0.5"), None,
          ("concat|views mean", "concat MLP, mean of 8 views (upload $\\times$1)"), ("concat|views sampled (x8 upload)", "concat MLP, 8 views sampled (upload $\\times$8)"),
          ("late fusion (fixed 0.5)|views mean", "late fusion, mean of 8 views (upload $\\times$1)")]
    rows = []
    for r in RR:
        if r is None:
            rows.append("\\midrule")
            continue
        k, lab = r
        rows.append(f"{lab} & " + " & ".join(pm(mr[e].get(k), "f1") for e in EN) + " \\\\")
    rows.append("\\midrule")
    for fr, lab in (("frac0.25", "25\\,\\%"), ("frac0.5", "50\\,\\%"), ("", "100\\,\\%")):
        cells = []
        for e in EN:
            a = mr[e].get("late fusion (fixed 0.5)" + (f"|{fr}" if fr else ""))
            b = mr[e].get("concat" + (f"|{fr}" if fr else ""))
            cells.append(f"{a['f1'] - b['f1']:+.1f}" if a and b else "--")
        rows.append(f"late $-$ concat, {lab} of the training folds & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    for sel, lab, ck in (("wloss", "stopping on the class-weighted val.\\ loss", "concat|stop wloss"), ("f1", "stopping on the val.\\ macro F1", "concat|stop f1"),
                         ("f1", "\\quad the same, concat with weight decay $10^{-3}$", "concat + weight decay 1e-3|stop f1")):
        cells = []
        for e in EN:
            a, b = mr[e].get(f"late fusion (fixed 0.5)|stop {sel}"), mr[e].get(ck)
            cells.append(f"{a['f1'] - b['f1']:+.1f}" if a and b else "--")
        rows.append(f"{'late $-$ concat, ' if 'quad' not in lab else ''}{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_milkreg.tex", "\n".join(rows))
    # same early-stopping rule for both heads, over the four encoders that were re-run (s19 PE_REG_EXTRA)
    E4 = [e for e in ("dinov2_g", "clip_l14", "dinov2_b", "clip_b16") if "concat|stop f1" in mr.get(e, {})]
    gs = [mr[e][f"late fusion (fixed 0.5)|stop {sel}"]["f1"] - mr[e][f"concat|stop {sel}"]["f1"] for e in E4 for sel in ("wloss", "f1")]
    gw = [mr[e]["late fusion (fixed 0.5)|stop f1"]["f1"] - mr[e]["concat + weight decay 1e-3|stop f1"]["f1"] for e in E4]
    if gs:
        macro("mrStopMin", f1(min(gs)))
        macro("mrStopMax", f1(max(gs)))
        macro("mrStopN", str(len(gs)))
        macro("mrWdMin", f1(min(gw)))
        macro("mrWdMax", f1(max(gw)))
        dw = [mr[e][f"{h}|stop wloss"]["f1"] - mr[e][h]["f1"] for e in E4 for h in ("concat", "late fusion (fixed 0.5)")]
        macro("mrWlossDropMax", f1(-min(dw)))
    for e, t in (("dinov2_g", "G"), ("clip_l14", "C"), ("monet", "M")):
        r = mr.get(e, {})
        for k, tag in (("concat", "Cat"), ("concat + input dropout 0.5", "Drop"), ("concat + feature mixup 0.4", "Mix"), ("concat + feature noise 0.5", "Noise"),
                       ("concat + PCA-256 per party", "Pca"), ("late fusion (weight on val)", "Late"), ("late fusion (fixed 0.5)", "LateHalf"),
                       ("concat|views mean", "CatViews"), ("concat|views sampled (x8 upload)", "CatSampled"), ("late fusion (fixed 0.5)|views mean", "LateViews"),
                       ("derm|views mean", "DermViews"), ("clin|views mean", "ClinViews")):
            if k in r:
                macro(f"mr{t}{k and tag}", f1(r[k]["f1"]))
        for fr, tag in (("frac0.25", "Q"), ("frac0.5", "H"), ("", "F")):
            a = r.get("late fusion (fixed 0.5)" + (f"|{fr}" if fr else ""))
            b = r.get("concat" + (f"|{fr}" if fr else ""))
            if a and b:
                macro(f"mr{t}Gap{tag}", f"{a['f1'] - b['f1']:+.1f}")

for enc, t in (("dinov2_g", "G"), ("clip_l14", "C"), ("monet", "M")):
    r = (load("milk_reg.json") or {}).get(enc, {})
    a, b = r.get("late fusion (fixed 0.5)"), r.get("concat")
    if a and b and "f1s" in a and "f1s" in b:
        x = np.array(a["f1s"]) - np.array(b["f1s"])
        macro(f"mr{t}PairD", f1(x.mean()))
        macro(f"mr{t}PairS", f1(x.std(ddof=1)))
        macro(f"mr{t}PairW", str(int((x > 0).sum())))

# price of freezing (MILK10k, clinical + dermoscopic)
FZ = [("resnet18", "ResNet-18, scratch", "split"), ("resnet18_in", "ResNet-18, ImageNet", "split"), ("dinov2_b_lora", "DINOv2-B, LoRA", "split"),
      ("dinov2_b_ft", "DINOv2-B, fine-tuned", "split"), ("clip_b16_lora", "CLIP-B, LoRA", "split"),
      ("dinov2_b", "DINOv2-B, frozen", "frozen"), ("clip_b16", "CLIP-B, frozen", "frozen"), ("dinov2_g", "DINOv2-g, frozen", "frozen"), ("clip_l14", "CLIP-L, frozen", "frozen")]
if msplit or milk:
    rows = []
    for k, lab, kind in FZ:
        if kind == "split":
            s = msplit.get(k)
            if not s:
                continue
            E = s["epochs"]  # training passes only (activations up, gradients down), as in Table 1: 3,144 training,
            up = s["bytes_per_train_sample"] * 2 * E * 3144 / (2 * E * 3144 + E * 1048 + 1048)  # 1,048 val. and test lesions
            rows.append(f"{lab} & split, {s['epochs']} ep. & {s['trainable_params'] / 1e6:.1f}\\,M & {size(up)} & {np.mean(s['secs_per_fold']) / 60:.1f} & {pm(s, 'f1')} & {pm(s, 'bacc')} & {pm(s)} \\\\")
            macro(f"fzMin{k}", f"{np.mean(s['secs_per_fold']) / 60:.1f}")
            macro(f"fzUp{k}", size(up))
            macro(f"fz{k.replace('_', '').replace('18', 'Eighteen').replace('16', 'Sixteen')}", f1(s["f1"]))
        else:
            s = (milk or {}).get(k, {}).get("clin+derm|mlp2|bal")
            if not s:
                continue
            if rows and rows[-1] != "\\midrule" and k == "dinov2_b":
                rows.append("\\midrule")
            d = {"dinov2_b": 768, "clip_b16": 768, "dinov2_g": 1536, "clip_l14": 1024}[k]
            hp = (2 * d * 1000 + 1000 + 2 * 1000 + 1000 * 7 + 7) / 1e6  # MLP head: Linear(2d,1000)-BN-ReLU-Linear(1000,7)
            ct = cost.get(k + "_224", {}).get("extract_secs_all_images")
            tm = f"{ct / 62832 * 10480 / 60:.1f}" if ct else "--"
            rows.append(f"{lab} & one-shot & {hp:.1f}\\,M & {size(2 * d * 2)} & {tm} & {pm(s, 'f1')} & {pm(s, 'bacc')} & {pm(s)} \\\\")
            macro(f"fz{k.replace('_', '').replace('16', 'Sixteen').replace('14', 'Fourteen')}", f1(s["f1"]))
    rows.append("\\midrule")
    for k, lab, d in (("dinov2_b", "DINOv2-B", 768), ("clip_b16", "CLIP-B", 768), ("dinov2_g", "DINOv2-g", 1536)):
        mrg = (load("milk_reg.json") or {}).get(k, {}).get("late fusion (fixed 0.5)|views mean")
        if not mrg:
            continue
        hp2 = 2 * (d * 1000 + 1000 + 2 * 1000 + 1000 * 7 + 7) / 1e6  # one MLP head per party
        ct = cost.get(k + "_224", {}).get("extract_secs_all_images")
        tm = f"{8 * ct / 62832 * 10480 / 60:.1f}" if ct else "--"
        rows.append(f"{lab}, frozen, 8 views, late fusion & one-shot & {hp2:.1f}\\,M & {size(2 * d * 2)} & {tm} & {pm(mrg, 'f1')} & {pm(mrg, 'bacc')} & {pm(mrg)} \\\\")
        lo = msplit.get(k + "_lora")
        fr = (milk or {}).get(k, {}).get("clin+derm|mlp2|bal")
        if lo and fr:
            t = {"dinov2_b": "B", "clip_b16": "Cb"}[k]
            macro(f"fzGap{t}", f1(lo["f1"] - fr["f1"]))
            macro(f"fzGapViews{t}", f1(lo["f1"] - mrg["f1"]))
            macro(f"fzViews{t}", f1(mrg["f1"]))
    write("tab_freeze.tex", "\n".join(rows))

# ------------------------------------------------------------------ WorldStrat: sensors, labels, resolution
if ws:
    TASKS = [("ipcc", "IPCC land cover (7)"), ("smod", "degree of urbanisation (8)"), ("lccs", "LCCS land cover (18)")]
    FTASKS = [("ipcc", "IPCC land cover (7 classes)"), ("lccs", "LCCS land cover (18 classes)"), ("smod", "degree of urbanisation (8 classes)")]
    GS = ["1.5m", "3m", "6m", "10m"]
    gx = [1.5, 3, 6, 10]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.35), sharex=True)
    for ax, (t, lab) in zip(axs, FTASKS):
        r = ws.get(t, {})
        for cfg, col, ls, l in (("SPOT {g}", C["blue"], "-", "SPOT alone"), ("SPOT {g} + S2 MS", C["green"], "-", "SPOT + S2 (RGB, NIR, SWIR)"),
                                ("SPOT {g} + S2 RGB", C["orange"], "--", "SPOT + S2 (RGB)")):
            pts = [(x, r.get(cfg.format(g=g) + "|mlp2")) for x, g in zip(gx, GS)]
            pts = [(x, s["f1"]) for x, s in pts if s]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, color=col, ls=ls, lw=1.2, label=l)
        for cfg, col, l in (("S2 MS", C["green"], "S2 alone (RGB, NIR, SWIR)"), ("S2 RGB", C["orange"], "S2 alone (RGB)")):
            s = r.get(cfg + "|mlp2")
            if s:
                ax.axhline(s["f1"], color=col, ls=":", lw=1, label=l)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(gx)
        ax.set_xticklabels(["1.5", "3", "6", "10"])
        ax.set_xlabel("SPOT ground sampling (m)")
        ax.set_title(lab, fontsize=7)
    axs[0].set_ylabel("macro F1 (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_ws.pdf"))
    plt.close(fig)
    WCFG = [("SPOT 1.5m", "SPOT, 1.5\\,m (A)"), ("SPOT 10m", "SPOT, degraded to 10\\,m"), ("S2 RGB", "Sentinel-2, RGB (B)"), ("S2 MS", "Sentinel-2, RGB + NIR + SWIR"),
            ("S2 SSL4EO", "Sentinel-2, SSL4EO (13 bands)"), ("SPOT 1.5m + S2 RGB", "A + B (RGB)"), ("SPOT 1.5m + S2 MS", "A + B (RGB, NIR, SWIR)"),
            ("SPOT 1.5m + S2 SSL4EO", "A + B (SSL4EO)"), ("SPOT 10m + S2 MS", "A at 10\\,m + B (RGB, NIR, SWIR)"), ("SPOT CLIP-L", "SPOT, CLIP-L 224\\,px"),
            ("SPOT CLIP-L + S2 CLIP-L", "A + B, CLIP-L")]
    rows = []
    for c, lab in WCFG:
        cells = [pm(ws.get(t, {}).get(c + "|mlp2"), "f1") for t, _ in TASKS]
        if all(x == "--" for x in cells):
            continue
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if c in ("S2 SSL4EO", "SPOT 10m + S2 MS"):
            rows.append("\\midrule")
    write("tab_ws.tex", "\n".join(r for i, r in enumerate(rows) if not (r == "\\midrule" and i == len(rows) - 1)))
    for t, tt in (("ipcc", "Ipcc"), ("smod", "Smod"), ("lccs", "Lccs")):
        r = ws.get(t, {})
        for c, tc in (("SPOT 1.5m", "Spot"), ("SPOT 10m", "SpotTen"), ("S2 RGB", "Srgb"), ("S2 MS", "Sms"), ("SPOT 1.5m + S2 MS", "Fus"), ("SPOT 1.5m + S2 RGB", "FusRgb"),
                       ("SPOT 10m + S2 MS", "FusTen"), ("SPOT 6m + S2 MS", "FusSix"), ("SPOT 6m", "SpotSix"), ("S2 SSL4EO", "Sssl"),
                       ("SPOT 3m", "SpotThree"), ("SPOT 3m + S2 MS", "FusThree")):
            s = r.get(c + "|mlp2")
            if s:
                macro(f"ws{tt}{tc}", f1(s["f1"]))
        for g, gt in (("1.5m", "A"), ("3m", "Three"), ("6m", "Six"), ("10m", "Ten")):
            ka, kb = f"SPOT {g} + S2 MS|mlp2", f"SPOT {g}|mlp2"
            if ka in r and kb in r:
                x = np.array(r[ka]["f1s"]) - np.array(r[kb]["f1s"])
                macro(f"wg{tt}{gt}", ("+" if x.mean() >= 0 else "$-$") + f"{abs(x.mean()):.1f}")
                macro(f"wg{tt}{gt}W", f"{int((x > 0).sum())}/{len(x)}")
        if "counts" in r:
            macro(f"ws{tt}N", f"{sum(r['counts']):,}".replace(",", "{,}"))

# ------------------------------------------------------------------ WorldStrat: making the second party help (s16)
wf = load("ws_fusion_grouped.json") or load("ws_fusion.json")
if wf:
    FT = [("ipcc", "IPCC"), ("lccs", "LCCS"), ("smod", "SMOD")]
    FR = [("SPOT|mlp2", "SPOT, full area", "--"), ("SPOT tiles mean|mlp2", "SPOT, mean of 16 tiles at full resolution", "--"),
          ("SPOT centre1k|mlp2", "SPOT, centre 1\\,km", "--"), ("SPOT centre|mlp2", "SPOT, centre 336\\,m", "--"),
          ("SPOT full+centre|mlp2", "SPOT, full area and centre 336\\,m", "--"), None,
          ("S2 mean|mlp2", "S2, mean of revisits", "1"), ("S2 mean+std|mlp2", "S2, mean and temporal std", "2"),
          ("S2 set16|set attn", "S2, 16 revisits, shared-weight set head", "16"), None,
          ("SPOT + S2 mean|concat", "concat MLP, S2 mean", "1"), ("SPOT + S2 mean|late fusion", "S2 mean, late fusion", "1"),
          ("SPOT + S2 mean+std|concat", "concat MLP, S2 mean and std", "2"),
          ("SPOT + S2 clear-mean|concat", "concat MLP, S2 mean of 8 clearest (C)", "1"), ("SPOT + S2 clear-mean|pca256 concat", "C, PCA-256 per party", "1"),
          ("SPOT + S2 clear-mean|late fusion", "C, late fusion", "1"), ("SPOT + S2 clear-mean|branch aux md0.3", "C, branches, aux.\\ losses, modality dropout", "1"),
          ("SPOT + S2 set16|set attn", "S2 by shared-weight set head (16 revisits)", "16"), None,
          ("SPOT full+centre + S2 centre clear-mean|late fusion", "full area and centre, late fusion", "1"),
          ("SPOT 3 scales + S2 3 scales clear-mean|concat", "3 scales per party, concat MLP", "3"),
          ("SPOT 3 scales + S2 3 scales clear-mean|late fusion", "3 scales per party, late fusion", "3")]
    rows = []
    for r in FR:
        if r is None:
            if rows and rows[-1] != "\\midrule":
                rows.append("\\midrule")
            continue
        key, lab, up = r
        cells = [pm(wf.get(t, {}).get(key), "f1") for t, _ in FT]
        if all(c == "--" for c in cells):
            continue
        rows.append(f"{lab} & {up} & " + " & ".join(cells) + " \\\\")
    write("tab_wsfusion.tex", "\n".join(rows))
    for t, tt in FT:
        r = wf.get(t, {})
        for key, tag in (("SPOT|mlp2", "Spot"), ("SPOT + S2 mean|concat", "Cat"), ("SPOT + S2 clear-mean|concat", "CatClear"), ("SPOT + S2 mean+std|concat", "CatStd"),
                         ("SPOT + S2 clear-mean|late fusion", "Late"), ("SPOT + S2 clear-mean|branch aux md0.3", "Branch"), ("SPOT + S2 set16|set attn", "Set"),
                         ("SPOT + S2 set8 clear|set attn", "SetClear"), ("S2 set16|set attn", "SolSet"), ("S2 mean|mlp2", "SolMean"), ("S2 clear-mean|mlp2", "SolClear"),
                         ("SPOT full+centre|mlp2", "SpotFC"), ("SPOT centre|mlp2", "SpotC"), ("SPOT full+centre + S2 centre clear-mean|concat", "FCCat"),
                         ("SPOT full+centre + S2 centre clear-mean|late fusion", "FCLate"), ("SPOT full+centre + S2 centre set8 clear|set attn", "FCSet"),
                         ("SPOT 3 scales|mlp2", "SpotMS"), ("S2 3 scales clear-mean|mlp2", "SolMS"), ("SPOT 3 scales + S2 3 scales clear-mean|concat", "MSCat"),
                         ("SPOT 3 scales + S2 3 scales clear-mean|late fusion", "MSLate"), ("SPOT 3 scales + S2 3 scales clear-mean|branch aux md0.3", "MSBranch")):
            s = r.get(key)
            if s:
                macro(f"wf{tt.capitalize()}{tag}", f1(s["f1"]))
        best = max(((k, v["f1"]) for k, v in r.items() if isinstance(v, dict) and "f1" in v and "+" in k), key=lambda kv: kv[1], default=None)
        if best:
            macro(f"wf{tt.capitalize()}BestFus", f1(best[1]))
        for tag, (ka, kb) in {"FusCat": ("SPOT + S2 mean|concat", "SPOT|mlp2"), "LateCat": ("SPOT + S2 clear-mean|late fusion", "SPOT + S2 clear-mean|concat"),
                              "SetCat": ("SPOT + S2 set16|set attn", "SPOT + S2 mean|concat"), "StdSolo": ("S2 mean+std|mlp2", "S2 mean|mlp2"),
                              "StdFus": ("SPOT + S2 mean+std|concat", "S2 mean+std|mlp2"), "Centre": ("SPOT|mlp2", "SPOT centre|mlp2"),
                              "FC": ("SPOT full+centre|mlp2", "SPOT|mlp2"), "Tiles": ("SPOT tiles mean|mlp2", "SPOT|mlp2"),
                              "Pca": ("SPOT + S2 clear-mean|pca256 concat", "SPOT + S2 clear-mean|concat"),
                              "Branch": ("SPOT + S2 clear-mean|branch aux md0.3", "SPOT + S2 clear-mean|concat"),
                              "LateMean": ("SPOT + S2 mean|late fusion", "SPOT + S2 mean|concat"),
                              "PcaMean": ("SPOT + S2 clear-mean|pca256 concat", "SPOT + S2 mean|concat"),
                              "TilesFus": ("SPOT tiles mean + S2 clear-mean|concat", "SPOT tiles mean|mlp2")}.items():
            if ka in r and kb in r:
                x = np.array(r[ka]["f1s"]) - np.array(r[kb]["f1s"])
                macro(f"wp{tt.capitalize()}{tag}", ("+" if x.mean() >= 0 else "$-$") + f"{abs(x.mean()):.1f}")
                macro(f"wp{tt.capitalize()}{tag}W", f"{int((x > 0).sum())}/{len(x)}")
        if "S2 mean+std|mlp2" in r:
            macro(f"wf{tt.capitalize()}SolStd", f1(r["S2 mean+std|mlp2"]["f1"]))
        if "SPOT tiles mean|mlp2" in r:
            macro(f"wf{tt.capitalize()}SpotTiles", f1(r["SPOT tiles mean|mlp2"]["f1"]))
        if "SPOT centre1k|mlp2" in r:
            macro(f"wf{tt.capitalize()}SpotCk", f1(r["SPOT centre1k|mlp2"]["f1"]))
        for dk, dt in (("diag SPOT vs S2 clear-mean", ""), ("diag SPOT 3 scales vs S2 3 scales clear-mean", "MS")):
            d = r.get(dk)
            if not d:
                continue
            for k2, tag in (("err_a", "ErrA"), ("err_b", "ErrB"), ("both_wrong", "Both"), ("oracle_acc", "Oracle"), ("both_wrong_if_independent", "Indep")):
                macro(f"wf{tt.capitalize()}{dt}{tag}", f1(d[k2]))

# ------------------------------------------------------------------ compression (Pareto, validation-selected points) at 224 px
par = Eg.get("E7_pareto_val") if isinstance(Eg.get("E7_pareto_val"), list) else None
D = 1536
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
    for yv, lab, ls in refs:
        ax.axhline(yv, color=C["grey"], ls=ls, lw=0.8)
        ax.text(1.2 * min(p["bytes_per_sample"] for p in par), yv + 0.12, lab, fontsize=5.5, color=C["grey"])
    budgets = [256, 512, 2048, 16 * D * 2]
    sel = {}
    for B in budgets:
        cand = [p for p in par if p["bytes_per_sample"] <= B]
        best = max(cand, key=lambda p: p["val_acc"])
        sel[B] = best
        ax.scatter([best["bytes_per_sample"]], [best["acc"]], s=40, facecolors="none", edgecolors="k", lw=0.8, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512, 2048, 8192, 16 * D * 2])
    ax.set_xticklabels(["32 B", "128 B", "512 B", "2 KiB", "8 KiB", "48 KiB"])
    ax.set_xlabel("one-time upload per sample, all 16 parties")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(title="precision", ncol=3, frameon=False, loc="lower right", fontsize=6, title_fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
    plt.close(fig)
    full = next(p for p in par if p["d"] == D and p["bits"] == 16)
    macro("fullAcc", f1(full["acc"]))
    for B, tag in zip(budgets, ("Two", "Five", "Twok", "Full")):
        p = sel[B]
        macro(f"sel{tag}Acc", f1(p["acc"]))
        macro(f"sel{tag}Cfg", f"PCA-{p['d']}, {p['bits']}-bit" if p["d"] < D else f"{p['bits']}-bit")
        macro(f"sel{tag}Bytes", size(p["bytes_per_sample"]))
    rows = []
    for p in sorted(par, key=lambda p: (-p["d"], -p["bits"])):
        rows.append(f"{p['d']} & {p['bits']} & {size(p['bytes_per_sample'])} & {p['val_acc']:.1f} & {pm(p)} & {pm(p, 'auc')} \\\\")
    write("tab_pareto.tex", "\n".join(rows))
    look = {(p["d"], p["bits"]): p for p in par}
    for (d, b), tag in (((64, 16), "PcaSf"), ((16, 16), "PcaSt"), ((256, 16), "PcaTf"), ((D, 1), "OneBit"), ((64, 4), "PcaSfFour")):
        if (d, b) in look:
            macro(f"cmp{tag}", f1(look[(d, b)]["acc"]))

# ------------------------------------------------------------------ parties: subsets, missing parties (concat + imputation, pooling heads)
e3, e2 = Eg.get("E3_parties"), Eg.get("E2_missing")
if isinstance(e3, dict):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ks = [s["k"] for s in e3["subsets"]]
    axs[0].errorbar(ks, [s["acc"] for s in e3["subsets"]], yerr=[s["acc_std"] for s in e3["subsets"]], marker="o", ms=3, capsize=2, color=C["blue"], label="DINOv2-g")
    e3c = Ec.get("E3_parties")
    if isinstance(e3c, dict):
        axs[0].errorbar([s["k"] for s in e3c["subsets"]], [s["acc"] for s in e3c["subsets"]], yerr=[s["acc_std"] for s in e3c["subsets"]], marker="s", ms=3, capsize=2, color=C["orange"], label="CLIP-L")
        axs[0].legend(frameon=False, fontsize=6)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(ks)
    axs[0].set_xticklabels(ks)
    axs[0].set_xlabel("number of parties $k$ (random subsets)")
    axs[0].set_ylabel("test accuracy (%)")
    if isinstance(e2, list):
        styles = {("0.0", "zero"): (C["grey"], "-", "concat, no dropout, zeros"), ("0.0", "train_mean"): (C["grey"], "--", "concat, no dropout, mean imputation"),
                  ("0.25", "train_mean"): (C["orange"], "--", "concat, party dropout 0.25, mean imp."), ("0.5", "zero"): (C["green"], "-", "concat, party dropout 0.5, zeros")}
        for (pd_, imp), (col, ls, lab) in styles.items():
            pts = sorted([(m["missing_at_test"], m["acc"], sd(m, "acc")) for m in e2 if f"{m['train_party_dropout']:g}" == f"{float(pd_):g}" and (m["imputation"] == imp or m["missing_at_test"] == 0)])
            if pts:
                axs[1].errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="o", ms=2.5, capsize=1.5, color=col, ls=ls, label=lab, lw=1)
    if pool:
        pts = sorted([(m["missing_at_test"], m["acc"], sd(m, "acc")) for m in pool["missing"] if m["arch"] == "mean"])
        if pts:
            axs[1].errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], marker="D", ms=2.5, capsize=1.5, color=C["blue"], lw=1.2, label="mean-pooling head (no imputation)")
            for m, a, _ in pts:
                macro(f"poolMiss{ {0: 'None', 4: 'Four', 8: 'Eight', 12: 'Twelve', 15: 'Fifteen'}[m] }".replace(" ", ""), f1(a))
    axs[1].set_xlabel("parties missing at inference (of 16)")
    axs[1].legend(frameon=False, fontsize=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_parties.pdf"))
    plt.close(fig)
    for s in e3["subsets"]:
        macro(f"partiesK{ {1: 'One', 2: 'Two', 4: 'Four', 8: 'Eight', 16: 'All'}[s['k']] }".replace(" ", ""), f1(s["acc"]))
    solo = [s["acc"] for s in e3["solo"]]
    macro("soloMean", f1(np.mean(solo)))
    macro("soloMin", f1(min(solo)))
    macro("soloMax", f1(max(solo)))
    if isinstance(e2, list):
        get = lambda p, m, imp: next((x for x in e2 if abs(x["train_party_dropout"] - p) < 1e-9 and x["missing_at_test"] == m and (x["imputation"] == imp or m == 0)), None)  # noqa: E731
        for p, tp in ((0.0, "Zero"), (0.25, "Q"), (0.5, "H")):
            for m, tm in ((8, "Eight"), (12, "Twelve")):
                for imp, ti in (("zero", "Z"), ("train_mean", "M")):
                    x = get(p, m, imp)
                    if x:
                        macro(f"miss{tp}{tm}{ti}", f1(x["acc"]))
if pool:
    for k, t in (("dinov2_g_224/mlp2", "GConcat"), ("dinov2_g_224/mean", "GMean"), ("dinov2_g_224/attention", "GAtt"), ("clip_l14_224/mlp2", "CConcat"),
                 ("clip_l14_224/mean", "CMean"), ("dinov2_g_224_pca64/mean", "GPcaMean"), ("dinov2_g_224_pca64/mlp2", "GPcaConcat")):
        s = pool["official"].get(k)
        if s:
            macro(f"pool{t}", f1(s["acc"]))

# ------------------------------------------------------------------ privacy at 224 px
e4 = Eg.get("E4_dp")
if inv and "single" in inv:
    accmap = {}
    if par:
        look = {(p["d"], p["bits"]): p for p in par}
        for name, key in (("full, 16-bit", (D, 16)), ("full, 8-bit", (D, 8)), ("full, 4-bit", (D, 4)), ("full, 1-bit", (D, 1)),
                          ("PCA-256", (256, 16)), ("PCA-64", (64, 16)), ("PCA-16", (16, 16)), ("PCA-64, 4-bit", (64, 4)), ("PCA-16, 4-bit", (16, 4))):
            if key in look:
                accmap[name] = (look[key], size(look[key]["bytes_per_sample"]))
    if isinstance(e4, list):
        for r in e4:
            if r["d"] == D and r["eps_upload"]:
                accmap[f"DP eps={r['eps_upload']}"] = (r, size(16 * D * 2))
    if "clip_l14_224" in heads224:
        accmap["CLIP-L, full, 16-bit"] = (heads224["clip_l14_224"]["mlp2"], size(16 * 1024 * 2))
    rows = []
    order = ["full, 16-bit", "full, 8-bit", "full, 4-bit", "full, 1-bit", "PCA-256", "PCA-64", "PCA-16", "PCA-64, 4-bit", "PCA-16, 4-bit", "DP eps=64", "DP eps=8", "CLIP-L, full, 16-bit"]
    pretty = {"noise rho=1": "noise $\\rho=1$", "noise rho=3": "noise $\\rho=3$", "DP eps=64": "clip + noise, $\\varepsilon=64$", "DP eps=8": "clip + noise, $\\varepsilon=8$"}
    V2INV = "ridge" in inv
    for name in order:
        r = inv["single"].get(name)
        if not r:
            continue
        a = accmap.get(name)
        if V2INV:  # strongest of the two attackers for each metric; L marks the linear decoder
            cand = [(r, ""), (inv["ridge"].get(name), "$^{\\ell}$"), (inv1.get("single", {}).get(name), "$^{s}$")]
            cand = [(x, m) for x, m in cand if x]

            def best(f):
                return max(((f(x), m) for x, m in cand if f(x) is not None), key=lambda t: t[0])
            (bp, lp), (bs, ls), (bi, li), (bl, ll) = (best(lambda x: x["psnr"]), best(lambda x: x["ssim"]), best(lambda x: x["reid"]["top1"]),
                                                      best(lambda x: x.get("reid_loc", {}).get("top1")))
            rows.append(f"{pretty.get(name, name)} & {a[1] if a else '--'} & {pm(a[0]) if a else '--'} & {bp:.1f}{lp} & {bs:.2f}{ls} & {bi:.2f}{li} & {bl:.1f}{ll} \\\\")
        else:
            ps = f"{r['psnr']:.1f}" + (f"\\,{{\\scriptsize$\\pm${np.std(r['psnr_seeds'], ddof=1):.1f}}}" if len(r.get("psnr_seeds", [])) > 1 else "")
            rows.append(f"{pretty.get(name, name)} & {a[1] if a else '--'} & {pm(a[0]) if a else '--'} & {ps} & {r['ssim']:.2f} & {r['reid']['top1']:.1f} \\\\")
        if name in ("PCA-16", "DP eps=8", "PCA-16, 4-bit", "full, 1-bit"):
            rows.append("\\midrule")
    rows.append("\\midrule")
    X = " & --" if V2INV else ""
    rows.append(f"\\emph{{reference:}} mean training image & -- & -- & {inv['mean_image']['psnr']:.1f} & {inv['mean_image']['ssim']:.2f} & 0.0{X} \\\\")
    if "mean_colour_oracle" in inv:
        mo = inv["mean_colour_oracle"]
        rl = f" & {mo['reid_loc']['top1']:.1f}" if V2INV else ""
        rows.append(f"\\emph{{reference:}} each image's mean colour & -- & -- & {mo['psnr']:.1f} & {mo['ssim']:.2f} & {mo['reid']['top1']:.{2 if V2INV else 1}f}{rl} \\\\")
    rows.append(f"\\emph{{reference:}} nearest training image & -- & -- & {inv['nearest_neighbour']['psnr']:.1f} & {inv['nearest_neighbour']['ssim']:.2f} & --{X} \\\\")
    if V2INV:
        rows.append(f"\\emph{{chance}} & -- & -- & -- & -- & {100 / inv['n_test_images']:.2f} & {100 * 16 / inv['n_test_images']:.1f} \\\\")
    rows = [r for i, r in enumerate(rows) if not (r == "\\midrule" and i + 1 < len(rows) and rows[i + 1] == "\\midrule")]
    write("tab_privacy.tex", "\n".join(rows))
    for name, tag in (("full, 16-bit", "Full"), ("PCA-64", "PCAsf"), ("PCA-16", "PCAst"), ("PCA-256", "PCAtf"), ("DP eps=8", "DPeight"), ("DP eps=64", "DPsf"),
                      ("full, 1-bit", "Onebit"), ("CLIP-L, full, 16-bit", "Clip"), ("noise rho=3", "Noisethree")):
        r = inv["single"].get(name)
        if r:
            macro(f"inv{tag}", f1(r["psnr"]))
            macro(f"inv{tag}Reid", f"{r['reid']['top1']:.{2 if 'reid_loc' in r else 1}f}")
            if "reid_loc" in r:
                macro(f"inv{tag}ReidLoc", f1(r["reid_loc"]["top1"]))
                macro(f"inv{tag}ReidLocTen", f1(r["reid_loc"]["top10"]))
                macro(f"inv{tag}TrainMse", f"{r['train_loss_last_epoch']:.4f}")
        q = inv.get("ridge", {}).get(name)
        if q:
            macro(f"invLin{tag}", f1(q["psnr"]))
            macro(f"invLin{tag}ReidLoc", f1(q["reid_loc"]["top1"]))
        o = inv1.get("single", {}).get(name)
        if r and q:  # best over the attackers, as in the table
            al = [x for x in (r, q, o) if x]
            macro(f"invBest{tag}", f1(max(x["psnr"] for x in al)))
            macro(f"invBest{tag}ReidLoc", f1(max(x["reid_loc"]["top1"] for x in al if "reid_loc" in x)))
            if o and name == "full, 16-bit":
                macro("invGainLong", f1(r["psnr"] - o["psnr"]))
                macro("invShortEpochs", str(inv1.get("epochs", 15)))
        j = inv.get("joint", {}).get(name)
        if j:
            macro(f"inv{tag}Joint", f1(j["psnr"]))
            macro(f"inv{tag}JointReid", f1(j["reid"]["top1"]) if "reid" in j else "--")
    macro("invMean", f1(inv["mean_image"]["psnr"]))
    macro("invNN", f1(inv["nearest_neighbour"]["psnr"]))
    if "mean_colour_oracle" in inv:
        mo = inv["mean_colour_oracle"]
        macro("invColour", f1(mo["psnr"]))
        macro("invColourReid", f1(mo["reid"]["top1"]))
        if "reid_loc" in mo:
            macro("invColourReidLoc", f1(mo["reid_loc"]["top1"]))
            macro("invColourReidLocTen", f1(mo["reid_loc"]["top10"]))
            macro("invColourTrainMse", f"{mo['train_mse']:.4f}")
            macro("invNLoc", f"{mo['reid_loc']['n_locations']:,}".replace(",", "{,}"))
            macro("invChanceLoc", f"{100 / mo['reid_loc']['n_locations']:.2f}")
            macro("invChanceLocTen", f"{1000 / mo['reid_loc']['n_locations']:.1f}")
            macro("invEpochs", str(inv["epochs"]))
if isinstance(e4, list):
    for r in e4:
        tag = {D: "Full", 64: "Sf", 16: "St"}[r["d"]] + {None: "None", 64: "Ssf", 8: "Eight", 1: "One"}[r["eps_upload"]]
        macro(f"dp{tag}", f1(r["acc"]))
if INV_NPZ and inv:
    z = INV_NPZ
    cols = [("original", "original"), ("mean colour", None), ("nearest_neighbour", "nearest\ntraining image"), ("single__full_16-bit", "single upload,\nfull"),
            ("single__PCA-64", "single upload,\nPCA-64"), ("single__DP_eps8", "single upload,\nclip+noise $\\varepsilon$=8"),
            ("single__CLIP-L_full_16-bit", "single upload,\nCLIP-L")]
    cols = [(k, t) for k, t in cols if k == "mean colour" or k in z]
    ex = [0, 1, 2, 3]
    GAIN = 3.0
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

# ------------------------------------------------------------------ generic public decoder (stage 20)
gi = load("generic_inversion.json") or {}
GI_NPZ = np.load(os.path.join(RES, "generic_inversion_examples.npz")) if os.path.exists(os.path.join(RES, "generic_inversion_examples.npz")) else None


def gi_macros(tag, r):
    if not r:
        return
    macro(f"gi{tag}Reid", f1(r["reid"]["top1"]))
    macro(f"gi{tag}ReidTen", f1(r["reid"]["top10"]))
    macro(f"gi{tag}MedRank", f"{r['reid']['median_rank']:.0f}")
    macro(f"gi{tag}PixReid", f1(r["reid_pix"]["top1"]) if "reid_pix" in r else "--")
    macro(f"gi{tag}Psnr", f1(r["psnr"]))
    lab = r.get("labels", {})
    for k, t in (("recon", "Lab"), ("real", "LabReal")):
        if k in lab:
            macro(f"gi{tag}{t}Acc", f1(lab[k]["acc"]))
            macro(f"gi{tag}{t}Fone", f1(lab[k]["f1"]))


v = gi.get("vb", {})
macro("giVbChanceFone", f1(100 / 4))  # label-independent guessing with the class priors: macro F1 = 1 / number of classes
macro("giMilkChanceFone", f1(100 / 7))
if v:
    gi_macros("Vb", v.get("paper upload"))
    gi_macros("VbClip", v.get("CLIP preprocessing"))
    macro("giVbN", str(v["n_attacked"]))
    macro("giVbNTest", f"{v['n_test']:,}".replace(",", "{,}"))
    macro("giVbChance", f"{100 / v['n_test']:.2f}")
    macro("giVbChanceTen", f"{1000 / v['n_test']:.1f}")
    macro("giVbCheck", f"{v.get('check_cos_pooled', float('nan')):.3f}")
    macro("giLink", f1(v["linkage_clip"]["top1"]))
    if "mean_colour" in v:
        macro("giVbColourPixReid", f1(v["mean_colour"]["reid_pix"]["top1"]))
    if "real_label_inference" in v:  # classifier on real test images of the party (upper reference)
        macro("giVbRealAcc", f1(v["real_label_inference"]["recon"]["acc"]))
        macro("giVbRealFone", f1(v["real_label_inference"]["recon"]["f1"]))
    macro("giSecs", f"{v['paper upload']['secs'] / v['n_attacked']:.1f}" if "paper upload" in v else "--")
m = gi.get("milk", {})
for kind, tag in (("clin", "Clin"), ("derm", "Derm")):
    o = m.get(kind)
    if not o:
        continue
    gi_macros(f"Milk{tag}", o.get("upload"))
    if "mean_colour" in o:
        macro(f"giMilk{tag}ColourReid", f1(o["mean_colour"]["reid"]["top1"]))
    if "real_label_inference" in o:
        macro(f"giMilk{tag}RealFone", f1(o["real_label_inference"]["recon"]["f1"]))
        macro(f"giMilk{tag}RealAcc", f1(o["real_label_inference"]["recon"]["acc"]))
    macro("giMilkN", str(o["n_attacked"]))
    macro("giMilkNGal", f"{o['n_gallery']:,}".replace(",", "{,}"))
    macro("giMilkChance", f"{100 / o['n_gallery']:.2f}")
    if "check_cos_pooled" in o:
        macro("giMilkCheck", f"{o['check_cos_pooled']:.3f}")
d = gi.get("defences", {})
if d.get("utility"):
    U, A, DC = d["utility"], d.get("attacks", {}), d.get("decoded", {})
    macro("giCosMean", f"{d.get('cos_mean_embedding', float('nan')):.2f}")
    NAMES = [("plain", "none (raw CLIP-L)"), ("rotation", "secret rotation"), ("PCA-64", "private PCA-64"), ("autoencoder-100", "autoencoder, 100"),
             ("noise-as-targets-100", "noise as targets, 100"), ("one-layer split learning-100", "split learning, 1 layer, 100")]
    ks = sorted({int(k[2:]) for a in A.values() for k in a if k[2:].isdigit()})
    rows = []
    for key, lab in NAMES:
        u, a = U.get(key), A.get(key, {})
        if not u:
            continue
        cells = [str(u["dim"]), pm(u)]
        for k in ks:
            c = a.get(f"k={k}")
            if key == "plain":  # no map to learn: the public projection gives the embedding itself
                cells.append(f1(v["linkage_clip"]["top1"]) if (k == 0 and v) else "--")
            else:
                cells.append(f1(c["linkage"]["top1"]) if c else "--")
        dk = [x for x in DC if x.startswith(key + "|")]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        t = {"plain": "Plain", "rotation": "Rot", "PCA-64": "Pca", "autoencoder-100": "Ae", "noise-as-targets-100": "Nat", "one-layer split learning-100": "Split"}[key]
        macro(f"gd{t}Acc", f1(u["acc"]))
        for k in ks:
            c = a.get(f"k={k}")
            if c:
                macro(f"gd{t}Link{k}", f1(c["linkage"]["top1"]))
                macro(f"gd{t}Cos{k}", f"{c['cos']:.2f}")
        for x in dk:
            macro(f"gd{t}Dec{x.split('=')[1].title()}", f1(DC[x]["reid"]["top1"]))
            if "labels" in DC[x]:
                macro(f"gd{t}Dec{x.split('=')[1].title()}Lab", f1(DC[x]["labels"]["recon"]["acc"]))
    macro("gdKs", ", ".join(str(k) for k in ks if k))
    for key, t in (("rotation", "Rot"), ("PCA-64", "Pca"), ("autoencoder-100", "Ae"), ("noise-as-targets-100", "Nat"), ("one-layer split learning-100", "Split")):
        if key in U and "plain" in U:
            macro(f"gd{t}Drop", f1(U["plain"]["acc"] - U[key]["acc"]))
    macro("gdNDec", str(DC.get("n", 0)))
    sp = U.get("one-layer split learning-100", {})
    if "epochs" in sp:
        macro("gdSplitEpochs", f"{np.mean(sp['epochs']):.0f}")
    write("tab_defences.tex", "\n".join(rows))
if GI_NPZ is not None:
    z = GI_NPZ

    def disp(img, stretch):
        x = np.transpose(img, (1, 2, 0)).astype(float) / 255
        if stretch:  # per image and channel, 2nd-98th percentile (dark Sentinel-2 L1C composites)
            lo, hi = np.percentile(x, 2, axis=(0, 1)), np.percentile(x, 98, axis=(0, 1))
            x = (x - lo) / np.maximum(hi - lo, 1e-3)
        return np.clip(x, 0, 1)

    blocks = [("vb__original", "vb__paper_upload", True, "VertiBench (one party)"), ("milk__clin__original", "milk__clin__upload", False, "MILK10k clinical"),
              ("milk__derm__original", "milk__derm__upload", False, "MILK10k dermoscopic")]
    blocks = [b for b in blocks if b[0] in z.files and b[1] in z.files]
    if blocks:
        ne = 4
        fig, axs = plt.subplots(2, ne * len(blocks), figsize=(6.6, 1.38), squeeze=False)
        for bi, (ko, kr, st, title) in enumerate(blocks):
            for e in range(ne):
                j = bi * ne + e
                axs[0, j].imshow(disp(z[ko][e], st))
                axs[1, j].imshow(disp(z[kr][e], False))
                for i in (0, 1):
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    for sp_ in axs[i, j].spines.values():
                        sp_.set_visible(False)
            axs[0, bi * ne].set_title(title, fontsize=6, loc="left")
        axs[0, 0].set_ylabel("original", fontsize=6)
        axs[1, 0].set_ylabel("decoded", fontsize=6)
        fig.subplots_adjust(left=0.025, right=0.998, top=0.88, bottom=0.005, wspace=0.04, hspace=0.04)
        fig.savefig(os.path.join(OUT, "fig_generic.pdf"))
        plt.close(fig)

# ------------------------------------------------------------------ domain-adapted generic decoder (stage 21)
gf = load("generic_inversion_ft.json") or {}
if gf:
    for key, tag in (("off-the-shelf (256 px)", "VbOff"), ("fine-tuned", "VbFt"), ("fine-tuned, best of 4", "VbBest"), ("nearest public image", "VbNn")):
        gi_macros(tag, gf.get(key))
    for key, tag in (("off-the-shelf (256 px)", "Off"), ("fine-tuned", "Ft"), ("fine-tuned, best of 4", "Best")):
        if "clip_cos" in gf.get(key, {}):
            macro(f"gift{tag}Clip", f"{gf[key]['clip_cos']:.2f}")
    if "nearest public image" in gf:
        macro("giftNnLabMatch", f1(gf["nearest public image"]["label_match"]))
    ft = gf.get("finetune", {})
    macro("giftPairs", f"{gf['n_public_images']:,}".replace(",", "{,}"))
    macro("giftLocs", f"{gf['n_public_images'] // 16:,}".replace(",", "{,}"))
    macro("giftEncMin", f"{gf['encode_minutes']:.0f}")
    macro("giftSteps", f"{ft.get('steps', 0):,}".replace(",", "{,}"))
    macro("giftMinutes", f"{ft.get('minutes', 0):.0f}")
    macro("giftSize", str(gf["size"]))
    macro("giftNBest", str(gf["n_best"]))
    macro("giftGuid", f"{gf['guidance']:g}")
    macro("giftGuidGrid", ", ".join(f"{float(g):g}" for g in gf.get("guidance_val", {})))
GF_NPZ = os.path.join(RES, "generic_inversion_ft_examples.npz")
if os.path.exists(GF_NPZ) and GI_NPZ is not None and "vb__paper_upload" in GI_NPZ.files:
    zf = np.load(GF_NPZ)
    rows_ = [("original", zf["original"], True), ("generic\ndecoder", GI_NPZ["vb__paper_upload"], False),
             (f"fine-tuned,\nbest of {gf.get('n_best', 4)}", zf["fine_tuned_best"], True), ("nearest\npublic image", zf["nearest_public"], True)]
    ne = min(8, len(zf["original"]))
    fig, axs = plt.subplots(len(rows_), ne, figsize=(6.6, 0.84 * len(rows_) * 8 / ne * 0.98), squeeze=False)
    for i, (lab, arr, st) in enumerate(rows_):
        for j in range(ne):
            axs[i, j].imshow(disp(arr[j], st))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            for sp_ in axs[i, j].spines.values():
                sp_.set_visible(False)
        axs[i, 0].set_ylabel(lab, fontsize=6)
    fig.subplots_adjust(left=0.045, right=0.998, top=0.998, bottom=0.002, wspace=0.03, hspace=0.03)
    fig.savefig(os.path.join(OUT, "fig_generic_ft.pdf"))
    plt.close(fig)

# ------------------------------------------------------------------ linear / encryption-friendly heads
HEDS = [("vertibench/dinov2_g_224", "VertiBench, DINOv2-g", 16), ("vertibench/clip_l14_224", "VertiBench, CLIP-L", 16),
        ("milk10k/dinov2_g", "MILK10k, DINOv2-g", 2), ("milk10k/clip_l14", "MILK10k, CLIP-L", 2),
        ("worldstrat_ipcc_1.5m_S2MS", "WorldStrat IPCC, DINOv2-g", 2), ("cifar10_dinov2_g_topbottom", "CIFAR-10 halves, DINOv2-g", 2),
        ("cifar100_dinov2_g_topbottom", "CIFAR-100 halves, DINOv2-g", 2)]
rows = []
# ridge with the party scaling (per coordinate or per block) chosen on validation, against the per-coordinate ridge of the table
_sd = [abs(c["ridge_sel"][he[t].get("metric", "acc")] - c["ridge"][he[t].get("metric", "acc")])
       for t in he if t != "he_cost" and isinstance(he[t], dict) for c in he[t].values() if isinstance(c, dict) and "ridge_sel" in c and "ridge" in c]
if _sd:
    macro("heSelMax", f1(max(_sd)))
    macro("heSelN", str(len(_sd)))
for key, lab, k in HEDS:
    r = he.get(key)
    if not r:
        continue
    met = "f1" if r["metric"] == "f1" else "acc"
    full = r.get("full", {})
    p256, p64, p16 = r.get("pca256", {}), r.get("pca64", {}), r.get("pca16", {})
    cells = [pm(full.get("mlp2"), met), pm(full.get("ridge"), met), pm(p256.get("ridge"), met), pm(p64.get("ridge"), met), pm(p16.get("ridge"), met)]
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    t = key.replace("/", "").replace("_", "").replace("224", "").replace("dinov2g", "G").replace("clipl14", "C").replace("vertibench", "Vb").replace("milk10k", "Milk")
    t = "".join(ch for ch in t if ch.isalpha())[:14]
    for h, hd, tag in (("mlp2", full, "Mlp"), ("linear_softmax", full, "Lin"), ("ridge", p64, "RidgeSf"), ("polylogistic", p64, "PolySf"), ("mlp2", p64, "MlpSf"),
                       ("ridge", p16, "RidgeSt"), ("ridge", p256, "RidgeTf"), ("polylogistic", p256, "PolyTf"), ("mlp2", p256, "MlpTf"), ("ridge", full, "RidgeFull")):
        s = hd.get(h)
        if s:
            macro(f"he{t}{tag}", f1(s[met]))
write("tab_he.tex", "\n".join(rows))
_c = (he.get("cifar100_dinov2_g_topbottom") or {}).get("pca64", {})  # CIFAR-100, PCA-64: squared loss vs linear softmax
for h, tag in (("linear_softmax", "Lin"), ("mlp2", "Mlp"), ("ridge", "Ridge")):
    if h in _c:
        macro(f"heCHundredSf{tag}", f1(_c[h]["acc"]))
hc = he.get("he_cost", {})
for key, tag in (("milk10k/dinov2_g/pca64", "Milk"), ("vertibench/dinov2_g_224/pca16", "VbSt"), ("vertibench/dinov2_g_224/pca64", "VbSf")):
    c = hc.get(key)
    if c:
        macro(f"hc{tag}Server", f"{c['extrapolated_secs_server'] / 60:.0f}")
        macro(f"hc{tag}Enc", f"{c['extrapolated_secs_encrypt_per_party']:.1f}")
        macro(f"hc{tag}Up", size(c["upload_bytes_per_party"]))
        macro(f"hc{tag}Plain", size(c["plain_fp16_bytes_per_party"]))
        macro(f"hc{tag}Err", f"{c['max_rel_error']:.0e}")
        macro(f"hc{tag}CtCt", f"{c['secs_ctct_inner_product'] * 1000:.0f}")
        macro(f"hc{tag}ServerH", f"{c['extrapolated_secs_server'] / 3600:.1f}")
        macro(f"hc{tag}Expand", f"{c['upload_bytes_per_party'] / c['plain_fp16_bytes_per_party']:.0f}")
        macro(f"hc{tag}Agg", size(c["aggregate_bytes_round_trip"]))
        macro(f"hc{tag}N", f"{c['N']:,}".replace(",", "{,}"))
        if tag == "Milk":  # class-weighted ridge: every party also computes its own weighted Gram block, X_k^T diag(Enc w) X_k,
            # with d(d+1)/2 ciphertext-plaintext inner products, and uploads its weighted columns too (twice the upload)
            K, d0, tp, tc = c["parties"], c["d_per_party"], c["secs_ptct_inner_product"], c["secs_ctct_inner_product"]
            macro("hcMilkServer", f"{(c['extrapolated_secs_server'] + K * d0 * (d0 + 1) / 2 * tp) / 60:.0f}")
            macro("hcMilkUp", size(2 * c["upload_bytes_per_party"]))
            nf = K * (K - 1) // 2 * 1536**2
            unw = nf * tc + K * 1536 * c["C"] * tp
            macro("hcMilkFullUnwH", f"{unw / 3600:.0f}")
            macro("hcMilkFullH", f"{(unw + K * 1536 * 1537 / 2 * tp) / 3600:.0f}")
            macro("hcMilkFullUp", size(2 * 1536 * c["chunks"] * c["ciphertext_bytes"]))

# ------------------------------------------------------------------ third review: audit of the VertiBench split and of the comparisons
if ana.get("mapping"):
    mp, sb = ana["mapping"], ana["sibling"]
    macro("auMapTrain", f"{mp['train_unique']:,}".replace(",", "{,}"))
    macro("auMapTest", str(mp["test_unique"]))
    macro("auMapOverlap", str(mp["train_test_same_area"]))
    macro("auScore", f"{mp['score_p5']:.3f}")
    macro("auGap", f"{mp['gap_p5']:.2f}")
    macro("auSibN", str(sb["test_with_train_sibling"]))
    macro("auSibShare", f1(sb["share"]))
    macro("auNoSibN", str(mp["n_test"] - sb["test_with_train_sibling"]))
AU = {"frozen clip_l14_224__mlp2": "Clip", "frozen dinov2_g_224__mlp2": "G", "frozen clip_b16_224__mlp2": "ClipB",
      "frozen clip_l14_224__logits16": "LgClip", "frozen dinov2_g_224__logits16": "LgG", "frozen clip_b16_224__logits16": "LgClipB",
      "SplitNN VertiBench recipe": "Sn", "split CLIP-B clip_b16_lora": "Lora", "split CLIP-B clip_b16_frozen": "SplitFrozen"}
for k, t in AU.items():
    e = ana.get("subsets", {}).get(k)
    if not e:
        continue
    macro(f"au{t}All", f1(e["all"]["acc"]))
    if "sibling" in e:
        macro(f"au{t}Sib", f1(e["sibling"]["acc"]))
        macro(f"au{t}NoSib", f1(e["no_sibling"]["acc"]))
        macro(f"au{t}SibDrop", f1(e["sibling"]["acc"] - e["no_sibling"]["acc"]))
    p = ana.get("paired", {}).get(k)
    if p:
        macro(f"pairSn{t}N", str(p["seeds"]))
        for sub, tag in (("all", ""), ("no_sibling", "NoSib")):
            if sub in p:
                macro(f"pairSn{t}{tag}Diff", f1(p[sub]["diff"]))
                macro(f"pairSn{t}{tag}Lo", f1(p[sub]["lo"]))
                macro(f"pairSn{t}{tag}Hi", f1(p[sub]["hi"]))
                pv = p[sub]["mcnemar_p"]
                macro(f"pairSn{t}{tag}P", f"{{=}}{pv:.3f}" if pv >= 0.001 else "{<}0.001")
# paired tests compare 3-seed ensembles (seed-averaged probabilities), the tables report means over seeds: give both,
# plus the direct LoRA split learning vs one-shot CLIP-B head comparison (same backbone, same seeds)
_pf = {k: os.path.join(RES, f) for k, f in (("fr", "vb_frozen_probs.npz"), ("sp", "vb_split_clipb_probs.npz"), ("sn", "vb_repro_splitresnet_preds_probs.npz"))}
if all(os.path.exists(f) for f in _pf.values()):
    from s23_audit_analysis import paired as _paired

    _fr, _sp, _sn = (np.load(_pf[k]) for k in ("fr", "sp", "sn"))
    _y = _fr["y"]
    _P = {"Clip": _fr["clip_l14_224__mlp2"], "G": _fr["dinov2_g_224__mlp2"], "ClipB": _fr["clip_b16_224__mlp2"],
          "LgClip": _fr["clip_l14_224__logits16"], "LgG": _fr["dinov2_g_224__logits16"], "LgClipB": _fr["clip_b16_224__logits16"],
          "Sn": np.stack([_sn[f"seed{i}"] for i in range(3)]),
          "Lora": np.stack([_sp[f"clip_b16_lora__seed{i}"] for i in range(3)]),
          "SplitFrozen": np.stack([_sp[f"clip_b16_frozen__seed{i}"] for i in range(3)])}
    for t, P in _P.items():
        macro(f"ens{t}Acc", f1((P[:3].mean(0).argmax(1) == _y).mean() * 100))  # 3-seed ensemble accuracy
        acc = (P.argmax(2) == _y[None]).mean(1) * 100
        macro(f"seed{t}Min", f1(acc.min()))
        macro(f"seed{t}Max", f1(acc.max()))
    _q = _paired(_P["Lora"].mean(0), _P["ClipB"][:3].mean(0), _y)
    macro("pairLoraHeadDiff", f1(_q["diff"]))
    macro("pairLoraHeadLo", f1(_q["lo"]))
    macro("pairLoraHeadHi", f1(_q["hi"]))
    macro("pairLoraHeadP", f"{{=}}{_q['mcnemar_p']:.3f}" if _q["mcnemar_p"] >= 0.001 else "{<}0.001")
    # Holm over the eight comparisons with the reproduced SplitNN (all test images)
    _ps = sorted((ana["paired"][k]["all"]["mcnemar_p"], t) for k, t in AU.items() if k in ana.get("paired", {}))
    _m, _surv, _stop = len(_ps), [], False
    for i, (pv, t) in enumerate(_ps):
        if _stop or pv * (_m - i) >= 0.05:
            _stop = True
            continue
        _surv.append(t)
    macro("holmN", str(_m))
    macro("holmSurvive", str(len(_surv)))
if ana.get("nnpublic"):
    n = ana["nnpublic"]
    for key, t in (("label_match_all", "All"), ("label_match_sibling", "Sib"), ("label_match_no_sibling", "NoSib"), ("nn_is_sibling_given_sibling", "IsSib")):
        if key in n:
            macro(f"nnPub{t}", f1(n[key]))
if aud.get("nnpublic"):
    macro("nnPubAuditAll", f1(aud["nnpublic"]["label_match_all"]))
lk = aud.get("linkage")
if lk:
    macro("lkN", f"{lk['n_images']:,}".replace(",", "{,}"))
    macro("lkDup", str(lk["duplicate_uploads"]))
    for k, t in (("same preprocessing", "Same"), ("bicubic resize", "Bicubic"), ("8-bit resized image", "Eightbit"), ("CLIP preprocessing", "Clip")):
        if k in lk:
            macro(f"lk{t}", f1(lk[k]["top1"]))
            macro(f"lk{t}Party", f1(lk[k]["top1_within_party_min"]))
            macro(f"lk{t}Ties", str(lk[k]["ties"]))
            macro(f"lk{t}Cos", f"{lk[k]['cos_true_min']:.3f}")
# MILK10k: the same split-learning recipe with frozen and LoRA bottoms (paired by fold)
for b, t in (("dinov2_b", "B"), ("clip_b16", "Cb")):
    fr, lo = msplit2.get(f"{b}_frozen"), msplit2.get(f"{b}_lora")
    if fr and lo:
        d = np.array(lo["f1s"]) - np.array(fr["f1s"])
        macro(f"srFrozen{t}", f1(fr["f1"]))
        macro(f"srLora{t}", f1(lo["f1"]))
        macro(f"srGap{t}", f1(d.mean()))
        macro(f"srGapSd{t}", f1(d.std(ddof=1)))
        macro(f"srWins{t}", str(int((d > 0).sum())))
        fz = (milk or {}).get(b, {}).get("clin+derm|mlp2|bal")
        if fz:
            macro(f"srVsHead{t}", f"{fr['f1'] - fz['f1']:+.1f}")  # split recipe vs the one-shot head; use in math mode
if msplit2.get("dinov2_b_ft"):
    macro("srFt", f1(msplit2["dinov2_b_ft"]["f1"]))
# WorldStrat per-class F1 (grouped out-of-fold predictions)
wsf = ana.get("ws_f1", {})
for task in ("ipcc", "lccs"):
    e = wsf.get(task)
    if not e or not e.get("classes"):
        continue
    cfg = e["configs"]
    a, bb = cfg.get("SPOT 1.5m|mlp2"), cfg.get("SPOT 1.5m + S2 MS|mlp2")
    if a and bb:
        gains = sorted(zip(np.array(bb) - np.array(a), e["classes"], e["counts"]), reverse=True)
        tex = lambda c: c.replace("\u2010", "-").replace("%", "\\%").replace("&", "\\&")  # noqa: E731
        macro(f"wc{task.capitalize()}Top", ", ".join(f"{tex(c)} ({g:+.0f}, {n} areas)" for g, c, n in gains[:3]))
        macro(f"wc{task.capitalize()}Neg", str(int(sum(g < 0 for g, _, _ in gains))))

# ------------------------------------------------------------------ placeholders, so that the paper compiles on partial results
for t in ("tab_main", "tab_encoders", "tab_combo_enc", "tab_classic", "tab_sun", "tab_milk", "tab_milkfusion", "tab_milkreg", "tab_freeze", "tab_ws", "tab_wsfusion",
          "tab_pareto", "tab_privacy", "tab_he", "tab_defences"):
    if not os.path.exists(os.path.join(OUT, t + ".tex")):
        write(t + ".tex", "\\multicolumn{2}{@{}l}{(pending)}\\\\")

# ------------------------------------------------------------------ macros
with open(os.path.join(OUT, "macros.tex"), "w") as f:
    for k, v in sorted(MACROS.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(MACROS), "macros")
