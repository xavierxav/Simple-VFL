"""Central figure: the Pareto frontier of image VFL (bytes exchanged per sample against accuracy / macro F1).

usage: python make_frontier.py [results_dir] [out_dir]
Reads the result JSONs of paper_results/ with the loading code and the cost accounting of make_assets2.py (copied here,
not imported: importing make_assets2 would rewrite every table), so that every number matches the generated tables
(tab_main, tab_encoders, tab_combo_enc, tab_pareto, tab_privacy, tab_he, tab_freeze) and macros.tex; check_tables()
verifies it on every run.  Writes, in out_dir:
  frontier_points.csv   every method as a point (all fields, 'source' = JSON file and key)
  fig_frontier.pdf/png  VertiBench Satellite (left) and MILK10k (right)
  macros_frontier.tex   a few macros, all prefixed with 'fr'
Bytes per sample are summed over all parties and both directions, half precision unless stated:
  one-shot      the upload (K x d x bits / 8)
  split / logits  the training traffic (activations or logits up, their gradients down, every training epoch), as in
                the tables; frozen bottoms need no gradients (one augmented upload per epoch)
  HE (CKKS)     the encrypted upload; the O(D^2) aggregates of the decryption round trip are not counted (they are
                listed in the CSV where they were measured)
Pareto-optimal: no other point is at least as cheap and at least as good, with one of the two strictly (means only).
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "paper_results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/src/gen"
os.makedirs(OUT, exist_ok=True)
# style of make_assets2.py
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
C = {"blue": "#2a6fbb", "orange": "#d9772b", "green": "#3a9a5b", "red": "#c0392b", "grey": "#7f8c8d", "purple": "#8e5bb5"}
KiB, MiB = 1024, 1024 * 1024


# ------------------------------------------------------------------ helpers, identical to make_assets2.py
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


DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"], strict=True))


# ------------------------------------------------------------------ inputs, as in make_assets2.py
rep = load("vb_repro_splitresnet.json")
enc = load("vb224_encoders.json") or {}
heads224 = enc.get("heads", {})
cost = enc.get("cost", {})
Eg = load("vb224_controls_dinov2_g.json") or {}
if (load("vb224_controls_dinov2_g_dp.json") or {}).get("E4_dp"):  # third review: clipping norm per party
    Eg["E4_dp"] = load("vb224_controls_dinov2_g_dp.json")["E4_dp"]
Eold = load("vb_controls.json") or {}
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
milk = load("milk_heads.json") or {}
msplit = load("milk_split.json") or {}
msplit2 = load("milk_split_v2.json") or {}  # frozen bottoms under the split-learning recipe
mreg = load("milk_reg.json") or {}
aud = load("vb_audit.json") or {}
clipb = load("vb_split_clipb.json") or {}
combo = load("vb224_combo.json") or {}
base = {k: load(f"baseline_{k}.json") for k in ["splitnn_all", "splitnn_rgb", "splitnn_rgb_pt", "ssl_all", "ssl_rgb", "nat_all"]}
K, N_TR = 16, 3142
N_TR_H, N_VA_H, N_TE = int(N_TR * 0.85), N_TR - int(N_TR * 0.85), 785
D = 1536

NAME = {"dinov2_g_224": "DINOv2-g", "dinov2_l_224": "DINOv2-L", "dinov2_b_224": "DINOv2-B", "dinov2_s_224": "DINOv2-S", "clip_l14_224": "CLIP-L",
        "clip_b16_224": "CLIP-B", "siglip_b16_224": "SigLIP-B", "resnet50_in1k_224": "ResNet-50 (ImageNet, supervised)",
        "dinov2_b_random_224": "ViT-B (random weights)", "ssl4eo_s2_vits16": "SSL4EO-S12 ViT-S/16 (13 bands)",
        "dinov2_g": "DINOv2-g", "clip_l14": "CLIP-L", "dinov2_b": "DINOv2-B", "clip_b16": "CLIP-B"}

# ------------------------------------------------------------------ families: (label, marker, colour, uses foundation)
FAM = {
    "fm_oneshot": ("one-shot, frozen", "o", C["blue"], True),
    "fm_pca": ("one-shot, PCA + quantised", "o", C["green"], True),
    "fm_logits": ("4-logit head, few rounds", "D", C["purple"], True),
    "fm_split_frozen": ("split, frozen bottoms", "v", C["orange"], True),
    "fm_split_lora": ("split, LoRA", "^", C["orange"], True),
    "fm_split_ft": ("split, full fine-tuning", "p", C["red"], True),
    "fm_he": ("HE linear head (CKKS)", "P", "#222222", True),
    "fm_dp": ("DP upload", "X", "#8c6d31", True),
    "nf_split": ("split, ResNet-18", "s", C["grey"], False),
    "nf_ssl": ("one-shot SSL (FedOnce)", "h", C["grey"], False),
    "nf_frozen": ("one-shot, frozen ResNet-50 / random ViT", "o", C["grey"], False),
}
FAM_LONG = {"fm_oneshot": "foundation one-shot frozen", "fm_pca": "foundation one-shot + PCA-k x bits", "fm_logits": "foundation few-round 4-logit",
            "fm_split_frozen": "foundation split learning frozen bottoms", "fm_split_lora": "foundation split LoRA",
            "fm_split_ft": "foundation split full fine-tuning", "fm_he": "HE head (foundation features)", "fm_dp": "DP upload (foundation features)",
            "nf_split": "non-foundation split learning (scratch or ImageNet ResNet)", "nf_ssl": "non-foundation one-shot self-supervised (FedOnce)",
            "nf_frozen": "non-foundation one-shot frozen (ImageNet ResNet-50 / random ViT)"}

P = []  # all points


def add(bench, name, fam, s, key, nbytes, rounds, backprop, sees, source, *, table=None, macros=(), cost_note="", encoder="", in_set=True,
        val_selected="", mean=None, std=None, n=None, extra_bytes=None, label=None):
    """One method = one point.  s: summary dict of the results (mean, per-seed values or std), key: 'acc' or 'f1'."""
    m = s[key] if s is not None else mean
    e = sd(s, key) if s is not None else std
    nn = (s.get("n") or len(s.get(key + "s", [])) or None) if s is not None else n
    P.append(dict(benchmark=bench, name=name, family=FAM_LONG[fam], fam=fam, uses_foundation=FAM[fam][3], encoder=encoder,
                  metric="test accuracy (%)" if key == "acc" else "macro F1 (%)", metric_mean=float(m), metric_std=float(e), n_runs=nn,
                  bytes_per_sample=float(nbytes), bytes_human=size(nbytes).replace("\\,", " "), rounds=rounds, party_backprop=backprop,
                  party_backprop_bool=backprop != "none", server_sees=sees, val_selected=val_selected, in_pareto_set=in_set,
                  cost_note=cost_note, aggregate_bytes_per_sample_not_counted=extra_bytes, source=source, table=table or "", label=label or name,
                  _check=(table, pm(s, key) if s is not None else f"{m:.1f}\\,{{\\scriptsize$\\pm${e:.1f}}}", size(nbytes)), _macros=macros))


VB, MK = "VertiBench Satellite", "MILK10k"
SN_TEXT = "4 x K x 512 x epochs (512-d fp16 activations up, gradients down, every epoch), as tab_main"

# ------------------------------------------------------------------ VertiBench: split learning (tab_main)
add(VB, "SplitNN, published (ResNet-18, 13 bands)", "nf_split", None, "acc", 4 * K * 512 * 50, 50, "full ResNet-18 bottom", "activations + gradients",
    "make_assets2.py (hard-coded, wu2024vertibench)", table="tab_main.tex", mean=81.2, std=0.4, label="SplitNN (published)",
    cost_note=SN_TEXT + "; literature number, left out of the Pareto set and the figure (our reproduction below is plotted)", in_set=False)
if rep:
    add(VB, "SplitNN, VertiBench recipe (ResNet-18, 13 bands)", "nf_split", rep["last_epoch"], "acc", 4 * K * 512 * 50, 50, "full ResNet-18 bottom",
        "activations + gradients", "vb_repro_splitresnet.json:last_epoch", table="tab_main.tex", cost_note=SN_TEXT, label="SplitNN")
for k, lab in (("splitnn_all", "SplitNN, our recipe, scratch (13 bands)"), ("splitnn_rgb", "SplitNN, our recipe, scratch (RGB)"),
               ("splitnn_rgb_pt", "SplitNN, our recipe, ImageNet init. (RGB)")):
    b = base[k]
    if b:
        add(VB, lab, "nf_split", b["summary"], "acc", 4 * K * 512 * b["epochs"], b["epochs"], "full ResNet-18 bottom", "activations + gradients",
            f"baseline_{k}.json:summary", table="tab_main.tex", cost_note=SN_TEXT)
for k, fam in (("clip_b16_lora", "fm_split_lora"), ("clip_b16_frozen", "fm_split_frozen")):
    s = clipb.get(k, {})
    if not s.get("summary"):
        continue
    E = s["infos"][0]["epochs"]
    tr_bytes = np.mean([i["bytes_per_train_sample"] for i in s["infos"]]) * 2 * E * N_TR_H / (2 * E * N_TR_H + E * N_VA_H + N_TE)
    if k.endswith("frozen"):  # frozen bottoms need no gradients: one augmented upload per epoch
        tr_bytes /= 2
    add(VB, f"CLIP-B bottoms, {'LoRA' if 'lora' in k else 'frozen'} (split)", fam, s["summary"], "acc", tr_bytes, E,
        "LoRA adapters (r=16)" if "lora" in k else "none", "activations + gradients" if "lora" in k else "activations (augmented views)",
        f"vb_split_clipb.json:{k}.summary", table="tab_main.tex", encoder="CLIP ViT-B/16",
        cost_note="mean(bytes_per_train_sample) x training share of the passes, as tab_main" + ("; halved: no gradients" if "frozen" in k else ""))
# few rounds: four logits per party (tab_main)
for kk, v in aud.get("heads", {}).items():
    k, h = kk.split("|")
    if not h.startswith("logits"):
        continue
    up = np.mean(v["epochs_run"]) * 2 * K * 4 * 2  # training passes only, as for split learning: 4 logits up, 4 gradients down, fp16
    add(VB, f"{NAME[k]}, 4-logit MLP per party", "fm_logits", v, "acc", up, float(np.mean(v["epochs_run"])), "party MLP head on frozen features",
        "logits + gradients", f"vb_audit.json:heads.{kk}", table="tab_main.tex", encoder=NAME[k],
        cost_note="mean(epochs_run) x 2 x K x 4 x 2 (4 fp16 logits up, 4 gradients down per epoch), as tab_main")
# one-shot, party-side self-supervised training (tab_main)
for k, lab in (("nat_all", "FedOnce, Noise As Targets (13 bands)"), ("ssl_all", "FedOnce, SimCLR (13 bands)"), ("ssl_rgb", "FedOnce, SimCLR (RGB)")):
    b = base[k]
    if b:
        add(VB, lab, "nf_ssl", b["summary"], "acc", K * b.get("feature_dim", 512) * 2, 1, "local SSL pre-training (no interaction)",
            "plain features (local SSL encoder)", f"baseline_{k}.json:summary", table="tab_main.tex", cost_note="K x feature_dim x 2 (fp16), as tab_main")
# one-shot, frozen features (tab_main, tab_encoders)
for k in ("dinov2_g_224", "clip_l14_224", "dinov2_l_224", "dinov2_b_224", "dinov2_s_224", "clip_b16_224", "siglip_b16_224", "resnet50_in1k_224",
          "dinov2_b_random_224", "ssl4eo_s2_vits16"):
    h = heads224.get(k)
    src = f"vb224_encoders.json:heads.{k}.mlp2"
    if k == "ssl4eo_s2_vits16" and not h and isinstance(Eold.get("E6_ssl4eo"), dict):
        h, src = {"dim": Eold["E6_ssl4eo"]["dim"], "mlp2": Eold["E6_ssl4eo"]["mlp2"]}, "vb_controls.json:E6_ssl4eo.mlp2"
    if not h:
        continue
    nf = k in ("resnet50_in1k_224", "dinov2_b_random_224")
    add(VB, f"{NAME[k]}, frozen, one upload", "nf_frozen" if nf else "fm_oneshot", h["mlp2"], "acc", K * h["dim"] * 2, 1, "none", "plain features", src,
        table="tab_main.tex" if k in ("dinov2_g_224", "clip_l14_224", "dinov2_b_224", "clip_b16_224", "ssl4eo_s2_vits16") else "tab_encoders.tex",
        encoder=NAME[k], cost_note="K x dim x 2 (fp16), as tab_main")
for k, v in combo.items():  # two encoders per party (tab_combo_enc)
    names = k.split("+")
    if len(names) < 2:
        continue
    add(VB, " + ".join(NAME[n] for n in names) + ", frozen, one upload", "fm_oneshot", v["mlp2"], "acc", K * v["dim"] * 2, 1, "none", "plain features",
        f"vb224_combo.json:{k}.mlp2", table="tab_combo_enc.tex", encoder=" + ".join(NAME[n] for n in names),
        cost_note="K x (d1 + d2) x 2 (fp16), same rule as tab_main")
# one-shot, PCA-k x b bits, validation-selected (E7_pareto_val, tab_pareto)
par = Eg.get("E7_pareto_val") if isinstance(Eg.get("E7_pareto_val"), list) else []
sel = set()
for B in sorted({p["bytes_per_sample"] for p in par}):  # every budget of the grid: best validation accuracy within the budget
    best = max((p for p in par if p["bytes_per_sample"] <= B), key=lambda p: p["val_acc"])  # same rule as the budgets of fig_pareto
    sel.add((best["d"], best["bits"]))
for p in sorted(par, key=lambda p: (-p["d"], -p["bits"])):
    vs = (p["d"], p["bits"]) in sel
    add(VB, f"DINOv2-g, {'PCA-' + str(p['d']) if p['d'] < D else 'full'}, {p['bits']}-bit", "fm_pca", p, "acc", p["bytes_per_sample"], 1, "none",
        "compressed features", f"vb224_controls_dinov2_g.json:E7_pareto_val[d={p['d']},bits={p['bits']}]", table="tab_pareto.tex", encoder="DINOv2-g",
        cost_note="bytes_per_sample of the JSON (K x d x bits / 8), as tab_pareto", in_set=vs, val_selected=vs)
# DP upload (E4_dp, clipping norm per party; tab_privacy for the full features, macros dp* for PCA)
for r in Eg.get("E4_dp") or []:
    if not r["eps_upload"]:
        continue
    tag = {D: "Full", 64: "Sf", 16: "St"}[r["d"]] + {64: "Ssf", 8: "Eight", 1: "One"}[r["eps_upload"]]
    add(VB, f"DINOv2-g, {'PCA-' + str(r['d']) if r['d'] < D else 'full'}, DP eps={r['eps_upload']} per upload", "fm_dp", r, "acc", 16 * r["d"] * 2, 1,
        "none", "DP-noised features", f"vb224_controls_dinov2_g_dp.json:E4_dp[d={r['d']},eps={r['eps_upload']}]",
        table="tab_privacy.tex" if r["d"] == D and r["eps_upload"] in (64, 8) else None, macros=((f"dp{tag}", f"{r['acc']:.1f}"),), encoder="DINOv2-g",
        cost_note="16 x d x 2 (fp16), as tab_privacy")
hc = he.get("he_cost", {})


def he_bytes(task, enc_key, d, weighted, enc=None):
    """Encrypted upload per sample, all parties: parties x d' columns x chunks x ciphertext bytes / N (= upload_bytes_per_party x parties / N),
    doubled with class weights (columns and weighted columns, as hcMilkUp / hcMilkFullUp in make_assets2.py).  Measured entries are used as such;
    other dimensions / encoders use the ciphertext size of the measured entry of the same task (the ciphertext count depends only on d', N and
    the chunking, not on the encoder): derived, flagged in cost_note."""
    meas = hc.get(f"{task}/{enc_key}/pca{d}")
    if meas:
        c, extra = meas, meas["aggregate_bytes_round_trip"] / meas["N"]
        note = "measured (he_cost)" if enc in (None, enc_key) else f"measured on {enc_key} PCA-{d} (he_cost; same ciphertext count)"
        up = c["upload_bytes_per_party"]
    else:
        ref = sorted((k for k in hc if k.startswith(task + "/")), key=lambda k: abs(int(k.rsplit("pca", 1)[1]) - d))
        if not ref:
            return None, "", None
        c = hc[ref[0]]
        up = d * c["chunks"] * c["ciphertext_bytes"]
        note, extra = f"derived: d' x chunks x ciphertext_bytes of {ref[0]} (formula of hcMilkFullUp)", None
    return c["parties"] * (2 if weighted else 1) * up / c["N"], note + (", x2 weighted columns" if weighted else ""), extra


for ek, et in (("dinov2_g_224", "G"), ("clip_l14_224", "C")):
    r = he.get(f"vertibench/{ek}")
    if not r:
        continue
    for pk, d, tg in (("pca256", 256, "Tf"), ("pca64", 64, "Sf"), ("pca16", 16, "St")):
        s = r.get(pk, {}).get("ridge")
        nb, note, extra = he_bytes("vertibench", "dinov2_g_224", d, False, ek)
        if s and nb:
            add(VB, f"{NAME[ek]}, PCA-{d}, ridge trained under CKKS", "fm_he", s, "acc", nb, 2, "none", "ciphertexts (CKKS)",
                f"he_heads.json:vertibench/{ek}.{pk}.ridge", table="tab_he.tex", macros=((f"heVb{et}Ridge{tg}", f"{s['acc']:.1f}"),), encoder=NAME[ek],
                cost_note=note + "; accuracy emulated in plaintext", extra_bytes=extra)

# ------------------------------------------------------------------ MILK10k (clinical + dermoscopic parties; tab_freeze, tab_he)
for k, fam, lab in (("resnet18", "nf_split", "ResNet-18, scratch (split)"), ("resnet18_in", "nf_split", "ResNet-18, ImageNet (split)"),
                    ("dinov2_b_lora", "fm_split_lora", "DINOv2-B, LoRA (split)"), ("dinov2_b_ft", "fm_split_ft", "DINOv2-B, fine-tuned (split)"),
                    ("clip_b16_lora", "fm_split_lora", "CLIP-B, LoRA (split)")):
    s = msplit.get(k)
    if not s:
        continue
    E = s["epochs"]  # training passes only (activations up, gradients down), as in Table 1: 3,144 training, 1,048 val. and test lesions
    up = s["bytes_per_train_sample"] * 2 * E * 3144 / (2 * E * 3144 + E * 1048 + 1048)
    add(MK, lab, fam, s, "f1", up, E, {"nf_split": "full ResNet-18 bottom", "fm_split_lora": "LoRA adapters (r=16)", "fm_split_ft": "full ViT backbone"}[fam],
        "activations + gradients", f"milk_split.json:{k}", table="tab_freeze.tex", encoder=lab.split(",")[0],
        cost_note="bytes_per_train_sample x training share of the passes, as tab_freeze")
for k, mt in (("dinov2_b_frozen", "srFrozenB"), ("clip_b16_frozen", "srFrozenCb")):
    s = msplit2.get(k)
    if not s:
        continue
    E = s["epochs"]
    up = s["bytes_per_train_sample"] * 2 * E * 3144 / (2 * E * 3144 + E * 1048 + 1048) / 2  # frozen bottoms need no gradients (rule of tab_main)
    add(MK, f"{NAME[k.rsplit('_', 1)[0]]}, frozen bottoms (split)", "fm_split_frozen", s, "f1", up, E, "none", "activations (augmented views)",
        f"milk_split_v2.json:{k}", macros=((mt, f"{s['f1']:.1f}"),), encoder=NAME[k.rsplit("_", 1)[0]],
        cost_note="derived: formula of tab_freeze, halved for frozen bottoms as tab_main (no table shows this upload)")
MD = {"dinov2_b": 768, "clip_b16": 768, "dinov2_g": 1536, "clip_l14": 1024}
for k in ("dinov2_b", "clip_b16", "dinov2_g", "clip_l14"):
    s = milk.get(k, {}).get("clin+derm|mlp2|bal")
    if s:
        add(MK, f"{NAME[k]}, frozen, one upload", "fm_oneshot", s, "f1", 2 * MD[k] * 2, 1, "none", "plain features", f"milk_heads.json:{k}.clin+derm|mlp2|bal",
            table="tab_freeze.tex", encoder=NAME[k], cost_note="2 x d x 2 (fp16), as tab_freeze")
for k in ("dinov2_b", "clip_b16", "dinov2_g"):
    s = mreg.get(k, {}).get("late fusion (fixed 0.5)|views mean")
    if s:
        add(MK, f"{NAME[k]}, frozen, 8 views averaged, late fusion", "fm_oneshot", s, "f1", 2 * MD[k] * 2, 1, "none", "plain features",
            f"milk_reg.json:{k}.late fusion (fixed 0.5)|views mean", table="tab_freeze.tex", encoder=NAME[k],
            cost_note="2 x d x 2 (fp16; mean of 8 views, upload x1), as tab_freeze")
for ek, et in (("dinov2_g", "G"), ("clip_l14", "C")):
    r = he.get(f"milk10k/{ek}")
    if not r:
        continue
    for pk, d, tg in (("pca256", 256, "Tf"), ("pca64", 64, "Sf"), ("pca16", 16, "St")):  # plain MLP head on PCA-k features (16-bit)
        s = r.get(pk, {}).get("mlp2")
        if s:
            add(MK, f"{NAME[ek]}, PCA-{d}, 16-bit", "fm_pca", s, "f1", 2 * d * 2, 1, "none", "compressed features", f"he_heads_milk.json:milk10k/{ek}.{pk}.mlp2",
                macros=((f"heMilk{et}Mlp{tg}", f"{s['f1']:.1f}"),) if tg != "St" else (), encoder=NAME[ek], cost_note="2 x d x 2 (fp16)")
    for pk, d in (("full", MD[ek]), ("pca256", 256), ("pca64", 64), ("pca16", 16)):
        s = r.get(pk, {}).get("ridge")
        nb, note, extra = he_bytes("milk10k", "dinov2_g", d, True, ek)  # class-weighted ridge
        if s and nb:
            add(MK, f"{NAME[ek]}, {'full' if pk == 'full' else 'PCA-' + str(d)}, class-weighted ridge under CKKS", "fm_he", s, "f1", nb, 2, "none",
                "ciphertexts (CKKS)", f"he_heads_milk.json:milk10k/{ek}.{pk}.ridge", table="tab_he.tex", encoder=NAME[ek],
                cost_note=note + "; F1 emulated in plaintext", extra_bytes=extra)


# ------------------------------------------------------------------ Pareto sets
def pareto(pts):
    """Upper-left envelope: fewer bytes and higher metric are better; ties in bytes keep the best metric."""
    out, best = [], -np.inf
    for p in sorted(pts, key=lambda p: (p["bytes_per_sample"], -p["metric_mean"])):
        if p["metric_mean"] > best:
            out.append(p)
            best = p["metric_mean"]
    return out


for bench in (VB, MK):
    pts = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    front = {id(p) for p in pareto(pts)}
    secure = {id(p) for p in pareto([p for p in pts if p["server_sees"] in ("ciphertexts (CKKS)", "DP-noised features")])}
    for p in P:
        if p["benchmark"] != bench:
            continue
        p["pareto"] = id(p) in front
        p["pareto_no_plaintext"] = id(p) in secure
        dom = [q for q in pts if q["bytes_per_sample"] <= p["bytes_per_sample"] and q is not p]
        q = max(dom, key=lambda q: (q["metric_mean"], -q["bytes_per_sample"]), default=None)
        p["best_point_at_most_as_costly"] = f"{q['name']} ({q['metric_mean']:.1f}, {q['bytes_human']})" if q and q["metric_mean"] >= p["metric_mean"] else ""


# ------------------------------------------------------------------ check every number against the generated tables and macros.tex
def check_tables():
    bad, n = [], 0
    mac = open(os.path.join(OUT, "macros.tex")).read() if os.path.exists(os.path.join(OUT, "macros.tex")) else ""
    for p in P:
        table, met, sz = p["_check"]
        if table and os.path.exists(os.path.join(OUT, table)):
            t = open(os.path.join(OUT, table)).read()
            n += 1
            if met not in t:
                bad.append(f"{p['name']}: {met} not in {table}")
            if table in ("tab_main.tex", "tab_freeze.tex", "tab_pareto.tex", "tab_privacy.tex") and sz not in t:
                bad.append(f"{p['name']}: {sz} not in {table}")
        for mname, val in p["_macros"]:
            n += 1
            if f"\\newcommand{{\\{mname}}}{{{val}}}" not in mac:
                bad.append(f"{p['name']}: \\{mname}={val} not in macros.tex")
    print(f"checked {n} table cells / macros:", "all match" if not bad else f"{len(bad)} MISMATCHES")
    for b in bad:
        print("  ", b)


check_tables()

# ------------------------------------------------------------------ CSV
YL = {VB: (73.3, 87.6), MK: (38.0, 65.5)}  # y range of the panels
for p in P:
    p["plotted"] = p["in_pareto_set"] and p["metric_mean"] >= YL[p["benchmark"]][0]
COLS = ["benchmark", "name", "family", "uses_foundation", "encoder", "metric", "metric_mean", "metric_std", "n_runs", "bytes_per_sample", "bytes_human",
        "rounds", "party_backprop", "party_backprop_bool", "server_sees", "val_selected", "in_pareto_set", "plotted", "pareto", "pareto_no_plaintext",
        "best_point_at_most_as_costly", "cost_note", "aggregate_bytes_per_sample_not_counted", "table", "source"]
with open(os.path.join(OUT, "frontier_points.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    for p in P:
        row = []
        for c in COLS:
            v = p[c]
            row.append(f"{v:.4f}" if isinstance(v, float) and c in ("metric_mean", "metric_std") else (f"{v:.1f}" if isinstance(v, float) else v))
        w.writerow(row)
print("wrote frontier_points.csv,", len(P), "points")

# ------------------------------------------------------------------ figure
XL = (20, 2.6 * MiB)
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.95), sharex=True)
for ax, bench in zip(axs, (VB, MK), strict=True):
    pts = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    lo, hi = YL[bench]
    # validation-selected PCA / quantisation curve
    pc = sorted([p for p in pts if p["fam"] == "fm_pca"], key=lambda p: p["bytes_per_sample"])
    if bench == VB and pc:
        ax.plot([p["bytes_per_sample"] for p in pc], [p["metric_mean"] for p in pc], color=C["green"], lw=0.8, zorder=2)
    # Pareto envelope (staircase), drawn under the markers
    fr = pareto(pts)
    xs, ys = [p["bytes_per_sample"] for p in fr], [p["metric_mean"] for p in fr]
    ax.step(xs + [XL[1]], ys + [ys[-1]], where="post", color="k", lw=0.7, ls=(0, (3, 2)), zorder=1)
    below = []
    for fam, (_lab, mk, col, fm) in FAM.items():
        q = [p for p in pts if p["fam"] == fam]
        vis = [p for p in q if p["metric_mean"] >= lo]
        below += [p for p in q if p["metric_mean"] < lo]
        if not vis:
            continue
        x, y, e = (np.array([p[c] for p in vis]) for c in ("bytes_per_sample", "metric_mean", "metric_std"))
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col if fm else C["grey"], elinewidth=0.5, alpha=0.45, capsize=0, zorder=2)
        ms = 12 if fam == "fm_pca" else (26 if mk in "PX" else 20)
        if fm:
            ax.scatter(x, y, s=ms, marker=mk, c=col, edgecolors="white", linewidths=0.4, zorder=3)
        else:
            ax.scatter(x, y, s=ms, marker=mk, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
    ax.scatter(xs, ys, s=58, facecolors="none", edgecolors="k", linewidths=0.6, zorder=4)
    if below:
        ax.text(0.98, 0.015, f"below the axis: {len(below)} points ({', '.join(sorted({FAM[p['fam']][0] for p in below}))}, "
                f"{min(p['metric_mean'] for p in below):.0f}–{max(p['metric_mean'] for p in below):.0f} %)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color="#555555")
    ax.set_xscale("log", base=2)
    ax.set_xlim(*XL)
    ax.set_ylim(lo, hi)
    ax.set_xticks([32, 1 * KiB, 32 * KiB, 1 * MiB])
    ax.set_xticklabels(["32 B", "1 KiB", "32 KiB", "1 MiB"])
    ax.set_xticks([2.0 ** i for i in range(5, 22)], minor=True)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, which="major", alpha=0.25)
    ax.set_xlabel("bytes exchanged per sample (all parties)", fontsize=7)
    ax.set_title(bench + (" (16 parties)" if bench == VB else " (2 parties)"), fontsize=7)
axs[0].set_ylabel("test accuracy (%)", fontsize=7)
axs[1].set_ylabel("macro F1 (%)", fontsize=7)

# regimes, sparingly: each label points at the point that defines it; label positions (data coordinates) sit in empty areas
RG = dict(fontsize=5.8, style="italic", color="#333333", va="center")


def regime(ax, txt, p, xytext, ha="left"):
    ax.annotate(txt, (p["bytes_per_sample"], p["metric_mean"]), xytext=xytext, textcoords="data", ha=ha,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666", shrinkA=1, shrinkB=3), **RG)


for ax, bench, pos in ((axs[0], VB, {"communication": (40, 75.9), "accuracy": (300 * KiB, 87.25), "mix": (1.3 * KiB, 82.6), "security": (2.4 * MiB, 79.4)}),
                       (axs[1], MK, {"communication": (110, 45.0), "accuracy": (12 * KiB, 64.6), "mix": (90, 60.5), "security": (2.4 * MiB, 51.0)})):
    fr = pareto([p for p in P if p["benchmark"] == bench and p["in_pareto_set"]])
    he_ = [p for p in P if p["benchmark"] == bench and p["fam"] == "fm_he"]
    mix = [p for p in P if p["benchmark"] == bench and p["fam"] == "fm_logits"] or [p for p in fr if p["fam"] == "fm_pca"][-1:]
    regime(ax, "communication", fr[0], pos["communication"])
    regime(ax, "accuracy", max(fr, key=lambda p: p["metric_mean"]), pos["accuracy"], ha="right" if bench == MK else "center")
    if mix:
        regime(ax, "mix", max(mix, key=lambda p: p["metric_mean"]), pos["mix"], ha="center" if bench == VB else "left")
    if he_:
        regime(ax, "security", max(he_, key=lambda p: p["metric_mean"]), pos["security"], ha="right")

# legend: foundation (filled) / no foundation (hollow) / Pareto
H1 = [Line2D([], [], ls="", marker=mk, ms=4.2 if mk not in "PX" else 4.8, mfc=col, mec="white", mew=0.4, label=lab) for fam, (lab, mk, col, fm) in FAM.items() if fm]
H1[1] = Line2D([], [], ls="-", lw=0.8, marker="o", ms=3.4, color=C["green"], mfc=C["green"], mec="white", mew=0.4, label=FAM["fm_pca"][0])
H2 = [Line2D([], [], ls="", marker=mk, ms=4.2, mfc="white", mec=col, mew=0.9, label=lab) for fam, (lab, mk, col, fm) in FAM.items() if not fm]
H2.append(Line2D([], [], ls=(0, (3, 2)), lw=0.7, color="k", marker="o", ms=6, mfc="none", mec="k", mew=0.6, label="Pareto-optimal (front)"))
kw = dict(frameon=False, fontsize=5.6, handletextpad=0.25, handlelength=1.3, columnspacing=0.8, borderaxespad=0.1, borderpad=0.1,
          title_fontsize=5.8, labelspacing=0.3, alignment="left")
fig.legend(handles=H1, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 0.088), title="foundation model (filled markers)", **kw)
fig.legend(handles=H2, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 0.0), title="no foundation model (hollow markers)", **kw)
fig.tight_layout(rect=(0, 0.215, 1, 1), w_pad=1.0)
fig.savefig(os.path.join(OUT, "fig_frontier.pdf"))
fig.savefig(os.path.join(OUT, "fig_frontier.png"), dpi=220)
plt.close(fig)
print("wrote fig_frontier.pdf, fig_frontier.png")

# ------------------------------------------------------------------ macros
MAC = {}


def macro(name, value):
    MAC["fr" + "".join(DIG.get(c, c) for c in name if c.isalnum())] = value


def tex(s):
    return s.replace("eps=", "$\\varepsilon{=}$").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


ONESHOT = ("fm_oneshot", "fm_pca", "nf_ssl", "nf_frozen")  # one upload, plain or compressed features
for bench, t in ((VB, "Vb"), (MK, "Milk")):
    allp = [p for p in P if p["benchmark"] == bench]
    pts = [p for p in allp if p["in_pareto_set"]]
    fr = [p for p in pts if p["pareto"]]
    macro(f"{t}NPoints", str(len(pts)))  # methods compared (the PCA grid counts with its validation-selected points only)
    macro(f"{t}NPointsAll", str(len(allp)))
    macro(f"{t}NPareto", str(len(fr)))
    macro(f"{t}NParetoFound", str(sum(p["uses_foundation"] for p in fr)))
    macro(f"{t}NParetoNonFound", str(sum(not p["uses_foundation"] for p in fr)))
    b = max(pts, key=lambda p: p["metric_mean"])
    macro(f"{t}BestName", tex(b["name"]))
    macro(f"{t}Best", f"{b['metric_mean']:.1f}")
    macro(f"{t}BestBytes", size(b["bytes_per_sample"]))
    nf = max((p for p in pts if not p["uses_foundation"]), key=lambda p: p["metric_mean"])
    macro(f"{t}BestNonFoundName", tex(nf["name"]))
    macro(f"{t}BestNonFound", f"{nf['metric_mean']:.1f}")
    macro(f"{t}BestNonFoundBytes", size(nf["bytes_per_sample"]))
    os_ = [p for p in pts if p["fam"] in ONESHOT]
    bo = max(os_, key=lambda p: p["metric_mean"])
    macro(f"{t}BestOneShotName", tex(bo["name"]))
    macro(f"{t}BestOneShot", f"{bo['metric_mean']:.1f}")
    macro(f"{t}BestOneShotBytes", size(bo["bytes_per_sample"]))
    ch = min((p for p in pts if p["metric_mean"] >= bo["metric_mean"] - 1.0), key=lambda p: (p["bytes_per_sample"], -p["metric_mean"]))
    macro(f"{t}CheapName", tex(ch["name"]))
    macro(f"{t}Cheap", f"{ch['metric_mean']:.1f}")
    macro(f"{t}CheapBytes", size(ch["bytes_per_sample"]))
    macro(f"{t}CheapRatio", f"{bo['bytes_per_sample'] / ch['bytes_per_sample']:.0f}")
    # cheapest foundation point that beats the best non-foundation point, and how much less it exchanges
    cb = min((p for p in pts if p["uses_foundation"] and p["metric_mean"] > nf["metric_mean"]), key=lambda p: p["bytes_per_sample"])
    macro(f"{t}BeatNonFoundName", tex(cb["name"]))
    macro(f"{t}BeatNonFound", f"{cb['metric_mean']:.1f}")
    macro(f"{t}BeatNonFoundBytes", size(cb["bytes_per_sample"]))
    macro(f"{t}BeatNonFoundRatio", f"{nf['bytes_per_sample'] / cb['bytes_per_sample']:,.0f}".replace(",", "{,}"))
with open(os.path.join(OUT, "macros_frontier.tex"), "w") as f:
    for k, v in sorted(MAC.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote macros_frontier.tex,", len(MAC), "macros")
