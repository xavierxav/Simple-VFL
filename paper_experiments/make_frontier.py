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
import re
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
FL = load("frontier_flops.json") or {}  # s25_frontier_flops.py: FLOPs of the ResNet-18 bottoms that vb224_encoders.json does not cover
GF = {k: v["flops_per_image"] / 1e9 for k, v in cost.items()}  # forward GFLOPs per image (vb224_encoders.json, 224 px; VertiBench bottom 158 px)
GF.update({k: v["flops_per_image"] / 1e9 for k, v in FL.items() if isinstance(v, dict) and "flops_per_image" in v})
HERE = os.path.dirname(os.path.abspath(__file__))


def code_int(fname, pattern):
    """An integer constant of the training code (batch sizes), read from its source so that it cannot drift from it."""
    m = re.search(pattern, open(os.path.join(HERE, fname)).read(), re.S)
    if not m:
        raise SystemExit(f"{fname}: pattern {pattern!r} not found")
    return int(m.group(1))


BS_REPRO = code_int("s8_vertibench_repro.py", r"for i in range\(0, len\(perm\), (\d+)\)")  # SplitNN, VertiBench recipe: batch 32
BS_CLIPB = code_int("s22_vb_audit.py", r'PE_SPLIT_BS", "(\d+)"')  # CLIP-B split learning on VertiBench (default, no chain overrides it)
BS_MILK = code_int("s11_milk10k.py", r"epochs=epochs_of\(kind\), bs=(\d+)")  # MILK10k split learning
BS_HEAD = code_int("pe_common.py", r"def train_head\(.*?\bbs=(\d+)")  # train_head (frozen-feature heads, four-logit head)
if any("PE_SPLIT_BS" in open(os.path.join(HERE, f)).read() for f in os.listdir(HERE) if f.endswith(".sh")):
    raise SystemExit("a chain script overrides PE_SPLIT_BS: read the batch size of vb_split_clipb.json from it")
# MILK10k folds (s11.folds: test = fold f, validation = fold f+1, training = the other three)
_fc = [sum(c) for c in (milk.get("fold_test_counts") or [])]
N_TR_MILK = [milk["n_lesions"] - _fc[f] - _fc[(f + 1) % len(_fc)] for f in range(len(_fc))] if _fc else []


def n_batches(n, bs):
    """Mini-batches per epoch of the training loops (range(0, n, bs), a last batch of one sample is skipped)."""
    return n // bs + (1 if n % bs >= 2 else 0)


# Sequential messages until the model is trained (one-way, party <-> server; synchronous steps, any number of parties in parallel):
#   one upload = 1; split learning / logit heads = epochs x (2 per training mini-batch: activations or logits up, gradients down,
#   + 1 per epoch for the validation pass when the recipe has one); frozen bottoms = 1 per mini-batch (no gradients down);
#   HE ridge = 1 upload + 2 per decryption round trip (aggregates; validation scores), + 1 for the encrypted class weights on MILK10k.
#   Interactive key generation (threshold CKKS) and inference are not counted.
def msgs_split(E, nb, val=True, grads=True):
    return E * (nb * (2 if grads else 1) + (1 if val else 0))

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


def add(bench, name, fam, s, key, nbytes, epochs, backprop, sees, source, *, table=None, macros=(), cost_note="", encoder="", in_set=True,
        val_selected="", mean=None, std=None, n=None, extra_bytes=None, label=None, msgs=1, msgs_rule="one upload", gflops=None, gflops_rule="",
        cacheable=False):
    """One method = one point.  s: summary dict of the results (mean, per-seed values or std), key: 'acc' or 'f1'.
    epochs: passes over the training data with interaction (1 for one upload); msgs: sequential messages (rule above msgs_split);
    gflops: party-side GFLOPs per training image of one party, forward and backward (3 x forward for a training step, as tab_main);
    cacheable: the party-side compute is one forward pass of a frozen encoder, reusable for any head or task."""
    m = s[key] if s is not None else mean
    e = sd(s, key) if s is not None else std
    nn = (s.get("n") or len(s.get(key + "s", [])) or None) if s is not None else n
    P.append(dict(benchmark=bench, name=name, family=FAM_LONG[fam], fam=fam, uses_foundation=FAM[fam][3], encoder=encoder,
                  metric="test accuracy (%)" if key == "acc" else "macro F1 (%)", metric_mean=float(m), metric_std=float(e), n_runs=nn,
                  bytes_per_sample=float(nbytes), bytes_human=size(nbytes).replace("\\,", " "), epochs=epochs, rounds=msgs, rounds_rule=msgs_rule,
                  party_gflops=None if gflops is None else float(gflops), gflops_rule=gflops_rule, party_compute_cacheable=cacheable,
                  party_backprop=backprop, party_backprop_bool=backprop != "none", server_sees=sees, val_selected=val_selected, in_pareto_set=in_set,
                  cost_note=cost_note, aggregate_bytes_per_sample_not_counted=extra_bytes, source=source, table=table or "", label=label or name,
                  _check=(table, pm(s, key) if s is not None else f"{m:.1f}\\,{{\\scriptsize$\\pm${e:.1f}}}", size(nbytes)), _macros=macros))


VB, MK = "VertiBench Satellite", "MILK10k"
SN_TEXT = "4 x K x 512 x epochs (512-d fp16 activations up, gradients down, every epoch), as tab_main"

# ------------------------------------------------------------------ VertiBench: split learning (tab_main)
add(VB, "SplitNN, published (ResNet-18, 13 bands)", "nf_split", None, "acc", 4 * K * 512 * 50, 50, "full ResNet-18 bottom", "activations + gradients",
    "make_assets2.py (hard-coded, wu2024vertibench)", table="tab_main.tex", mean=81.2, std=0.4, label="SplitNN (published)",
    cost_note=SN_TEXT + "; literature number, left out of the Pareto set and the figure (our reproduction below is plotted)", in_set=False,
    msgs=msgs_split(50, n_batches(N_TR, BS_REPRO), val=False), msgs_rule=f"50 epochs x {n_batches(N_TR, BS_REPRO)} batches of {BS_REPRO} x 2 (s8 recipe)",
    gflops=3 * GF["vertibench_resnet18_bottom_158px"] * 50, gflops_rule="3 x forward x 50 epochs (tab_main)")
if rep:
    nb = n_batches(N_TR, BS_REPRO)  # s8: the whole official training set, batch 32, no validation pass (last epoch is tested)
    add(VB, "SplitNN, VertiBench recipe (ResNet-18, 13 bands)", "nf_split", rep["last_epoch"], "acc", 4 * K * 512 * 50, 50, "full ResNet-18 bottom",
        "activations + gradients", "vb_repro_splitresnet.json:last_epoch", table="tab_main.tex", cost_note=SN_TEXT, label="SplitNN",
        msgs=msgs_split(50, nb, val=False), msgs_rule=f"50 epochs x {nb} batches of {BS_REPRO} x 2 (s8, no validation pass)",
        gflops=3 * GF["vertibench_resnet18_bottom_158px"] * 50, gflops_rule="3 x forward x 50 epochs (tab_main)")
for k, lab in (("splitnn_all", "SplitNN, our recipe, scratch (13 bands)"), ("splitnn_rgb", "SplitNN, our recipe, scratch (RGB)"),
               ("splitnn_rgb_pt", "SplitNN, our recipe, ImageNet init. (RGB)")):
    b = base[k]
    if b:
        nb, fk = n_batches(N_TR_H, b["batch_size"]), "splitnn_bottom_13band_158px" if k == "splitnn_all" else "splitnn_bottom_rgb_158px"
        add(VB, lab, "nf_split", b["summary"], "acc", 4 * K * 512 * b["epochs"], b["epochs"], "full ResNet-18 bottom", "activations + gradients",
            f"baseline_{k}.json:summary", table="tab_main.tex", cost_note=SN_TEXT, msgs=msgs_split(b["epochs"], nb),
            msgs_rule=f"{b['epochs']} epochs x ({nb} batches of {b['batch_size']} x 2 + 1 validation pass) (s3)",
            gflops=3 * GF[fk] * b["epochs"] if fk in GF else None, gflops_rule=f"3 x forward x epochs, forward of {fk} (frontier_flops.json, s25)")
for k, fam in (("clip_b16_lora", "fm_split_lora"), ("clip_b16_frozen", "fm_split_frozen")):
    s = clipb.get(k, {})
    if not s.get("summary"):
        continue
    E = s["infos"][0]["epochs"]
    tr_bytes = np.mean([i["bytes_per_train_sample"] for i in s["infos"]]) * 2 * E * N_TR_H / (2 * E * N_TR_H + E * N_VA_H + N_TE)
    if k.endswith("frozen"):  # frozen bottoms need no gradients: one augmented upload per epoch
        tr_bytes /= 2
    nb = n_batches(N_TR_H, BS_CLIPB)
    add(VB, f"CLIP-B bottoms, {'LoRA' if 'lora' in k else 'frozen'} (split)", fam, s["summary"], "acc", tr_bytes, E,
        "LoRA adapters (r=16)" if "lora" in k else "none", "activations + gradients" if "lora" in k else "activations (augmented views)",
        f"vb_split_clipb.json:{k}.summary", table="tab_main.tex", encoder="CLIP ViT-B/16",
        cost_note="mean(bytes_per_train_sample) x training share of the passes, as tab_main" + ("; halved: no gradients" if "frozen" in k else ""),
        msgs=msgs_split(E, nb, grads="lora" in k),
        msgs_rule=f"{E} epochs x ({nb} batches of {BS_CLIPB} x {2 if 'lora' in k else 1} + 1 validation pass) (s22 / pe_split)"
        + ("" if "lora" in k else "; frozen bottoms do not depend on the server: could be one message"),
        gflops=(3 if "lora" in k else 1) * GF["clip_b16_224"] * E, gflops_rule=f"{'3 x' if 'lora' in k else '1 x'} forward x epochs (tab_main)")
# few rounds: four logits per party (tab_main)
for kk, v in aud.get("heads", {}).items():
    k, h = kk.split("|")
    if not h.startswith("logits"):
        continue
    up = np.mean(v["epochs_run"]) * 2 * K * 4 * 2  # training passes only, as for split learning: 4 logits up, 4 gradients down, fp16
    nb, Em = n_batches(N_TR_H, BS_HEAD), float(np.mean(v["epochs_run"]))
    add(VB, f"{NAME[k]}, 4-logit MLP per party", "fm_logits", v, "acc", up, Em, "party MLP head on frozen features",
        "logits + gradients", f"vb_audit.json:heads.{kk}", table="tab_main.tex", encoder=NAME[k],
        cost_note="mean(epochs_run) x 2 x K x 4 x 2 (4 fp16 logits up, 4 gradients down per epoch), as tab_main",
        msgs=msgs_split(Em, nb), msgs_rule=f"mean(epochs_run) x ({nb} batches of {BS_HEAD} x 2 + 1 validation pass) (train_head)",
        gflops=GF[k] + 6 * v["params"] / K * Em / 1e9, cacheable=True,
        gflops_rule="encoder forward (tab_main) + party MLP training, 6 x params / K x mean(epochs_run)")
# one-shot, party-side self-supervised training (tab_main)
for k, lab in (("nat_all", "FedOnce, Noise As Targets (13 bands)"), ("ssl_all", "FedOnce, SimCLR (13 bands)"), ("ssl_rgb", "FedOnce, SimCLR (RGB)")):
    b = base[k]
    if b:
        # party-side SimCLR / NAT pre-training on all its training images (s3.run_ssl): epochs x views x 3 x forward on 128 px crops, full
        # batches only, then one forward pass on the whole 158 px image for the upload
        tg = {"nat_all": "nat_13band", "ssl_all": "13band", "ssl_rgb": "rgb"}[k]
        ft, ff, views = GF.get(f"fedonce_{tg}_train_128px"), GF.get(f"fedonce_{tg}_feat_158px"), 1 if k.startswith("nat") else 2
        g = b["epochs"] * views * 3 * ft * (N_TR // b["batch_size"] * b["batch_size"]) / N_TR + ff if ft and ff else None
        add(VB, lab, "nf_ssl", b["summary"], "acc", K * b.get("feature_dim", 512) * 2, 1, "local SSL pre-training (no interaction)",
            "plain features (local SSL encoder)", f"baseline_{k}.json:summary", table="tab_main.tex", cost_note="K x feature_dim x 2 (fp16), as tab_main",
            gflops=g, gflops_rule=f"{b['epochs']} epochs x {views} view(s) x 3 x forward (128 px) x full-batch share + 1 forward (158 px); s25 FLOPs")
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
        encoder=NAME[k], cost_note="K x dim x 2 (fp16), as tab_main", gflops=GF.get(k), cacheable=True,
        gflops_rule="one forward pass (vb224_encoders.json cost)" if k in GF else "no FLOPs recorded")
for k, v in combo.items():  # two encoders per party (tab_combo_enc)
    names = k.split("+")
    if len(names) < 2:
        continue
    add(VB, " + ".join(NAME[n] for n in names) + ", frozen, one upload", "fm_oneshot", v["mlp2"], "acc", K * v["dim"] * 2, 1, "none", "plain features",
        f"vb224_combo.json:{k}.mlp2", table="tab_combo_enc.tex", encoder=" + ".join(NAME[n] for n in names),
        cost_note="K x (d1 + d2) x 2 (fp16), same rule as tab_main", gflops=sum(GF[n] for n in names), cacheable=True,
        gflops_rule="sum of the two forward passes (as tab_combo_enc)")
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
        cost_note="bytes_per_sample of the JSON (K x d x bits / 8), as tab_pareto", in_set=vs, val_selected=vs, gflops=GF["dinov2_g_224"], cacheable=True,
        gflops_rule="DINOv2-g forward (PCA projection and quantisation: < 0.01 GFLOPs, not counted)")
# DP upload (E4_dp, clipping norm per party; tab_privacy for the full features, macros dp* for PCA)
for r in Eg.get("E4_dp") or []:
    if not r["eps_upload"]:
        continue
    tag = {D: "Full", 64: "Sf", 16: "St"}[r["d"]] + {64: "Ssf", 8: "Eight", 1: "One"}[r["eps_upload"]]
    add(VB, f"DINOv2-g, {'PCA-' + str(r['d']) if r['d'] < D else 'full'}, DP eps={r['eps_upload']} per upload", "fm_dp", r, "acc", 16 * r["d"] * 2, 1,
        "none", "DP-noised features", f"vb224_controls_dinov2_g_dp.json:E4_dp[d={r['d']},eps={r['eps_upload']}]",
        table="tab_privacy.tex" if r["d"] == D and r["eps_upload"] in (64, 8) else None, macros=((f"dp{tag}", f"{r['acc']:.1f}"),), encoder="DINOv2-g",
        cost_note="16 x d x 2 (fp16), as tab_privacy", gflops=GF["dinov2_g_224"], cacheable=True, gflops_rule="DINOv2-g forward (clipping, noise: negligible)")
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
            add(VB, f"{NAME[ek]}, PCA-{d}, ridge trained under CKKS", "fm_he", s, "acc", nb, 1, "none", "ciphertexts (CKKS)",
                f"he_heads.json:vertibench/{ek}.{pk}.ridge", table="tab_he.tex", macros=((f"heVb{et}Ridge{tg}", f"{s['acc']:.1f}"),), encoder=NAME[ek],
                cost_note=note + "; accuracy emulated in plaintext", extra_bytes=extra, msgs=5,
                msgs_rule="1 encrypted upload + 2 x decryption round trip (aggregates, validation scores); key generation not counted",
                gflops=GF[ek], cacheable=True, gflops_rule="encoder forward; CKKS encryption is not counted in FLOPs (he_cost: seconds)")

# ------------------------------------------------------------------ MILK10k (clinical + dermoscopic parties; tab_freeze, tab_he)
for k, fam, lab in (("resnet18", "nf_split", "ResNet-18, scratch (split)"), ("resnet18_in", "nf_split", "ResNet-18, ImageNet (split)"),
                    ("dinov2_b_lora", "fm_split_lora", "DINOv2-B, LoRA (split)"), ("dinov2_b_ft", "fm_split_ft", "DINOv2-B, fine-tuned (split)"),
                    ("clip_b16_lora", "fm_split_lora", "CLIP-B, LoRA (split)")):
    s = msplit.get(k)
    if not s:
        continue
    E = s["epochs"]  # training passes only (activations up, gradients down), as in Table 1: 3,144 training, 1,048 val. and test lesions
    up = s["bytes_per_train_sample"] * 2 * E * 3144 / (2 * E * 3144 + E * 1048 + 1048)
    nb = float(np.mean([n_batches(n, BS_MILK) for n in N_TR_MILK]))
    nb = int(nb) if nb.is_integer() else nb
    fk = "milk_resnet18_224px" if k.startswith("resnet18") else k.rsplit("_", 1)[0] + "_224"  # MILK10k images are 224 px (s11 SIZE)
    add(MK, lab, fam, s, "f1", up, E, {"nf_split": "full ResNet-18 bottom", "fm_split_lora": "LoRA adapters (r=16)", "fm_split_ft": "full ViT backbone"}[fam],
        "activations + gradients", f"milk_split.json:{k}", table="tab_freeze.tex", encoder=lab.split(",")[0],
        cost_note="bytes_per_train_sample x training share of the passes, as tab_freeze", msgs=msgs_split(E, nb),
        msgs_rule=f"{E} epochs x ({nb:g} batches of {BS_MILK} x 2 + 1 validation pass) (s11 / pe_split)",
        gflops=3 * GF[fk] * E if fk in GF else None, gflops_rule=f"3 x forward x epochs (rule of tab_main), forward of {fk}")
for k, mt in (("dinov2_b_frozen", "srFrozenB"), ("clip_b16_frozen", "srFrozenCb")):
    s = msplit2.get(k)
    if not s:
        continue
    E = s["epochs"]
    up = s["bytes_per_train_sample"] * 2 * E * 3144 / (2 * E * 3144 + E * 1048 + 1048) / 2  # frozen bottoms need no gradients (rule of tab_main)
    nb = float(np.mean([n_batches(n, BS_MILK) for n in N_TR_MILK]))
    nb = int(nb) if nb.is_integer() else nb
    add(MK, f"{NAME[k.rsplit('_', 1)[0]]}, frozen bottoms (split)", "fm_split_frozen", s, "f1", up, E, "none", "activations (augmented views)",
        f"milk_split_v2.json:{k}", macros=((mt, f"{s['f1']:.1f}"),), encoder=NAME[k.rsplit("_", 1)[0]],
        cost_note="derived: formula of tab_freeze, halved for frozen bottoms as tab_main (no table shows this upload)", msgs=msgs_split(E, nb, grads=False),
        msgs_rule=f"{E} epochs x ({nb:g} batches of {BS_MILK} + 1 validation pass); frozen bottoms do not depend on the server: could be one message",
        gflops=GF[k.rsplit("_", 1)[0] + "_224"] * E, gflops_rule="1 x forward x epochs (augmented views; rule of tab_main)")
MD = {"dinov2_b": 768, "clip_b16": 768, "dinov2_g": 1536, "clip_l14": 1024}
for k in ("dinov2_b", "clip_b16", "dinov2_g", "clip_l14"):
    s = milk.get(k, {}).get("clin+derm|mlp2|bal")
    if s:
        add(MK, f"{NAME[k]}, frozen, one upload", "fm_oneshot", s, "f1", 2 * MD[k] * 2, 1, "none", "plain features", f"milk_heads.json:{k}.clin+derm|mlp2|bal",
            table="tab_freeze.tex", encoder=NAME[k], cost_note="2 x d x 2 (fp16), as tab_freeze", gflops=GF[k + "_224"], cacheable=True,
            gflops_rule="one forward pass at 224 px (vb224_encoders.json cost: same encoder and input size)")
for k in ("dinov2_b", "clip_b16", "dinov2_g"):
    s = mreg.get(k, {}).get("late fusion (fixed 0.5)|views mean")
    if s:
        add(MK, f"{NAME[k]}, frozen, 8 views averaged, late fusion", "fm_oneshot", s, "f1", 2 * MD[k] * 2, 1, "none", "plain features",
            f"milk_reg.json:{k}.late fusion (fixed 0.5)|views mean", table="tab_freeze.tex", encoder=NAME[k],
            cost_note="2 x d x 2 (fp16; mean of 8 views, upload x1), as tab_freeze", gflops=8 * GF[k + "_224"], cacheable=True,
            gflops_rule="8 forward passes (8 views; x8 as the time of tab_freeze)")
for ek, et in (("dinov2_g", "G"), ("clip_l14", "C")):
    r = he.get(f"milk10k/{ek}")
    if not r:
        continue
    for pk, d, tg in (("pca256", 256, "Tf"), ("pca64", 64, "Sf"), ("pca16", 16, "St")):  # plain MLP head on PCA-k features (16-bit)
        s = r.get(pk, {}).get("mlp2")
        if s:
            add(MK, f"{NAME[ek]}, PCA-{d}, 16-bit", "fm_pca", s, "f1", 2 * d * 2, 1, "none", "compressed features", f"he_heads_milk.json:milk10k/{ek}.{pk}.mlp2",
                macros=((f"heMilk{et}Mlp{tg}", f"{s['f1']:.1f}"),) if tg != "St" else (), encoder=NAME[ek], cost_note="2 x d x 2 (fp16)",
                gflops=GF[ek + "_224"], cacheable=True, gflops_rule="encoder forward at 224 px (PCA projection not counted)")
    for pk, d in (("full", MD[ek]), ("pca256", 256), ("pca64", 64), ("pca16", 16)):
        s = r.get(pk, {}).get("ridge")
        nb, note, extra = he_bytes("milk10k", "dinov2_g", d, True, ek)  # class-weighted ridge
        if s and nb:
            add(MK, f"{NAME[ek]}, {'full' if pk == 'full' else 'PCA-' + str(d)}, class-weighted ridge under CKKS", "fm_he", s, "f1", nb, 1, "none",
                "ciphertexts (CKKS)", f"he_heads_milk.json:milk10k/{ek}.{pk}.ridge", table="tab_he.tex", encoder=NAME[ek],
                cost_note=note + "; F1 emulated in plaintext", extra_bytes=extra, msgs=6,
                msgs_rule="encrypted class weights down + 1 encrypted upload + 2 x decryption round trip (aggregates, validation scores); key generation not counted",
                gflops=GF[ek + "_224"], cacheable=True, gflops_rule="encoder forward at 224 px; CKKS operations are not counted in FLOPs (he_cost: seconds)")


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


# ------------------------------------------------------------------ the other axes: exposure ladder and Pareto sets per axis
# Ordinal exposure of the parties' data, from most to least exposed (what the server receives):
EXPO = ["activations + gradients, every step (split learning)", "plain features, once", "PCA-compressed / quantised features, once",
        "per-party logits + gradients, every step", "DP-noised features, eps=64 per upload", "DP-noised features, eps=8 per upload",
        "DP-noised features, eps=1 per upload", "ciphertexts only (CKKS; the label owner learns X^T X)"]
EXPO_TICK = ["act. +\ngrad.", "plain\nfeat.", "PCA /\nquant.", "logits\n+ grad.", "DP\n$\\varepsilon$=64", "DP\n$\\varepsilon$=8", "DP\n$\\varepsilon$=1", "CKKS"]


def exposure_rank(p):
    f = p["fam"]
    if f in ("nf_split", "fm_split_lora", "fm_split_ft", "fm_split_frozen"):
        return 0  # frozen bottoms: activations of augmented views every step, without gradients
    if f in ("fm_oneshot", "nf_ssl", "nf_frozen"):
        return 1
    if f == "fm_pca":
        return 2
    if f == "fm_logits":
        return 3
    if f == "fm_dp":
        return {64: 4, 8: 5, 1: 6}[int(re.search(r"eps=(\d+)", p["name"]).group(1))]
    return 7  # fm_he


def pareto_by(pts, xf, low=True):
    """Pareto set for (x, metric): x lower (low=True) or higher is better, metric higher is better; ties in x keep the best metric."""
    out, best = [], -np.inf
    for p in sorted(pts, key=lambda p: (xf(p) if low else -xf(p), -p["metric_mean"])):
        if p["metric_mean"] > best:
            out.append(p)
            best = p["metric_mean"]
    return out


XF = {"rounds": lambda p: p["rounds"], "gflops": lambda p: p["party_gflops"], "exposure": lambda p: p["exposure_rank"]}
for p in P:
    p["exposure_rank"] = exposure_rank(p)
    p["exposure_level"] = EXPO[p["exposure_rank"]]
for bench in (VB, MK):
    pts = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    for ax_, low in (("rounds", True), ("gflops", True), ("exposure", False)):
        fr = {id(q) for q in pareto_by([q for q in pts if XF[ax_](q) is not None], XF[ax_], low)}
        for q in P:
            if q["benchmark"] == bench:
                q[f"pareto_{ax_}"] = id(q) in fr


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
        "epochs", "rounds", "rounds_rule", "party_gflops", "gflops_rule", "party_compute_cacheable", "exposure_rank", "exposure_level",
        "pareto_rounds", "pareto_gflops", "pareto_exposure", "party_backprop", "party_backprop_bool", "server_sees", "val_selected", "in_pareto_set", "plotted", "pareto", "pareto_no_plaintext",
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

# ------------------------------------------------------------------ the same frontier on other axes: fig_frontier_axes.pdf (VertiBench), _milk.pdf
# (a) bytes (as fig_frontier), (b) sequential messages, (c) party-side GFLOPs per training image, (d) exposure ladder (ordinal, jittered)
AXES = {
    "bytes": dict(x=lambda p: p["bytes_per_sample"], low=True, base=2, xlim=(20, 2.6 * MiB), xlabel="bytes exchanged per sample (all parties)",
                  ticks=([32, KiB, 32 * KiB, MiB], ["32 B", "1 KiB", "32 KiB", "1 MiB"]), title="(a) communication volume"),
    "rounds": dict(x=lambda p: p["rounds"], low=True, base=10, xlim=(0.6, 3e4), xlabel="sequential messages until trained",
                   ticks=([1, 10, 100, 1000, 10000], ["1", "10", "100", "1k", "10k"]), title="(b) interaction"),
    "gflops": dict(x=lambda p: p["party_gflops"], low=True, base=10, xlim=(4, 9000), xlabel="party GFLOPs per training image (fwd + bwd)",
                   ticks=([10, 100, 1000], ["10", "100", "1000"]), title="(c) party-side compute"),
    "exposure": dict(x=lambda p: p["exposure_rank"], low=False, base=None, xlim=(-0.6, len(EXPO) - 0.4), xlabel="what the server sees (most to least exposed)",
                     ticks=(list(range(len(EXPO))), EXPO_TICK), title="(d) exposure of the parties' data"),
}


def jitter(pts):
    """Deterministic spread of the points of one exposure level, in increasing upload size (left: smallest)."""
    out = {}
    for r in {p["exposure_rank"] for p in pts}:
        q = sorted((p for p in pts if p["exposure_rank"] == r), key=lambda p: (p["bytes_per_sample"], p["metric_mean"]))
        for i, p in enumerate(q):
            out[id(p)] = r + (0.0 if len(q) == 1 else -0.32 + 0.64 * i / (len(q) - 1))
    return out


def frontier_panel(ax, bench, axis, yl):
    """One panel of fig_frontier_axes; returns the points it cannot show (below the y-range, no x value) for the figure's footnote."""
    A = AXES[axis]
    allp = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    pts = [p for p in allp if A["x"](p) is not None]
    nox = [p for p in allp if A["x"](p) is None]
    lo, hi = yl
    jx = jitter(pts) if axis == "exposure" else {}
    xp = (lambda p: jx[id(p)]) if axis == "exposure" else A["x"]
    for p in pts:
        p["_x"] = xp(p)
    if axis == "bytes" and bench == VB:  # validation-selected PCA / quantisation curve, as fig_frontier
        pc = sorted([p for p in pts if p["fam"] == "fm_pca"], key=lambda p: p["bytes_per_sample"])
        ax.plot([p["bytes_per_sample"] for p in pc], [p["metric_mean"] for p in pc], color=C["green"], lw=0.8, zorder=2)
    fr = pareto_by(pts, A["x"], A["low"])
    assert {id(p) for p in fr} == {id(p) for p in pts if p["pareto" if axis == "bytes" else "pareto_" + axis]}, (bench, axis)
    if axis == "exposure":  # best metric at this level of exposure or a lower one: a non-increasing staircase
        env = [max((p["metric_mean"] for p in pts if p["exposure_rank"] >= r), default=np.nan) for r in range(len(EXPO))]
        xs_, ys_ = [], []
        for r, v in enumerate(env):
            xs_ += [r - 0.5, r + 0.5]
            ys_ += [v, v]
        ax.plot(xs_, ys_, color="k", lw=0.7, ls=(0, (3, 2)), zorder=1)
    else:
        xs, ys = [A["x"](p) for p in fr], [p["metric_mean"] for p in fr]
        ax.step(xs + [A["xlim"][1]], ys + [ys[-1]], where="post", color="k", lw=0.7, ls=(0, (3, 2)), zorder=1)
    below = []
    for fam, (_lab, mk, col, fm) in FAM.items():
        q = [p for p in pts if p["fam"] == fam]
        vis = [p for p in q if p["metric_mean"] >= lo]
        below += [p for p in q if p["metric_mean"] < lo]
        if not vis:
            continue
        x, y, e = np.array([p["_x"] for p in vis]), np.array([p["metric_mean"] for p in vis]), np.array([p["metric_std"] for p in vis])
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col if fm else C["grey"], elinewidth=0.5, alpha=0.45, capsize=0, zorder=2)
        ms = 12 if fam == "fm_pca" else (26 if mk in "PX" else 20)
        if fm:
            ax.scatter(x, y, s=ms, marker=mk, c=col, edgecolors="white", linewidths=0.4, zorder=3)
        else:
            ax.scatter(x, y, s=ms, marker=mk, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
    ax.scatter([p["_x"] for p in fr], [p["metric_mean"] for p in fr], s=58, facecolors="none", edgecolors="k", linewidths=0.6, zorder=4)
    if A["base"]:
        ax.set_xscale("log", base=A["base"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=1.5)
        if A["base"] == 2:
            ax.set_xticks([2.0 ** i for i in range(5, 22)], minor=True)
    ax.set_xlim(*A["xlim"])
    ax.set_ylim(lo, hi)
    ax.set_xticks(A["ticks"][0])
    ax.set_xticklabels(A["ticks"][1])
    ax.tick_params(labelsize=6.5)
    if axis == "exposure":
        ax.tick_params(axis="x", labelsize=5.2, length=0)
        ax.grid(False, axis="x")
    ax.grid(True, which="major", axis="y" if axis == "exposure" else "both", alpha=0.25)
    ax.set_xlabel(A["xlabel"], fontsize=6.5)
    ax.set_title(A["title"], fontsize=7, loc="left")
    return below, nox


NOTE = dict(fontsize=4.8, color="#555555")
DIM = re.compile(r"(PCA-\d+|full)")


def axes_figure(bench, yl, yl_exp, one_y, fname):
    """2 x 2 panels; one_y: height (data units) of the "one upload" pointer in (b), in a gap of the HE column."""
    H = 5.3
    fig, axs = plt.subplots(2, 2, figsize=(4.8, H))
    foot = []
    for i, (ax, axis) in enumerate(zip(axs.flat, AXES, strict=True)):
        below, nox = frontier_panel(ax, bench, axis, yl_exp if axis == "exposure" else yl)
        if below:
            fams = ", ".join(sorted({FAM[p["fam"]][0] for p in below}))
            foot.append(f"({'abcd'[i]}) {len(below)} points below the y-range ({fams}, "
                        f"{min(p['metric_mean'] for p in below):.0f}–{max(p['metric_mean'] for p in below):.0f} %)")
        if nox:
            foot.append(f"({'abcd'[i]}) not shown, no FLOPs recorded: " + ", ".join(p["name"].split(",")[0] for p in nox))
        if axis == "exposure":
            ax.tick_params(axis="x", pad=1.5)
    # merge "(a) 7 points below ... (b) 7 points below ..." into "(a)-(c) 7 points below ..."
    merged = {}
    for f in foot:
        merged.setdefault(f[4:], []).append(f[1])
    foot = [(f"({k[0]})" if len(k) == 1 else f"({k[0]})–({k[-1]})") + " " + t for t, k in merged.items()]
    pts = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    have = {p["fam"] for p in pts}
    miss = [n for f, n in (("fm_dp", "DP uploads"), ("fm_logits", "4-logit heads")) if f not in have and any(q["fam"] == f for q in P)]
    if miss:
        foot.append("not run on this benchmark yet: " + " and ".join(miss))
    ylab = "test accuracy (%)" if bench == VB else "macro F1 (%)"
    for ax in axs[:, 0]:
        ax.set_ylabel(ylab, fontsize=7)
    # (b): the one-upload column
    axs[0, 1].annotate("one upload", (1, one_y), xytext=(12, one_y), textcoords="data", va="center", fontsize=5.8, style="italic", color="#333333",
                       arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#444444", shrinkA=1, shrinkB=4, mutation_scale=5))
    # (c): what is cacheable
    axs[1, 0].text(0.02, 0.45, "frozen encoders:\none forward pass\nper image, cacheable", transform=axs[1, 0].transAxes, ha="left", va="top", **NOTE)
    # (d): best HE point, order inside the DP columns, and what PCA buys under DP
    ax = axs[1, 1]
    he = max((p for p in pts if p["fam"] == "fm_he"), key=lambda p: p["metric_mean"], default=None)
    if he:
        ax.annotate(", ".join(he["name"].split(", ")[:2]), (he["_x"], he["metric_mean"]), xytext=(-5, 3), textcoords="offset points", ha="right",
                    va="bottom", **NOTE)
    dp = {(p["exposure_rank"], DIM.search(p["name"]).group(1)): p for p in pts if p["fam"] == "fm_dp"}
    for r in {r for r, _ in dp}:
        order = [d for (_, d) in sorted(((dp[k]["_x"], k[1]) for k in dp if k[0] == r))]
        assert order == ["PCA-16", "PCA-64", "full"], order  # the note below states this order
    if dp:
        ax.text(0.02, 0.03, "DP columns, left to right:\nPCA-16, PCA-64, full dim.", transform=ax.transAxes, ha="left", va="bottom", **NOTE)
    if (4, "full") in dp and (4, "PCA-16") in dp:
        a, b = dp[(4, "full")], dp[(4, "PCA-16")]
        ax.annotate("", (b["_x"], b["metric_mean"]), xytext=(a["_x"], a["metric_mean"]),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="#444444", shrinkA=3, shrinkB=3, mutation_scale=5))
        ax.text(4.15, (a["metric_mean"] + b["metric_mean"]) / 2, f"PCA-16 vs full:\n+{b['metric_mean'] - a['metric_mean']:.0f} pts", ha="left",
                va="center", style="italic", **NOTE)
    for r in range(len(EXPO)):
        if not any(p["exposure_rank"] == r for p in pts):
            ax.text(r, yl_exp[0] + 0.04 * (yl_exp[1] - yl_exp[0]), "not\nrun", ha="center", va="bottom", fontsize=4.6, color="#888888", style="italic")
    fig.suptitle(bench + (" (16 parties)" if bench == VB else " (2 parties)"), fontsize=7.5, y=0.995)
    fig.text(0.01, 0.66 / H, "\n".join(foot), ha="left", va="bottom", **NOTE)
    fig.legend(handles=H1, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 0.26 / H), title="foundation model (filled markers)", **kw)
    fig.legend(handles=H2, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 0.0), title="no foundation model (hollow markers)", **kw)
    fig.tight_layout(rect=(0, 0.8 / H, 1, 1), w_pad=1.0, h_pad=0.6)
    fig.savefig(os.path.join(OUT, fname + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=220)
    plt.close(fig)
    for p in P:
        p.pop("_x", None)
    print(f"wrote {fname}.pdf, {fname}.png")


axes_figure(VB, YL[VB], (50.0, 88.5), 80.75, "fig_frontier_axes")
axes_figure(MK, YL[MK], (38.0, 65.5), 59.0, "fig_frontier_axes_milk")

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

# other axes (fig_frontier_axes): new names only, the macros above keep their names and values
SPLIT = ("nf_split", "fm_split_frozen", "fm_split_lora", "fm_split_ft")


def num(v, nd=None):
    nd = (1 if v < 100 else 0) if nd is None else nd
    return f"{v:,.{nd}f}".replace(",", "{,}")


for bench, t in ((VB, "Vb"), (MK, "Milk")):
    pts = [p for p in P if p["benchmark"] == bench and p["in_pareto_set"]]
    for xk, k in (("rounds", "Rounds"), ("gflops", "Gflops"), ("exposure", "Exposure")):
        fr = [p for p in pts if p["pareto_" + xk]]
        macro(f"{t}NPareto{k}", str(len(fr)))
        macro(f"{t}NPareto{k}NonFound", str(sum(not p["uses_foundation"] for p in fr)))
        macro(f"{t}Pareto{k}NonFoundName", tex("; ".join(p["name"] for p in fr if not p["uses_foundation"])) or "none")
    b = max(pts, key=lambda p: p["metric_mean"])  # = \fr<t>BestName
    macro(f"{t}BestRounds", num(b["rounds"], 0))
    macro(f"{t}BestGflops", num(b["party_gflops"]))
    # SplitNN (best non-foundation split) against the cheapest one-shot foundation point at least as good, on each axis
    sn = max((p for p in pts if p["fam"] == "nf_split"), key=lambda p: p["metric_mean"])
    macro(f"{t}SplitNNName", tex(sn["name"]))
    macro(f"{t}SplitNN", f"{sn['metric_mean']:.1f}")
    for xk, k in (("rounds", "Rounds"), ("party_gflops", "Gflops")):
        m = min((p for p in pts if p["fam"] in ("fm_oneshot", "fm_pca") and p[xk] is not None and p["metric_mean"] >= sn["metric_mean"]), key=lambda p: (p[xk], -p["metric_mean"]))
        macro(f"{t}{k}SplitNN", num(sn[xk], 0 if xk == "rounds" else None))
        macro(f"{t}{k}MatchName", tex(m["name"]))
        macro(f"{t}{k}Match", num(m[xk], 0 if xk == "rounds" else None))
        macro(f"{t}{k}MatchAcc", f"{m['metric_mean']:.1f}")
        macro(f"{t}{k}Ratio", num(sn[xk] / m[xk], 0 if sn[xk] / m[xk] >= 10 else 1))
    sp = [p["rounds"] for p in pts if p["fam"] in SPLIT]
    macro(f"{t}SplitRoundsMin", num(min(sp), 0))
    macro(f"{t}SplitRoundsMax", num(max(sp), 0))
    lg = [p["rounds"] for p in pts if p["fam"] == "fm_logits"]
    if lg:
        macro(f"{t}LogitRoundsMin", num(min(lg), 0))
        macro(f"{t}LogitRoundsMax", num(max(lg), 0))
    he = [p for p in pts if p["fam"] == "fm_he"]
    assert len({p["rounds"] for p in he}) == 1
    macro(f"{t}HeRounds", num(he[0]["rounds"], 0))
    hb = max(he, key=lambda p: p["metric_mean"])
    macro(f"{t}HeBestName", tex(hb["name"]))
    macro(f"{t}HeBest", f"{hb['metric_mean']:.1f}")
    # DP: what PCA-16 buys over the full dimension at the same epsilon (VertiBench only so far)
    dp = {(int(re.search(r"eps=(\d+)", p["name"]).group(1)), DIM.search(p["name"]).group(1)): p for p in pts if p["fam"] == "fm_dp"}
    for eps in sorted({e for e, _ in dp}):
        f_, c_ = dp[(eps, "full")], dp[(eps, "PCA-16")]
        macro(f"{t}DpFullEps{eps}", f"{f_['metric_mean']:.1f}")
        macro(f"{t}DpPcaSixteenEps{eps}", f"{c_['metric_mean']:.1f}")
        macro(f"{t}DpPcaGainEps{eps}", f"{c_['metric_mean'] - f_['metric_mean']:.1f}")
with open(os.path.join(OUT, "macros_frontier.tex"), "w") as f:
    for k, v in sorted(MAC.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("wrote macros_frontier.tex,", len(MAC), "macros")
