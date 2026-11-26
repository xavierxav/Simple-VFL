"""Stage 27 - MILK10k: how many rendezvous does adapting the encoder need?  One upload vs few rounds vs split learning.

Same task, folds, backbone and optimiser as the LoRA split-learning point of stage 11 (DINOv2-B with LoRA r=16 on the
attention query/value projections, one per party, top Linear(1536,1000)-BN-ReLU-Linear(1000,7), AdamW wd 1e-4, one-
cycle schedule, peak lr 5e-4 for the LoRA weights and 1e-3 for the top, batch 64, class-weighted cross-entropy, GPU flip
+ random-resized-crop augmentation, 10 epochs), with complete logs.  Two protocols, per fold (seed = fold):

  split    split learning (pe_split.train_split, re-implemented to log): after 5, 10 and 25 steps and after every epoch,
           the training loss (running mean since the last log), the validation loss and metrics, the test metrics and
           test probabilities, cumulative sequential messages and bytes, wall time.  Final = checkpoint with the lowest
           validation loss (as stage 11).
  rounds   few rendezvous: round 0 is the one upload (embeddings of every lesion with the frozen encoder).  Round r = 1..R:
           the label owner trains the top for E epochs on the uploaded training embeddings (fixed, warm-started across
           rounds) and sends each party one vector per training lesion (one message down); each party takes E epochs of
           LoRA steps on its own images, then uploads the new embeddings of every lesion (one message up).
             --local target (default): the vector is a target embedding t_i = h_i - eta g_i (g_i: gradient of the class-
               weighted loss w.r.t. the uploaded embedding; eta: smallest step of a log grid at which the top's weighted
               training loss falls to --shrink (0.5) times its value); the party regresses f(aug(x_i)) on t_i (MSE), a
               bounded objective, so lesions already fitted barely move (target propagation / proximal block update).
             --local linear: the vector is g_i and the party minimises mean_i <g_i, f(aug(x_i))> (stale-gradient block
               updates as in FedBCD); unbounded with E epochs of Adam steps: on fold 0 the embeddings drifted (warm-top
               validation loss 12.8 then 43.9 after rounds 1-2) and the fresh head did not improve, so it was stopped.  Messages after r rounds: 1 + 2r; bytes count the training and validation embeddings and the gradients
           (the test embeddings, inference, are counted apart).  After every upload the label owner scores (i) the warm top and (ii) a
           fresh one-upload head (pe_common.train_head, mlp2, class weights, z-scored features, as stage 11) on validation
           and test; a protocol that stops after r rounds keeps the round with the lowest validation loss among 0..r.
           Schedules are one-cycle over the R*E epochs of the run, so each prefix r is a valid protocol with fewer
           rounds (the learning rate has not decayed yet; that is the price of stopping early).
  The per-round logs therefore give the whole accuracy-versus-rendezvous curve of one run.

  python s27_milk_rounds.py split  [--folds 0,1,2,3,4]
  python s27_milk_rounds.py rounds [--folds ...] [--R 10] [--E 1] [--local target|linear] [--shrink 0.5]
Output ($PE_OUT): milk_rounds_split.json, milk_rounds_target_R{R}E{E}.json (linear: milk_rounds_R{R}E{E}.json), *_probs.npz
(test probabilities of every logged point).  JSON written after every logged point; finished folds are skipped.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe_common import DEV, OUT, metrics_from_proba, predict_proba, save_json, seed_all, train_head  # noqa: E402
from pe_split import SplitModel, _augment, _batch, _predict  # noqa: E402
from s11_milk10k import ROOT, class_weights, folds, lesions, zscore  # noqa: E402

KIND = "dinov2_b_lora"
LR_LORA, LR_TOP, WD, BS, EPOCHS = 5e-4, 1e-3, 1e-4, 64, 10
NCLS = 7
EMB_BYTES = 2  # fp16 activations / gradients on the wire


def predict(model, parties, idx):
    """Logits of the joint model (eval mode, bf16 autocast, batches of 256), as pe_split.train_split."""
    return _predict(model, parties, idx, None, 256)


def n_steps(n, bs=BS):
    return n // bs + (1 if n % bs >= 2 else 0)


@torch.no_grad()
def embed(bottom, P, idx, bs=256):
    """One party's embeddings (eval mode, no augmentation, bf16 autocast) of the lesions idx -> fp16-rounded fp32."""
    bottom.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            out.append(bottom(_batch([P], idx[i : i + bs], None)[0]).float())
    return torch.cat(out).half().float()


def scores(y, idx, logits):
    p = F.softmax(logits, 1).cpu().numpy()
    m = metrics_from_proba(y[idx], p)
    return {k: v * 100 for k, v in m.items()}, p


def load():
    les = lesions()
    y = les["dx7"].to_numpy()
    parties = [np.load(f"{ROOT}/img_clin.npy"), np.load(f"{ROOT}/img_derm.npy")]
    return y, folds(y), parties


def resume(name):
    path = os.path.join(OUT, f"{name}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    npz = os.path.join(OUT, f"{name}_probs.npz")
    probs = dict(np.load(npz)) if os.path.exists(npz) else {}
    return res, probs, npz


def flush(name, res, probs, npz):
    """Probabilities first (atomic rename), then the JSON that carries "done": a fold marked done always has its npz."""
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, **probs)
    os.replace(tmp, npz)
    save_json(f"{name}.json", res)


def finished(res, probs, f, keys):
    return res.get(str(f), {}).get("done") and all(k in probs for k in keys)


# ------------------------------------------------------------------ split learning with complete logs
def run_split(fold_ids):
    y, Fo, parties = load()
    name = "milk_rounds_split"
    res, probs, npz = resume(name)
    for f in fold_ids:
        if finished(res, probs, f, (f"f{f}_final", f"f{f}_te")):
            continue
        tr, va, te = Fo[f]
        seed_all(f)
        torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
        model = SplitModel([KIND, KIND], NCLS).to(DEV).to(memory_format=torch.channels_last)
        groups = [{"params": [p for p in b.parameters() if p.requires_grad], "lr": LR_LORA} for b in model.bottoms]
        groups.append({"params": model.top.parameters(), "lr": LR_TOP})
        opt = torch.optim.AdamW(groups, weight_decay=WD)
        spe = n_steps(len(tr))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=EPOCHS * spe, pct_start=0.15)
        w = torch.tensor(class_weights(y[tr], NCLS), dtype=torch.float32, device=DEV)
        yt = torch.as_tensor(y, device=DEV)
        g = torch.Generator().manual_seed(f)
        d = sum(b.dim for b in model.bottoms)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        log, best, best_state, step, msgs, nbytes, t0 = [], float("inf"), None, 0, 0, 0, time.time()
        run_loss, run_n = 0.0, 0

        def evaluate(ep, eligible=True):
            """eligible: an end of epoch, the only checkpoints stage 11 (pe_split.train_split) can select."""
            nonlocal msgs, nbytes, best, best_state
            model.eval()
            lv, lt = predict(model, parties, va), predict(model, parties, te)
            msgs += 1  # validation pass: one upload of the validation embeddings (the test pass is not part of training)
            nbytes += len(va) * d * EMB_BYTES
            vl = F.cross_entropy(lv, yt[va]).item()
            vs, _ = scores(y, va, lv)
            ts, pt = scores(y, te, lt)
            rec = {"epoch": ep, "step": step, "train_loss": run_loss / max(run_n, 1), "val_loss": vl, **{f"val_{k}": v for k, v in vs.items()},
                   **{f"test_{k}": v for k, v in ts.items()}, "messages": msgs, "bytes": nbytes, "secs": time.time() - t0}
            log.append(rec)
            probs[f"f{f}_step{step}"] = pt.astype(np.float32)
            rec["eligible"] = eligible
            if eligible and vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in trainable or "running" in k or "num_batches" in k}
                rec["best_so_far"] = True
            res[str(f)] = {"log": log, "done": False}
            flush(name, res, probs, npz)
            print(f"split f{f} ep {ep:.2f} step {step} val loss {vl:.4f} val F1 {vs['f1']:.2f} test F1 {ts['f1']:.2f} msgs {msgs} ({rec['secs']:.0f}s)", flush=True)
            model.train()

        for ep in range(EPOCHS):
            model.train()
            perm = tr[torch.randperm(len(tr), generator=g).numpy()]
            for i in range(0, len(perm), BS):
                idx = perm[i : i + BS]
                if len(idx) < 2:
                    continue
                xs = [_augment(x, g) for x in _batch(parties, idx, None)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                    loss = F.cross_entropy(model(xs).float(), yt[idx], weight=w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                msgs += 2  # activations up, gradients down
                nbytes += 2 * len(idx) * d * EMB_BYTES
                run_loss, run_n = run_loss + loss.item() * len(idx), run_n + len(idx)
                if ep == 0 and step in (5, 10, 25):
                    evaluate(step / spe, eligible=False)
                    run_loss, run_n = 0.0, 0
            evaluate(ep + 1)
            run_loss, run_n = 0.0, 0
        model.load_state_dict(best_state, strict=False)
        model.eval()
        lt = predict(model, parties, te)
        fin, pt = scores(y, te, lt)
        probs[f"f{f}_final"] = pt.astype(np.float32)
        best_rec = min((r for r in log if r["eligible"]), key=lambda r: r["val_loss"])
        res[str(f)] = {"log": log, "final": fin, "final_from_step": best_rec["step"], "messages_total": msgs, "bytes_total": nbytes,
                       "steps_per_epoch": spe, "secs": time.time() - t0, "gpu": torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu",
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0, "done": True}
        probs[f"f{f}_te"] = te
        print(f"split f{f} final (val-selected step {best_rec['step']}) test F1 {fin['f1']:.2f} acc {fin['acc']:.2f}", flush=True)
        flush(name, res, probs, npz)
        del model, opt
        torch.cuda.empty_cache() if DEV.type == "cuda" else None


# ------------------------------------------------------------------ few rendezvous
def run_rounds(fold_ids, R, E, local="target", shrink=0.5):
    y, Fo, parties = load()
    name = f"milk_rounds_R{R}E{E}" if local == "linear" else f"milk_rounds_{local}_R{R}E{E}"
    res, probs, npz = resume(name)
    yt = torch.as_tensor(y, device=DEV)
    assert R >= 1 and E >= 1, "round 0 (the one upload) is part of every run"
    for f in fold_ids:
        if finished(res, probs, f, (f"f{f}_te", f"f{f}_r{R}_fresh")):
            continue
        tr, va, te = Fo[f]
        allx = np.arange(len(y))
        seed_all(f)
        torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
        model = SplitModel([KIND, KIND], NCLS).to(DEV).to(memory_format=torch.channels_last)
        bottoms, top = model.bottoms, model.top
        spe = n_steps(len(tr))
        opt_p = [torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=LR_LORA, weight_decay=WD) for b in bottoms]
        opt_t = torch.optim.AdamW(top.parameters(), lr=LR_TOP, weight_decay=WD)
        sch_p = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=LR_LORA, total_steps=R * E * spe, pct_start=0.15) for o in opt_p]
        sch_t = torch.optim.lr_scheduler.OneCycleLR(opt_t, max_lr=LR_TOP, total_steps=R * E * spe, pct_start=0.15)
        cw = class_weights(y[tr], NCLS)
        w = torch.tensor(cw, dtype=torch.float32, device=DEV)
        g = torch.Generator().manual_seed(f)
        dims = [b.dim for b in bottoms]
        log, msgs, nbytes, nbytes_test, t0 = [], 0, 0, 0, time.time()

        def upload(r, extra=None):
            nonlocal msgs, nbytes, nbytes_test
            H = torch.cat([embed(b, P, allx) for b, P in zip(bottoms, parties)], 1)  # [N, 1536]
            msgs += 1  # training and validation embeddings (test embeddings travel with them; counted apart, inference)
            nbytes += (len(tr) + len(va)) * sum(dims) * EMB_BYTES
            nbytes_test += len(te) * sum(dims) * EMB_BYTES
            top.eval()
            with torch.no_grad():
                lv, lt = top(H[va]), top(H[te])
            rec = {"round": r, "messages": msgs, "bytes": nbytes, "bytes_test_inference": nbytes_test, "secs_uploaded": time.time() - t0, **(extra or {})}
            if r > 0:  # the warm top (trained in the rounds); round 0 has none yet
                vs, _ = scores(y, va, lv)
                ts, pt = scores(y, te, lt)
                rec.update({"warm_val_loss": F.cross_entropy(lv, yt[va]).item(), **{f"warm_val_{k}": v for k, v in vs.items()},
                            **{f"warm_test_{k}": v for k, v in ts.items()}})
                probs[f"f{f}_r{r}_warm"] = pt.astype(np.float32)
            Z = torch.as_tensor(zscore(H.cpu().numpy(), tr), device=DEV)  # one-upload head of stage 11 on these embeddings
            m = train_head(Z[tr], yt[tr], Z[va], yt[va], NCLS, arch="mlp2", seed=f, class_weight=cw)
            with torch.no_grad():
                pv, pt = predict_proba(m, Z[va]), predict_proba(m, Z[te])
            vs = {k: v * 100 for k, v in metrics_from_proba(y[va], pv.cpu().numpy()).items()}
            ts = {k: v * 100 for k, v in metrics_from_proba(y[te], pt.cpu().numpy()).items()}
            rec.update({"fresh_val_loss": F.nll_loss(torch.log(pv.clamp_min(1e-12)), yt[va]).item(), "fresh_epochs": m.epochs_run,
                        **{f"fresh_val_{k}": v for k, v in vs.items()}, **{f"fresh_test_{k}": v for k, v in ts.items()}})
            probs[f"f{f}_r{r}_fresh"] = pt.cpu().numpy().astype(np.float32)
            rec["secs"] = time.time() - t0  # scores available
            log.append(rec)
            print(f"R{R}E{E} f{f} round {r}: fresh val F1 {vs['f1']:.2f} test F1 {ts['f1']:.2f}" +
                  (f" | warm test F1 {rec['warm_test_f1']:.2f}" if r > 0 else "") + f" msgs {msgs} ({rec['secs']:.0f}s)", flush=True)
            res[str(f)] = {"log": log, "done": False}
            flush(name, res, probs, npz)
            return H

        H = upload(0)
        for r in range(1, R + 1):
            # label owner: E epochs of the top on the uploaded (fixed) training embeddings, then the gradients
            top.train()
            Htr, ytr = H[tr], yt[tr]
            tl = 0.0
            for _ in range(E):
                perm = torch.randperm(len(tr), generator=g).to(DEV)
                for i in range(0, len(tr), BS):
                    idx = perm[i : i + BS]
                    if len(idx) < 2:
                        continue
                    loss = F.cross_entropy(top(Htr[idx]), ytr[idx], weight=w)
                    opt_t.zero_grad(set_to_none=True)
                    loss.backward()
                    opt_t.step()
                    sch_t.step()
                    tl += loss.item() * len(idx)
            top.eval()
            Hg = Htr.clone().requires_grad_(True)
            per = F.cross_entropy(top(Hg), ytr, reduction="none") * w[ytr]
            (per.sum() / w[ytr].mean()).backward()  # per-sample gradients of w_i * CE_i (normalised by the mean weight)
            G = Hg.grad.detach()
            eta, ls = 0.0, {}
            if local == "target":
                # targets t_i = h_i - eta g_i, eta: the smallest step of a log grid at which the (fixed) top's weighted training
                # loss falls to `shrink` times its current value (argmin of the grid if never); fitted lesions barely move
                with torch.no_grad():
                    def wl(e):
                        return float((F.cross_entropy(top(Htr - e * G), ytr, reduction="none") * w[ytr]).sum() / w[ytr].sum())
                    l0 = wl(0.0)
                    ls = {f"{e:.3g}": wl(e) for e in np.logspace(-1, 5, 61)}
                    ok = [float(e) for e, v in ls.items() if v <= shrink * l0]
                    eta = min(ok) if ok else float(min(ls, key=ls.get))
                    l_eta = wl(eta)
                G = (Htr - eta * G).half().float()  # the targets, fp16 on the wire
            else:
                G = G.half().float()  # the gradients, fp16 on the wire
            msgs += 1
            nbytes += len(tr) * sum(dims) * EMB_BYTES
            # parties: E epochs of local LoRA steps, regressing on the targets (MSE) or on the fixed gradients (linear surrogate)
            off, pl = 0, []
            pos = {int(j): n for n, j in enumerate(tr)}
            for k, (b, P) in enumerate(zip(bottoms, parties)):
                Gk = G[:, off : off + dims[k]]
                off += dims[k]
                b.train()
                lsum = 0.0
                for _ in range(E):
                    perm = tr[torch.randperm(len(tr), generator=g).numpy()]
                    for i in range(0, len(perm), BS):
                        idx = perm[i : i + BS]
                        if len(idx) < 2:
                            continue
                        x = _augment(_batch([P], idx, None)[0], g)
                        gi = Gk[torch.as_tensor([pos[int(j)] for j in idx], device=DEV)]
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                            z = b(x).float()
                        loss = F.mse_loss(z, gi) if local == "target" else (z * gi).sum(1).mean()
                        opt_p[k].zero_grad(set_to_none=True)
                        loss.backward()
                        opt_p[k].step()
                        sch_p[k].step()
                        lsum += loss.item()
                pl.append(lsum)
            H = upload(r, {"top_train_loss": tl / (E * len(tr)), "party_local_loss_sum": pl, "local": local, "eta": eta,
                           "train_wloss_before": l0 if ls else None, "train_wloss_at_targets": l_eta if ls else None,
                           "message_down_rms": float(G.pow(2).mean().sqrt())})
        # protocols that stop after r rounds: best round <= r on the validation loss of the fresh head
        sel = []
        for r in range(R + 1):
            b = min(log[: r + 1], key=lambda q: q["fresh_val_loss"])
            sel.append({"rounds": r, "messages": 1 + 2 * r, "selected_round": b["round"], **{k[6:]: v for k, v in b.items() if k.startswith("fresh_test_")}})
        res[str(f)] = {"log": log, "stop_after_r": sel, "R": R, "E": E, "messages_total": msgs, "bytes_total": nbytes, "bytes_test_inference": nbytes_test,
                       "secs": time.time() - t0,
                       "gpu": torch.cuda.get_device_name() if DEV.type == "cuda" else "cpu",
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else 0.0, "done": True}
        probs[f"f{f}_te"] = te
        flush(name, res, probs, npz)
        del model, opt_p, opt_t
        torch.cuda.empty_cache() if DEV.type == "cuda" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "rounds"])
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--E", type=int, default=1)
    ap.add_argument("--local", default="target", choices=["target", "linear"])
    ap.add_argument("--shrink", type=float, default=0.5)
    a = ap.parse_args()
    fo = [int(x) for x in a.folds.split(",")]
    run_split(fo) if a.mode == "split" else run_rounds(fo, a.R, a.E, a.local, a.shrink)


if __name__ == "__main__":
    main()
