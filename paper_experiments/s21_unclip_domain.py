"""Stage 21 - a stronger generic decoder for the CLIP-L uploads of VertiBench: Stable unCLIP-L adapted to the domain
with public images, and best-of-N sampling.

Stage 20 applied the public decoder as it is.  A determined attacker would (1) fine-tune it on public images of the
domain (Sentinel-2 is open data; here the training images of the benchmark stand for them, the test images are never
used), encoded with the public encoder and the published preprocessing, and (2) draw several images per upload and keep
the one whose CLIP embedding, computed with the same public encoder and preprocessing, is closest to the upload.  We
also report a retrieval attacker that needs no decoder: the public image whose embedding is nearest to the upload.

The decoder is trained to output the RGB bands as the encoder saw them, up to a fixed gain (x * GAIN / 255, clipped):
the encoder input is then an exact affine function of the decoded image, which makes the best-of-N criterion exact.
Evaluation is that of stage 20 (DINOv2-B re-identification among the 785 test images of the party, label inference,
pixel metrics), on the same 160 test images.
  python s21_unclip_domain.py            (PE_FT_MINUTES: training budget, default 60; PE_FT_SMOKE=1: a few steps)
Outputs: generic_inversion_ft.json, generic_inversion_ft_examples.npz.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from pe_common import DEV, OUT, seed_all
from s20_generic_inversion import CACHE, NEX, Decoder, Judge, evaluate, label_inference, mean_colour, resize_u8, thumbs16
from s7_vb_encoders import MEAN, RGB, STD

SMOKE = bool(os.environ.get("PE_FT_SMOKE"))
MINUTES = float(os.environ.get("PE_FT_MINUTES", "60"))
MAX_STEPS = 6 if SMOKE else int(os.environ.get("PE_FT_STEPS", "6000"))
BS, LR, SIZE, GAIN = 32, 1e-5, 256, 1.6
N_VB = 8 if SMOKE else int(os.environ.get("PE_GI_NVB", "160"))
NBEST = 4
N_VAL = 8 if SMOKE else 32
GUID = (3.0,) if SMOKE else (1.5, 3.0, 6.0, 10.0)
STEPS = 25
PATH = os.path.join(OUT, "generic_inversion_ft" + ("_smoke" if SMOKE else "") + ".json")
NPZ = os.path.join(OUT, "generic_inversion_ft" + ("_smoke" if SMOKE else "") + "_examples.npz")
M3, S3 = MEAN[RGB].view(1, 3, 1, 1), STD[RGB].view(1, 3, 1, 1)


def save(res, ex=None):
    json.dump(res, open(PATH, "w"), indent=1)
    if ex:
        np.savez_compressed(NPZ, **ex)
    print("saved", PATH, flush=True)


def to_target(rgb_u8):
    """uint8 RGB [n,3,h,w] (raw L1C bands) -> decoder target in [0,1] at SIZE."""
    x = torch.as_tensor(np.ascontiguousarray(rgb_u8)).to(DEV).float() * GAIN / 255
    return F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False).clamp(0, 1)


def to_raw_u8(t):
    """decoded images in [0,1] (target space) -> uint8 raw-band estimate (values clipped by the gain are lost)."""
    return (t * 255 / GAIN).clamp(0, 255).round().byte().cpu().numpy()


class Domain:
    def __init__(self, dec):
        self.dec, self.pipe = dec, dec.pipe
        ids = self.pipe.tokenizer([""], padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.pe = self.pipe.text_encoder(ids.to(DEV))[0]  # empty prompt, as at decoding time
        self.sf = self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def latents(self, t):
        return self.pipe.vae.encode((t * 2 - 1).half()).latent_dist.mean * self.sf

    @torch.no_grad()
    def clip_embed(self, t):
        """CLIP embedding of decoded images, computed as the party computes its upload (band standardisation, 224 px)."""
        raw = t.float() * 255 / GAIN
        z = F.interpolate((raw - M3.to(DEV)) / S3.to(DEV), size=(224, 224), mode="bilinear", align_corners=False)
        return self.dec.W(self.dec.enc.vision_model(pixel_values=z.half()).pooler_output).float()

    def finetune(self, Lat, E, minutes):
        from diffusers import DDPMScheduler

        pipe = self.pipe
        sched = DDPMScheduler.from_config(pipe.scheduler.config)
        pred = sched.config.prediction_type
        unet = pipe.unet.float()
        unet.enable_gradient_checkpointing()
        unet.train()
        opt = torch.optim.AdamW(unet.parameters(), lr=LR, weight_decay=1e-2)
        g = torch.Generator(device="cpu").manual_seed(0)
        t0, step, losses = time.time(), 0, []
        while step < MAX_STEPS and (time.time() - t0) / 60 < minutes:
            idx = torch.randint(0, len(Lat), (BS,), generator=g).to(DEV)
            x0 = Lat[idx].float()
            with torch.no_grad():
                cl = pipe.noise_image_embeddings(E[idx].half(), noise_level=0).float()
                cl[torch.rand(BS, device=DEV) < 0.1] = 0  # conditioning dropout: the unconditional branch of guidance
            noise = torch.randn_like(x0)
            ts = torch.randint(0, sched.config.num_train_timesteps, (BS,), device=DEV)
            xt = sched.add_noise(x0, noise, ts)
            target = sched.get_velocity(x0, noise, ts) if pred == "v_prediction" else noise
            for p_ in opt.param_groups:
                p_["lr"] = LR * min(1.0, (step + 1) / 200)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = unet(xt, ts, encoder_hidden_states=self.pe.expand(BS, -1, -1), class_labels=cl).sample
            loss = F.mse_loss(out.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0 or SMOKE:
                print(f"step {step} loss {np.mean(losses[-100:]):.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        unet.eval()
        unet.half()
        del opt
        torch.cuda.empty_cache()
        return {"steps": step, "minutes": (time.time() - t0) / 60, "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
                "prediction_type": pred, "batch": BS, "lr": LR, "size": SIZE}

    @torch.no_grad()
    def decode(self, E, guidance, seed, bs=16):
        out = []
        for i in range(0, len(E), bs):
            e = E[i : i + bs].half()
            gen = torch.Generator(device=DEV).manual_seed(seed * 100003 + i)
            im = self.pipe(prompt=[""] * len(e), image_embeds=e, num_inference_steps=STEPS, guidance_scale=guidance, height=SIZE, width=SIZE,
                           generator=gen, output_type="pt").images
            out.append(im.float().clamp(0, 1))
        return torch.cat(out)


def main():
    seed_all(0)
    res = {"gain": GAIN, "size": SIZE, "n_best": NBEST}
    dec, judge = Decoder(), Judge()
    res["decoder"] = getattr(dec, "repo", None)
    dom = Domain(dec)
    A = {s: np.load(f"{CACHE}/full13_{s}.npy", mmap_mode="r") for s in ("train", "test")}
    Fc = {s: np.load(f"{CACHE}/f7_clip_l14_224_{s}.npy", mmap_mode="r") for s in ("train", "test")}
    y = {s: np.load(f"{CACHE}/labels_{s}.npy") for s in ("train", "test")}
    L, P = A["train"].shape[:2]
    perm = np.random.default_rng(0).permutation(L)
    val_loc, ft_loc = np.sort(perm[:200]), np.sort(perm[200:])
    if SMOKE:
        ft_loc = ft_loc[:40]
    # ---- the attacker's public in-domain data: training images of the other locations, all dates, and their uploads
    t0 = time.time()
    Lat, E = [], []
    for i in range(0, len(ft_loc), 32):
        loc = ft_loc[i : i + 32]
        x = np.ascontiguousarray(A["train"][loc][:, :, RGB]).reshape(-1, 3, *A["train"].shape[-2:])
        for j in range(0, len(x), 128):
            Lat.append(dom.latents(to_target(x[j : j + 128])).half())
        E.append(dec.project(np.asarray(Fc["train"][loc], dtype=np.float32).reshape(-1, Fc["train"].shape[-1])).half())
    Lat, E = torch.cat(Lat), torch.cat(E)
    res["n_public_images"] = int(len(Lat))
    res["encode_minutes"] = (time.time() - t0) / 60
    print("public pairs", tuple(Lat.shape), f"{res['encode_minutes']:.1f} min", flush=True)
    # ---- evaluation data (as stage 20: party 0, 160 evenly spaced test images)
    p = 0
    rgb_te = np.ascontiguousarray(A["test"][:, p][:, RGB])
    sel = np.linspace(0, len(y["test"]) - 1, N_VB).round().astype(int)
    Ete = dec.project(np.asarray(Fc["test"][sel, p], dtype=np.float32))
    G_te = judge.feats(rgb_te, stretched=True)
    G_tr = judge.feats(np.ascontiguousarray(A["train"][:, p][:, RGB]), stretched=True)
    clf = (G_tr, y["train"], y["test"][sel])
    T16 = thumbs16(rgb_te)
    ex = {"original": resize_u8(rgb_te[sel[:NEX]], 128)}
    res["mean_colour"] = evaluate(mean_colour(rgb_te[sel]), rgb_te[sel], None, sel, None, True, None, T16)
    # ---- retrieval attacker: the public image whose CLIP embedding is nearest to the upload (no decoder)
    En = F.normalize(E.float(), dim=1)
    nn_idx = (F.normalize(Ete, dim=1) @ En.T).argmax(1).cpu().numpy()
    nn_loc, nn_date = ft_loc[nn_idx // P], nn_idx % P
    nn_img = np.stack([np.asarray(A["train"][l_, d_][RGB]) for l_, d_ in zip(nn_loc, nn_date)])
    res["nearest public image"] = evaluate(nn_img, rgb_te[sel], G_te, sel, judge, True, clf, T16)
    res["nearest public image"]["label_match"] = float((y["train"][nn_loc] == y["test"][sel]).mean() * 100)
    ex["nearest_public"] = resize_u8(nn_img[:NEX], 128)
    print("nearest", json.dumps(res["nearest public image"])[:300], flush=True)
    save(res, ex)
    # ---- off-the-shelf decoder in the same output space, before fine-tuning (one sample, guidance 10)
    r0 = dom.decode(Ete, 10.0, 0)
    res["off-the-shelf (256 px)"] = evaluate(to_raw_u8(r0), rgb_te[sel], G_te, sel, judge, True, clf, T16)
    res["off-the-shelf (256 px)"]["clip_cos"] = float(F.cosine_similarity(dom.clip_embed(r0), Ete).mean())
    ex["off_the_shelf"] = resize_u8(to_raw_u8(r0[:NEX]), 128)
    save(res, ex)
    # ---- fine-tuning on the public pairs
    res["finetune"] = dom.finetune(Lat, E, MINUTES)
    del Lat
    torch.cuda.empty_cache()
    save(res, ex)
    # ---- guidance chosen on held-out public images, by the embedding consistency the attacker can compute
    vl = val_loc[np.linspace(0, len(val_loc) - 1, N_VAL).round().astype(int)]
    Ev = dec.project(np.asarray(Fc["train"][vl, p], dtype=np.float32))
    res["guidance_val"] = {}
    for gsc in GUID:
        rv = dom.decode(Ev, gsc, 1)
        res["guidance_val"][str(gsc)] = float(F.cosine_similarity(dom.clip_embed(rv), Ev).mean())
        print("guidance", gsc, res["guidance_val"][str(gsc)], flush=True)
    gbest = float(max(res["guidance_val"], key=res["guidance_val"].get))
    res["guidance"] = gbest
    # ---- test: NBEST samples per upload, the first one and the best by CLIP consistency
    samples, cos = [], []
    for s in range(NBEST):
        r = dom.decode(Ete, gbest, 10 + s)
        samples.append(r.cpu())
        cos.append(F.cosine_similarity(dom.clip_embed(r), Ete).cpu())
        print("sample", s, float(cos[-1].mean()), flush=True)
    cos = torch.stack(cos)  # [NBEST, n]
    best = cos.argmax(0)
    first, chosen = samples[0], torch.stack([samples[b][i] for i, b in enumerate(best.tolist())])
    for name, r, c in (("fine-tuned", first, cos[0]), (f"fine-tuned, best of {NBEST}", chosen, cos.max(0).values)):
        res[name] = evaluate(to_raw_u8(r), rgb_te[sel], G_te, sel, judge, True, clf, T16)
        res[name]["clip_cos"] = float(c.mean())
        print(name, json.dumps(res[name])[:400], flush=True)
    ex["fine_tuned"] = resize_u8(to_raw_u8(first[:NEX]), 128)
    ex["fine_tuned_best"] = resize_u8(to_raw_u8(chosen[:NEX]), 128)
    res["real_label_inference"] = label_inference(G_tr, y["train"], G_te[torch.as_tensor(sel)], y["test"][sel])
    save(res, ex)


if __name__ == "__main__":
    main()
