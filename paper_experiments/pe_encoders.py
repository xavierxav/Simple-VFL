"""Frozen encoders for natural-colour images (MILK10k, CIFAR, ModelNet): one place for loading and normalisation.

encoder(name) -> (fn, dim): fn takes a float tensor [B,3,H,W] in [0,1] on DEV (H, W multiples of the patch size;
non-square inputs are allowed, the position embeddings are interpolated) and returns pooled features [B,dim].
Each model sees its own pre-training normalisation.  Satellite data (VertiBench) keep their own per-band
standardisation (s7_vertibench_geo.prep).
"""

import torch
import torch.nn.functional as F

from pe_common import DEV

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIPN = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
HF = {"dinov2_s": "facebook/dinov2-small", "dinov2_b": "facebook/dinov2-base", "dinov2_l": "facebook/dinov2-large",
      "dinov2_g": "facebook/dinov2-giant", "clip_b16": "openai/clip-vit-base-patch16", "clip_l14": "openai/clip-vit-large-patch14",
      "monet": "suinleelab/monet"}


def _norm(x, stats):
    m = torch.tensor(stats[0], device=x.device)[None, :, None, None]
    s = torch.tensor(stats[1], device=x.device)[None, :, None, None]
    return (x - m) / s


def encoder(name):
    from transformers import AutoModel, CLIPVisionModel, CLIPVisionModelWithProjection

    if name.startswith("dinov2"):
        m = AutoModel.from_pretrained(HF[name]).to(DEV).eval()
        return (lambda x: m(pixel_values=_norm(x, IMAGENET)).pooler_output), m.config.hidden_size
    if name in ("clip_b16", "clip_l14"):
        m = CLIPVisionModel.from_pretrained(HF[name]).to(DEV).eval()

        def fn(x):
            sq = x.shape[-1] == x.shape[-2] == m.config.image_size
            return m(pixel_values=_norm(x, CLIPN), interpolate_pos_encoding=not sq).pooler_output

        return fn, m.config.hidden_size
    if name == "monet":  # dermatology CLIP ViT-L/14 (Kim et al., Nature Medicine 2024); optional
        m = CLIPVisionModelWithProjection.from_pretrained(HF[name]).to(DEV).eval()
        return (lambda x: m(pixel_values=_norm(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False), CLIPN)).image_embeds), m.config.projection_dim
    if name == "biomedclip":  # PubMed image-text CLIP, ViT-B/16 (Zhang et al. 2023); optional
        import open_clip

        m, _, _ = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        m = m.to(DEV).eval()
        return (lambda x: m.encode_image(_norm(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False), CLIPN))), 512
    raise ValueError(name)


@torch.no_grad()
def extract(fn, images, size, bs=128, crop=None):
    """images: uint8 numpy/tensor [N,3,h,w]; resized (bilinear, antialiased) to `size` (int or (H,W)); crop(x)->x optional
    (e.g. a half of the image).  Returns float16 numpy [N,dim]."""
    import numpy as np

    out = []
    for i in range(0, len(images), bs):
        x = torch.as_tensor(np.asarray(images[i : i + bs])).to(DEV).float() / 255
        if x.shape[1] == 1:  # grey-level images
            x = x.expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=size if isinstance(size, tuple) else (size, size), mode="bilinear", antialias=True, align_corners=False)
        if crop is not None:
            x = crop(x)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            out.append(fn(x).float().cpu())
    return torch.cat(out).numpy().astype("float16")
