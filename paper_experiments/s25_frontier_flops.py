"""Stage 25 - party-side FLOPs of the bottoms that vb224_encoders.json does not cover (for the compute frontier).

  python s25_frontier_flops.py            (CPU is enough: FLOPs are counted, nothing is trained)

Counts forward FLOPs per image with the counter of s12_vb224 (torch FlopCounterMode, s12.flops_per_image) on the
models built by the training scripts themselves, at the input sizes they train on:
  splitnn_bottom_{13band,rgb}_158px   s3_baselines.resnet18(c_in) at 158 px (SplitNN, our recipe; the ImageNet-
                                      initialised bottom has the RGB architecture, weights do not change FLOPs)
  fedonce_{13band,rgb}_train_128px    s3_baselines.resnet18(c_in) + SimCLR projection (512-512-128) on a 128 px view
  fedonce_{13band,rgb}_feat_158px     the same encoder on the whole 158 px image (feature extraction)
  fedonce_nat_13band_{train_128px,feat_158px}  the same with the Noise-As-Targets projection (512 -> 64)
  milk_resnet18_224px                 pe_split.Bottom("resnet18") at 224 px (MILK10k split learning)
and recounts the VertiBench-recipe bottom of vb224_encoders.json (s12.vertibench_bottom_flops) as a check.
-> $PE_OUT/frontier_flops.json
"""

import json
import os

import torch.nn as nn
from pe_common import save_json
from pe_split import Bottom
from s3_baselines import resnet18
from s12_vb224 import flops_per_image, vertibench_bottom_flops


def enc_proj(c_in, proj):
    enc = resnet18(c_in, False, False).eval()
    return lambda x: proj(enc(x / 255)) if proj is not None else enc(x / 255)


def main():
    res = {}
    for tag, c in (("13band", 13), ("rgb", 3)):
        res[f"splitnn_bottom_{tag}_158px"] = {"flops_per_image": flops_per_image(enc_proj(c, None), (1, c, 158, 158)),
                                              "model": f"s3_baselines.resnet18({c}, small_stem=False)", "input": [c, 158, 158]}
        simclr = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).eval()
        res[f"fedonce_{tag}_train_128px"] = {"flops_per_image": flops_per_image(enc_proj(c, simclr), (1, c, 128, 128)),
                                             "model": f"s3_baselines.resnet18({c}) + SimCLR projection 512-512-128", "input": [c, 128, 128]}
        res[f"fedonce_{tag}_feat_158px"] = {"flops_per_image": flops_per_image(enc_proj(c, None), (1, c, 158, 158)),
                                            "model": f"s3_baselines.resnet18({c}), features", "input": [c, 158, 158]}
    nat = nn.Linear(512, 64).eval()
    res["fedonce_nat_13band_train_128px"] = {"flops_per_image": flops_per_image(enc_proj(13, nat), (1, 13, 128, 128)),
                                             "model": "s3_baselines.resnet18(13) + NAT projection 512-64", "input": [13, 128, 128]}
    res["fedonce_nat_13band_feat_158px"] = {"flops_per_image": flops_per_image(enc_proj(13, nat), (1, 13, 158, 158)),
                                            "model": "s3_baselines.resnet18(13) + NAT projection 512-64, features", "input": [13, 158, 158]}
    b = Bottom("resnet18").eval()
    res["milk_resnet18_224px"] = {"flops_per_image": flops_per_image(lambda x: b(x / 255), (1, 3, 224, 224)),
                                  "model": "pe_split.Bottom('resnet18')", "input": [3, 224, 224]}
    stored = json.load(open(os.path.join(os.path.dirname(__file__), "..", "paper_results", "vb224_encoders.json")))["cost"]
    r = vertibench_bottom_flops()
    res["check"] = {"vertibench_resnet18_bottom_158px": {"recounted": r, "stored": stored["vertibench_resnet18_bottom_158px"]["flops_per_image"],
                                                         "match": r == stored["vertibench_resnet18_bottom_158px"]["flops_per_image"]}}
    res["counter"] = "s12_vb224.flops_per_image (torch.utils.flop_counter.FlopCounterMode, 2 FLOPs per multiply-add)"
    for k, v in res.items():
        print(k, v)
    save_json("frontier_flops.json", res)


if __name__ == "__main__":
    main()
