#!/bin/bash
# Chain R2 (answers to the second review, MILK10k and CIFAR) on a fresh VM:  VB_SESSION=vr2 bash vbctl.sh launch chain 'bash run_chain_r2.sh'
#   frozen ViT-B (DINOv2-B, CLIP-B) with 8 dihedral views and late fusion, to compare with LoRA on the same backbone;
#   one early-stopping rule for every head (weighted val loss / val macro F1) and weight decay (s19, PE_REG_EXTRA);
#   ridge with the party scaling chosen on validation on MILK10k and split CIFAR (s13, ridge_sel).
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/scratch_out
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
pip install -q peft open_clip_torch tenseal > /dev/null 2>&1
cp -n ../paper_results/milk_reg.json /content/results/  # s19 adds keys to the existing file
step milk_prep s11_milk10k.py prepare
PE_OUT=/content/scratch_out step milk_feat s11_milk10k.py features --encoders dinov2_g,clip_l14,dinov2_b,clip_b16
step milk_views s19_milk_reg.py views --encoders dinov2_b,clip_b16
PE_REG_EXTRA=1 step milk_reg s19_milk_reg.py heads --encoders dinov2_b,clip_b16,dinov2_g,clip_l14
PE_OUT=/content/scratch_out step classic s14_classic.py --datasets cifar10,cifar100 --encoders dinov2_g
PE_HE_HEADS=ridge_sel PE_HE_NOCOST=1 PE_HE_OUT=he_heads_r_milk.json PE_CACHE=/content/none step s13 s13_he_heads.py
echo CHAIN_DONE
