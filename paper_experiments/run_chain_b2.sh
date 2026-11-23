#!/bin/bash
# Chain B2 (third review, VertiBench + MILK10k): audit of the frozen heads (kept probabilities, four-logit head, linkage
# with a re-encoded gallery, nearest public location, fingerprints), DP with per-party clipping (accuracy and inversion),
# MILK10k split learning with frozen bottoms under the LoRA recipe, CLIP-B split learning on VertiBench, and the
# VertiBench SplitNN reproduction with kept probabilities (paired tests).
#   VB_SESSION=simple-vfl-vb bash vbctl.sh launch chain 'bash run_chain_b2.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/vb_raw
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
URL=https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z
which aria2c > /dev/null || apt-get install -y -q aria2 > /dev/null 2>&1
which 7z > /dev/null || apt-get install -y -q p7zip-full > /dev/null 2>&1
pip install -q torchgeo peft > /dev/null 2>&1
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
step s3pack s3_baselines.py pack
step extract -c "import s12_vb224 as s; s.extract(['clip_l14_224', 'dinov2_g_224', 'clip_b16_224'])"
step audit_fp s22_vb_audit.py fingerprint
step audit_preds s22_vb_audit.py preds
step audit_link s22_vb_audit.py linkage
step audit_nn s22_vb_audit.py nnpublic
PE_SKIP=E1,E2,E3,E5,E6,E7 PE_MAIN=dinov2_g_224 PE_PAIRED=dinov2_g_224 PE_CONTROLS_OUT=vb224_controls_dinov2_g_dp.json step s9dp s9_controls.py
PE_INV_V2=1 PE_INV_EPOCHS=40 PE_INV_FEAT=f7_dinov2_g_224 PE_INV_OUT=vb224_inversion_v2dp PE_INV_KEEP="DP eps=64;DP eps=8" step s4dp s4_inversion.py
step milk_prep s11_milk10k.py prepare
PE_MILK_SPLIT_OUT=milk_split_v2 step milk_split s11_milk10k.py split --kinds dinov2_b_frozen,clip_b16_frozen,dinov2_b_lora,clip_b16_lora,dinov2_b_ft
PE_GC=1 PE_SEEDS=0 step vbsplit0 s22_vb_audit.py split
PE_S8_SUFFIX=_preds step s8 s8_vertibench_repro.py --seeds 0,1,2
PE_GC=1 PE_SEEDS=1,2 step vbsplit12 s22_vb_audit.py split
echo CHAIN_DONE
