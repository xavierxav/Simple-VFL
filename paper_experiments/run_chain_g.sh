#!/bin/bash
# Chain G (VertiBench at 224 px) on a fresh A100 VM:  bash vbctl.sh launch chain 'bash run_chain_g.sh'
#   s12 encoder table at 224 px + FLOPs, s9 controls (parties, missing parties, exact DP, Pareto) for DINOv2-g and
#   CLIP-L at 224 px, s10 party-pooling heads, s4 inversion on DINOv2-g at 224 px, SplitNN with ImageNet-initialised
#   bottoms (RGB), FedOnce with its own Noise-As-Targets objective (13 bands), s13 encrypted-friendly linear heads.
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/vb_raw
URL=https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
which aria2c > /dev/null || apt-get install -y -q aria2 > /dev/null 2>&1
which 7z > /dev/null || apt-get install -y -q p7zip-full > /dev/null 2>&1
pip install -q tenseal torchgeo > /dev/null 2>&1
cp -n ../paper_results/vb_controls.json /content/results/ 2>/dev/null
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
echo "$(date +%H:%M:%S) downloaded $(stat -c %s /content/vb_raw/Vertibench.7z)"
step s3pack s3_baselines.py pack
step s12 s12_vb224.py
export PE_SKIP=E1,E6
PE_MAIN=dinov2_g_224 PE_PAIRED=clip_l14_224,dinov2_g_224 PE_CONTROLS_OUT=vb224_controls_dinov2_g.json step s9g s9_controls.py
PE_MAIN=clip_l14_224 PE_SKIP=E1,E2,E5,E6 PE_CONTROLS_OUT=vb224_controls_clip_l14.json step s9c s9_controls.py
PE_POOL_NAMES=dinov2_g_224,clip_l14_224 PE_MAIN=dinov2_g_224 PE_POOL_OUT=vb224_pooling.json step s10 s10_pooling.py
PE_INV_FEAT=f7_dinov2_g_224 PE_INV_OUT=vb224_inversion step s4 s4_inversion.py
step s13 s13_he_heads.py
step s3pt s3_baselines.py splitnn --bands rgb --pretrained --seeds 0,1,2
step s3nat s3_baselines.py ssl --bands all --method nat
echo CHAIN_DONE
