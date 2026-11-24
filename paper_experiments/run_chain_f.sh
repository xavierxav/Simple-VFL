#!/bin/bash
# Chain F (fourth review, frontier controls, s24) on a fresh L4 VM:  VB_SESSION=simple-vfl-f bash vbctl.sh launch chain 'bash run_chain_f.sh'
#   B (first, small downloads): MILK10k PCA-k x bits grid with validation-selected points and the few-round logit head
#     (C = 7 logits per party) on the frozen DINOv2-B, CLIP-B, CLIP-L, DINOv2-g features      -> milk_frontier.json
#   A: VertiBench at 224 px, ImageNet ResNet-18 / ResNet-50 features, one-shot heads and the E7 grid -> vb224_controls_resnet.json
#   The VertiBench archive (8.2 GB) downloads in the background while B runs.
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/vb_raw /content/scratch_out
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
pip uninstall -y -q torchao > /dev/null 2>&1  # the Colab image's torchao breaks recent peft (not needed here)
(
  for i in $(seq 1 30); do
    [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
    aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
      -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
  done
  echo "$(date +%H:%M:%S) VertiBench download finished" >> $L/aria.log
) &
DL=$!
# ---- B: MILK10k
step milk_prep s11_milk10k.py prepare
PE_OUT=/content/scratch_out step milk_feat s11_milk10k.py features --encoders dinov2_b,clip_b16,clip_l14,dinov2_g
step milk_frontier s24_frontier_controls.py milk --encoders dinov2_b,clip_b16,clip_l14,dinov2_g
# ---- A: VertiBench (s3pack verifies the archive size and resumes the download with curl if needed)
wait $DL
echo "$(date +%H:%M:%S) VertiBench archive ready"
step s3pack s3_baselines.py pack
step vb_resnet s24_frontier_controls.py vb --encoders resnet18_in1k_224,resnet50_in1k_224
echo CHAIN_DONE
