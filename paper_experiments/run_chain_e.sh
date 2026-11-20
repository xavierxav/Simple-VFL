#!/bin/bash
# Chain E (review controls) on a fresh VM:  bash vbctl.sh launch chain 'bash run_chain_e.sh'
#   s9 controls (regularisation, missing parties, 5-seed parties, exact DP mechanism, paired tests, validation-chosen
#   Pareto points, SSL4EO-S12), s4 corrected inversion attack, s3 SplitNN with ImageNet-initialised bottoms (RGB),
#   s5 spatial-split sensitivity (k = 10, 50) with train-test distances.  Ends with CHAIN_DONE.
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
cp -n ../paper_results/ws_results.json /content/results/ 2>/dev/null
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
echo "$(date +%H:%M:%S) downloaded $(stat -c %s /content/vb_raw/Vertibench.7z)"
step s3pack s3_baselines.py pack
step s9 s9_controls.py
step s4 s4_inversion.py
step s3_splitnn_rgb_pt s3_baselines.py splitnn --bands rgb --pretrained
step s5k s5_worldstrat.py --regimes random,random_cv,spatial --ks 10,50
echo CHAIN_DONE
