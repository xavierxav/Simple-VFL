#!/bin/bash
# Full GPU pipeline on a fresh Colab VM (download included).  Launch detached:  bash vbctl.sh launch chain 'bash run_chain.sh A'
#   A: frozen features (s1), heads (s2), inversion (s4), timing (s6)
#   B: full-resolution end-to-end baselines (s3)
#   C: RGB baselines and encoder timing only (re-run on a second GPU)
#   D: why CLIP > DINOv2 (s7: encoder ablations) and the reproduction of the published VertiBench SplitNN (s8)
# Progress: /content/logs/chain.log (one log per step).  Ends with CHAIN_DONE (keepalive.sh then harvests and stops the VM).
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
echo "$(date +%H:%M:%S) download"
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
echo "$(date +%H:%M:%S) downloaded $(stat -c %s /content/vb_raw/Vertibench.7z)"
if [ "$1" = A ]; then
  step s1 s1_vertibench_features.py
  step s2 s2_vertibench_heads.py
  step s4 s4_inversion.py
  step s6 s6_timing.py
elif [ "$1" = D ]; then
  step s3pack s3_baselines.py pack
  step s7 s7_vb_encoders.py
  step s8 s8_vertibench_repro.py
elif [ "$1" = C ]; then
  step s3pack s3_baselines.py pack
  step s3_splitnn_rgb s3_baselines.py splitnn --bands rgb
  step s3_ssl_rgb s3_baselines.py ssl --bands rgb
  step s6 s6_timing.py
else
  step s3pack s3_baselines.py pack
  step s3_splitnn_all s3_baselines.py splitnn --bands all
  step s3_ssl_all s3_baselines.py ssl --bands all
  step s3_splitnn_rgb s3_baselines.py splitnn --bands rgb
  step s3_ssl_rgb s3_baselines.py ssl --bands rgb
fi
echo CHAIN_DONE
