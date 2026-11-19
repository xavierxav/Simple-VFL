#!/bin/bash
# Runs every remaining GPU stage sequentially on the Colab VM as soon as stage 1 has finished.
# Launch:  bash vbctl.sh launch chain 'bash run_chain.sh'      progress: /content/logs/chain.log, one log per step.
cd "$(dirname "$0")"
L=/content/logs
until grep -q "^DONE" $L/s1.log 2>/dev/null; do
  if grep -q "Traceback" $L/s1.log; then echo "s1 failed"; exit 1; fi
  sleep 30
done
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  echo "$(date +%H:%M:%S) end $n (exit $?)"
}
step s2 s2_vertibench_heads.py
step s4 s4_inversion.py
step s3pack s3_baselines.py pack
step s3_splitnn_all s3_baselines.py splitnn --bands all
step s3_ssl_all s3_baselines.py ssl --bands all
step s3_splitnn_rgb s3_baselines.py splitnn --bands rgb
step s3_ssl_rgb s3_baselines.py ssl --bands rgb
echo CHAIN_DONE
