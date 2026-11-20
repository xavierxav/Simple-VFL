#!/bin/bash
# Chain H2 (MILK10k, the price of freezing): re-runs the split-learning kinds that failed in chain H (LoRA target names).
#   bash vbctl.sh launch chain2 'bash run_chain_h2.sh'      (keepalive: keepalive.sh vh 10 chain2 CHAIN2_DONE)
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
step milk_split2 s11_milk10k.py split --kinds dinov2_b_lora,clip_b16_lora,dinov2_b_ft
until grep -qx CHAIN_DONE $L/chain.log 2>/dev/null; do sleep 60; done  # s13 of chain H
echo CHAIN2_DONE
