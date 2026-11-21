#!/bin/bash
# Chain R2b, replaces the end of chain R2 on the same VM (vr2): only the top/bottom party features of CIFAR are
# needed for the ridge with validated party scaling, not the 9 pieces of stage 14.
#   VB_SESSION=vr2 bash vbctl.sh launch chain 'bash run_chain_r2b.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
PE_OUT=/content/scratch_out PE_CLASSIC_HE_ONLY=1 step classic_he s14_classic.py --datasets cifar10,cifar100 --encoders dinov2_g
PE_HE_HEADS=ridge_sel PE_HE_NOCOST=1 PE_HE_OUT=he_heads_r_milk.json PE_CACHE=/content/none step s13 s13_he_heads.py
echo CHAIN_DONE
