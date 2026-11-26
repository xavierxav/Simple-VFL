#!/bin/bash
# Chain M (MILK10k rendezvous, stage 27) on an L4 VM:  VB_SESSION=Simple-VFL-milk bash vbctl.sh launch chainm 'bash run_chain_m.sh [prep]'
#   prep: data only (download + decode, CPU); otherwise the runs, most important first (each resumes per fold).
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
  [ $rc = 0 ] || FAIL=1
  return $rc
}
FAIL=0
pip uninstall -y -q torchao > /dev/null 2>&1  # peft refuses the old torchao of the image; LoRA does not need it
pip install -q peft > /dev/null 2>&1
step milk_prep s11_milk10k.py prepare || { echo CHAINM_FAILED; exit 1; }
[ "$1" = prep ] && { echo CHAINM_PREP_DONE; exit 0; }
step rounds_R10E1 s27_milk_rounds.py rounds --R 10 --E 1
step rounds_split s27_milk_rounds.py split
[ "$1" = noextra ] || step rounds_R4E3 s27_milk_rounds.py rounds --R 4 --E 3
[ $FAIL = 0 ] && echo CHAINM_DONE || { echo CHAINM_FAILED; exit 1; }
