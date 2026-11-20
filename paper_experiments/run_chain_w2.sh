#!/bin/bash
# Chain W2 (WorldStrat fusion study, s16) on the chain-W VM: waits for the s15 features, then runs s16.
#   bash vbctl.sh launch chain2 'bash run_chain_w2.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
until [ -f /content/ws2/feature_secs.json ]; do sleep 60; done
step ws_extra s16_ws_fusion.py extra
step ws_fusion s16_ws_fusion.py heads
echo CHAIN2_DONE
