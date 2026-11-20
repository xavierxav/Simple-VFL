#!/bin/bash
# Chain W3 (WorldStrat repair): the DINOv2 RGB features of Sentinel-2 had been computed on a partly written array.
# Waits for chain W and the GPU stage of chain W2, recomputes the features, then re-runs every head that uses them.
#   bash vbctl.sh launch chain3 'bash run_chain_w3.sh'      (keepalive: keepalive.sh vw 10 chain3 CHAIN3_DONE)
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
# wait for chain W (s13) and for the GPU stage of chain W2 (s16 extra); its heads would use the bad features: stop them
until grep -qx CHAIN_DONE $L/chain.log 2>/dev/null && grep -q "end ws_extra" $L/chain2.log 2>/dev/null; do sleep 60; done
pkill -f run_chain_w2.sh; pkill -f "s16_ws_fusion.py heads"; sleep 5
rm -f /content/ws2/f_s2_dinov2_g_rgb.npy /content/results/ws_sensors.json /content/results/ws_fusion.json /content/results/he_heads_ws.json
step ws_feat2 s15_worldstrat_sensors.py features
step ws_heads2 s15_worldstrat_sensors.py heads
step ws_fusion2 s16_ws_fusion.py heads
step s13_ws s13_he_heads.py
echo CHAIN3_DONE
