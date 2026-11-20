#!/bin/bash
# Chain W (WorldStrat, SPOT + Sentinel-2 L1C, three labels) on a fresh VM:  bash vbctl.sh launch chain 'bash run_chain_w.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
which aria2c > /dev/null || apt-get install -y -q aria2 pigz > /dev/null 2>&1
pip install -q tifffile tenseal > /dev/null 2>&1
step ws_prep s15_worldstrat_sensors.py prepare
step ws_feat s15_worldstrat_sensors.py features
step ws_heads s15_worldstrat_sensors.py heads
step s13 s13_he_heads.py
echo CHAIN_DONE
