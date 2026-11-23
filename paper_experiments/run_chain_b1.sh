#!/bin/bash
# Chain B1 (third review, WorldStrat): rebuild the WorldStrat arrays and features, re-run every WorldStrat head with
# cross-validation folds grouped by site (tiles of the same Amnesty / ASMSpotter site in the same fold), keep the
# out-of-fold probabilities, compute image fingerprints (VertiBench <-> WorldStrat mapping), save the features to Drive.
#   VB_SESSION=simple-vfl-ws bash vbctl.sh launch chain 'bash run_chain_b1.sh'   (Drive mounted beforehand if possible)
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
export PE_WS_SUFFIX=_grouped PE_WS_GROUPED=1
step ws_prep s15_worldstrat_sensors.py prepare
step ws_fp -c "
import numpy as np, s22_vb_audit as a
np.savez_compressed('/content/results/ws_fingerprints.npz', fp=a.fingerprint(np.load('/content/ws2/s2_l1c.npy', mmap_mode='r')),
                    seen=np.load('/content/ws2/s2_seen.npy'))"
step ws_feat s15_worldstrat_sensors.py features
step ws_extra s16_ws_fusion.py extra
D=/content/drive/MyDrive/claude-projects/Simple-VFL/data/ws2_features
if [ -d /content/drive/MyDrive ]; then
  mkdir -p $D && cp /content/ws2/f_*.npy /content/ws2/aoi.csv /content/ws2/*_seen.npy /content/ws2/*.json $D/ 2>/dev/null
  cp /content/ws2/band_stats.npz /content/ws2/s2_cloud.npy $D/ 2>/dev/null
  echo "$(date +%H:%M:%S) features saved to Drive: $(du -sh $D | cut -f1)"
fi
step ws_heads s15_worldstrat_sensors.py heads
step ws_fusion s16_ws_fusion.py heads
PE_HE_OUT=he_heads_ws_grouped.json step s13_ws s13_he_heads.py
echo CHAIN_DONE
