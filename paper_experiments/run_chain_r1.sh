#!/bin/bash
# Chain R1 (answers to the second review, VertiBench) on a fresh VM:  VB_SESSION=vr1 bash vbctl.sh launch chain 'bash run_chain_r1.sh'
#   revised inversion attacker (s4 with PE_INV_V2: shared decoder, AdamW, 40 epochs, attacker-side normalisation of the
#   perturbed uploads, closed-form linear decoder, per-location re-identification) and the ridge head with the party
#   scaling (per coordinate or per block) chosen on validation (s13, ridge_sel).
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
pip install -q tenseal torchgeo > /dev/null 2>&1
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
echo "$(date +%H:%M:%S) downloaded $(stat -c %s /content/vb_raw/Vertibench.7z)"
step s3pack s3_baselines.py pack
step extract -c "import s12_vb224 as s; s.extract(['dinov2_g_224', 'clip_l14_224'])"
PE_INV_V2=1 PE_INV_EPOCHS=40 PE_INV_FEAT=f7_dinov2_g_224 PE_INV_OUT=vb224_inversion_v2 step s4 s4_inversion.py
PE_HE_HEADS=ridge_sel PE_HE_NOCOST=1 PE_HE_OUT=he_heads_r_vb.json PE_MILK=/content/none PE_HE_DS=/content/none step s13 s13_he_heads.py
echo CHAIN_DONE
