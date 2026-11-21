#!/bin/bash
# Chain R5: Stable unCLIP-L adapted to the domain with public images, and best-of-4 sampling (s21) -> generic_inversion_ft.json.
# Meant to follow chain R4 on the same VM (waits for it, reuses its cache); on a fresh VM it prepares the cache itself.
#   VB_SESSION=vr4 bash vbctl.sh launch chain2 'bash run_chain_r5.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/vb_raw /content/scratch_out
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
if [ -f $L/chain.log ]; then
  until grep -qx CHAIN_DONE $L/chain.log; do sleep 30; done
  echo "$(date +%H:%M:%S) chain R4 done"
fi
if [ ! -f /content/cache/vb/f7_clip_l14_224_train.npy ]; then
  URL=https://huggingface.co/datasets/xavvierxav/L1C_dataset/resolve/main/Vertibench.7z
  which aria2c > /dev/null || apt-get install -y -q aria2 > /dev/null 2>&1
  which 7z > /dev/null || apt-get install -y -q p7zip-full > /dev/null 2>&1
  pip install -q torchgeo > /dev/null 2>&1
  for i in $(seq 1 30); do
    [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
    aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
      -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
  done
  step s3pack s3_baselines.py pack
  step extract -c "import s12_vb224 as s; s.extract(['clip_l14_224'])"
fi
python -c "from diffusers import StableUnCLIPImg2ImgPipeline" > /dev/null 2>&1 || pip install -q "diffusers==0.31.0" >> $L/pip.log 2>&1
PE_OUT=/content/scratch_out PE_FT_SMOKE=1 step ft_smoke s21_unclip_domain.py
if grep -q '"fine-tuned, best of' /content/scratch_out/generic_inversion_ft_smoke.json 2> /dev/null; then
  PE_FT_MINUTES=60 step ft s21_unclip_domain.py
else
  echo "smoke test failed, see $L/ft_smoke.log"
fi
echo CHAIN_DONE
