#!/bin/bash
# Chain R4: inversion of the CLIP-L uploads with a public, generic decoder (Stable unCLIP 2.1 small), on VertiBench and
# MILK10k, and party-side transforms of the upload against it (s20_generic_inversion.py) -> generic_inversion.json.
#   VB_SESSION=vr4 bash vbctl.sh launch chain 'bash run_chain_r4.sh'
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
pip install -q torchgeo > /dev/null 2>&1
for i in $(seq 1 30); do
  [ -f /content/vb_raw/Vertibench.7z ] && [ ! -f /content/vb_raw/Vertibench.7z.aria2 ] && break
  aria2c -q -x 16 -s 16 -k 8M -c --max-tries=50 --retry-wait=3 --timeout=30 --file-allocation=none \
    -d /content/vb_raw -o Vertibench.7z "$URL" >> $L/aria.log 2>&1 || sleep 5
done
echo "$(date +%H:%M:%S) downloaded $(stat -c %s /content/vb_raw/Vertibench.7z)"
step s3pack s3_baselines.py pack
step extract -c "import s12_vb224 as s; s.extract(['clip_l14_224'])"
step milk_prepare -c "import s11_milk10k as m; m.prepare()"
# the public decoder needs diffusers; keep the preinstalled one if it imports, otherwise a pinned version
python -c "from diffusers import StableUnCLIPImg2ImgPipeline" > /dev/null 2>&1 || pip install -q "diffusers==0.31.0" > $L/pip.log 2>&1
python -c "from diffusers import StableUnCLIPImg2ImgPipeline; import transformers, diffusers; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)" >> $L/pip.log 2>&1 \
  || pip install -q -U diffusers "huggingface-hub<1.0" >> $L/pip.log 2>&1
step gi_vb s20_generic_inversion.py vb
step gi_milk s20_generic_inversion.py milk
step gi_defences s20_generic_inversion.py defences
echo CHAIN_DONE
