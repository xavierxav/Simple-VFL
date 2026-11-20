#!/bin/bash
# Chain H (MILK10k + classic split benchmarks) on a fresh A100 VM:  bash vbctl.sh launch chain 'bash run_chain_h.sh'
cd "$(dirname "$0")"
L=/content/logs; mkdir -p $L /content/results /content/modelnet
step() {
  local n=$1; shift
  echo "$(date +%H:%M:%S) start $n"
  python -u "$@" > $L/$n.log 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) end $n (exit $rc)"
}
which aria2c > /dev/null || apt-get install -y -q aria2 > /dev/null 2>&1
pip install -q peft open_clip_torch tenseal > /dev/null 2>&1
step milk_prep s11_milk10k.py prepare
step milk_feat s11_milk10k.py features
step milk_heads s11_milk10k.py heads
( aria2c -q -x 8 -s 8 -c --max-tries=30 -d /content -o modelnet.tar.gz \
    "https://huggingface.co/datasets/Mestoukirdi/modelnet40_multi_view/resolve/main/modelnet40_multi_view.tar.gz" \
  && tar xzf /content/modelnet.tar.gz -C /content/modelnet && rm /content/modelnet.tar.gz ) > $L/modelnet_dl.log 2>&1 &
step classic s14_classic.py --datasets cifar10,cifar100
wait
step classic_mn s14_classic.py --datasets modelnet40
step milk_split s11_milk10k.py split
step s13 s13_he_heads.py
echo CHAIN_DONE
