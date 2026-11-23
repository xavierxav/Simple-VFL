#!/bin/bash
# VPS-side keep-alive and harvester for a Colab session running run_chain.sh.
# Colab Pro has no background execution: a runtime without client activity is reclaimed after ~30 min,
# so we touch it every 4 min, push partial results every ~30 min, and stop the VM as soon as the chain is done.
# usage:  setsid nohup bash paper_experiments/keepalive.sh SESSION [HOURS] [LOG] [MARKER] > ~/keepalive_SESSION.log 2>&1 &
#         (LOG: the chain log to watch, default "chain"; MARKER: its final line, default CHAIN_DONE)
# stop early: touch /tmp/stop_keepalive_SESSION
S=$1; H=${2:-10}; LOG=${3:-chain}; MARK=${4:-CHAIN_DONE}
cd /home/opencode/projects/Simple-VFL || exit 1
end=$(( $(date +%s) + H * 3600 )); n=0
while [ "$(date +%s)" -lt $end ]; do
  out=$(VB_SESSION=$S bash paper_experiments/vbctl.sh tail $LOG 4 2>&1)
  echo "$(date +%H:%M) $(echo "$out" | grep -v '^$' | head -4 | tr '\n' ' ' | cut -c1-300)"
  n=$((n + 1))
  if echo "$out" | grep -qx "$MARK"; then  # exact line: process command lines may contain the word
    VB_SESSION=$S bash paper_experiments/vbctl.sh fetch "paper: results ($S, final)"
    colab stop -s "$S"; echo "$(date +%H:%M) chain done, $S stopped"; exit 0
  fi
  # a transient connection error also reads "appears to be lost": give up only after 3 failed checks in a row
  if echo "$out" | grep -q -E "not found|appears to be lost"; then
    lost=$((${lost:-0} + 1)); [ $lost -ge 3 ] && { echo "$(date +%H:%M) session $S lost"; exit 1; }
    sleep 60; continue
  fi
  lost=0
  [ $((n % 7)) -eq 0 ] && VB_SESSION=$S bash paper_experiments/vbctl.sh fetch "paper: results ($S, partial)" > /dev/null 2>&1
  [ -f /tmp/stop_keepalive_$S ] && { echo "stop requested"; exit 0; }
  sleep 240
done
VB_SESSION=$S bash paper_experiments/vbctl.sh fetch "paper: results ($S, time limit)" > /dev/null 2>&1
colab stop -s "$S"; echo "$(date +%H:%M) time limit reached, $S stopped"
