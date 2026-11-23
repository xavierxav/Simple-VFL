#!/bin/bash
# VPS-side helper for a Colab session ($VB_SESSION, default "vb").  usage:
#   vbctl.sh setup | launch NAME "CMD" | tail NAME [N] | ls | fetch "commit message" | rclone
cd /home/opencode/projects/Simple-VFL || exit 1
run() {
  local out
  for i in 1 2 3 4; do
    out=$(timeout $(( ${T:-90} + 90 )) colab exec -s "${VB_SESSION:-vb}" --timeout "${T:-90}" -f paper_experiments/colab_ctl.py "$@" 2>&1)  # the CLI can hang
    if echo "$out" | grep -q "Connection was lost"; then sleep 4; else break; fi
  done
  echo "$out"
}
case "$1" in
  setup)  run --env CTL_ACTION=setup ;;
  launch) run --env CTL_ACTION=launch --env "CTL_NAME=$2" --env "CTL_CMD=$3" ;;
  tail)   run --env CTL_ACTION=tail --env "CTL_NAME=$2" --env "CTL_N=${3:-40}" ;;
  ls)     run --env CTL_ACTION=ls ;;
  fetch)  # pull results from the VM into paper_results/ and push them to the paper branch
    out=$(T=180 run --env CTL_ACTION=b64)
    mkdir -p paper_results
    echo "$out" | tr -d '\n' | sed 's/.*B64START//; s/B64END.*//' | base64 -d | tar xzf - -C paper_results
    ls -la paper_results
    git add paper_results; git commit -qm "${2:-paper: results}" || true  # push even when nothing new (earlier commits)
    git pull -q --rebase origin claude/simple-vfl-paper-publication-qx73nw && git push -q origin claude/simple-vfl-paper-publication-qx73nw; git log --oneline -1 ;;
  rclone)  # Drive for the runtime through the VPS's rclone remote gdrive: (the config is uploaded as a file, never printed)
    if command -v colab-rclone > /dev/null; then colab-rclone "${VB_SESSION:-vb}"; else bash paper_experiments/colab-rclone "${VB_SESSION:-vb}"; fi ;;
  *) echo "usage: vbctl.sh setup|launch|tail|ls|fetch|rclone" ;;
esac
