#!/usr/bin/env bash
# Pod-side autonomous expansion: wait for the current FFHQ run to finish, then forever generate
# 10k synthetic-caption images per cycle and append them to AbstractPhil/qwen-synth-characters.
# Stop by: touch /workspace/STOP_SYNTH   (stops after the current batch). Survives agent-session culls.
exec >> /workspace/synth_loop.log 2>&1
cd /workspace
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
export HF_HOME=/workspace/hf
COUNT="${SYNTH_COUNT:-10000}"

echo "[synth_loop $(date -u +%F_%H:%M:%S)] started (count=$COUNT); waiting for the FFHQ run to finish"
sleep 30
while pgrep -f 'qwen_lightning_extraction --rank' >/dev/null; do sleep 120; done
echo "[synth_loop $(date -u +%F_%H:%M:%S)] FFHQ run finished; beginning synthetic expansion"

B=0
while [ ! -f /workspace/STOP_SYNTH ]; do
  CAP=/workspace/synth_caps_$B.tsv
  if ! python3 /workspace/qwen_extraction/synth_captions.py --count "$COUNT" --batch "$B" --out "$CAP"; then
    echo "[synth_loop $(date -u +%H:%M:%S)] caption gen FAILED for batch $B; retrying in 60s"; sleep 60; continue
  fi
  echo "[synth_loop $(date -u +%F_%H:%M:%S)] batch $B: $(wc -l < "$CAP") captions; launching 2-GPU extraction"
  for R in 0 1; do
    CUDA_VISIBLE_DEVICES=$R setsid python3 -m qwen_extraction.qwen_lightning_extraction \
      --rank $R --world-size 2 --out-repo AbstractPhil/qwen-synth-characters \
      --local-dir /workspace/qwen_synth_out --batch-size 16 --offload off \
      --prompts-file "$CAP" > /workspace/synth_b${B}_rank$R.log 2>&1 < /dev/null &
  done
  sleep 60
  while pgrep -f 'qwen_lightning_extraction --rank' >/dev/null; do sleep 120; done
  d0=$(grep -ac "DONE" /workspace/synth_b${B}_rank0.log 2>/dev/null)
  d1=$(grep -ac "DONE" /workspace/synth_b${B}_rank1.log 2>/dev/null)
  echo "[synth_loop $(date -u +%F_%H:%M:%S)] batch $B complete (rank0 done=$d0 rank1 done=$d1)"
  rm -f "$CAP"
  B=$((B + 1))
done
echo "[synth_loop $(date -u +%F_%H:%M:%S)] STOP_SYNTH seen; exiting after $B batches"
