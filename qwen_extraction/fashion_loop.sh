#!/usr/bin/env bash
# Autonomous continuation: after the initial ~20k deepfashion run finishes, forever generate 10k
# synthetic fashion images per cycle (predominantly female + male substrate) and append them to
# AbstractPhil/qwen-deepfashion. Stop with: touch /workspace/STOP_DF. Survives session culls.
exec >> /workspace/fashion_loop.log 2>&1
cd /workspace
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="${DF_REPO:-AbstractPhil/qwen-deepfashion}"
OUT=/workspace/qwen_df_out
FCOUNT="${FCOUNT:-7000}"   # female per 10k cycle (predominantly female)
MCOUNT="${MCOUNT:-3000}"   # male substrate per 10k cycle

echo "[fashion_loop $(date -u +%F_%H:%M:%S)] started; waiting for the initial deepfashion run to finish"
sleep 30
while pgrep -f 'qwen_lightning_extraction --rank' > /dev/null || pgrep -f run_deepfashion.sh > /dev/null; do sleep 120; done
echo "[fashion_loop $(date -u +%F_%H:%M:%S)] initial run finished; beginning 10k fashion cycles (${FCOUNT}F+${MCOUNT}M)"

B=1
while [ ! -f /workspace/STOP_DF ]; do
  CAP=/workspace/floop_${B}.tsv
  # batch offset 100+B keeps ids disjoint from the initial run's synth batch 0
  python3 -m qwen_extraction.fashion_captions --gender female --count "$FCOUNT" --batch $((100 + B)) --out /workspace/floop_f.tsv || { echo "[fashion_loop] female gen FAILED cycle $B"; sleep 60; continue; }
  python3 -m qwen_extraction.fashion_captions --gender male --count "$MCOUNT" --batch $((100 + B)) --out /workspace/floop_m.tsv || { echo "[fashion_loop] male gen FAILED cycle $B"; sleep 60; continue; }
  cat /workspace/floop_f.tsv /workspace/floop_m.tsv > "$CAP"
  echo "[fashion_loop $(date -u +%F_%H:%M:%S)] cycle $B: $(wc -l < "$CAP") captions; launching 2-GPU extraction"
  for R in 0 1; do
    CUDA_VISIBLE_DEVICES=$R setsid python3 -m qwen_extraction.qwen_lightning_extraction \
      --rank $R --world-size 2 --domain fashion --offload off --batch-size 16 \
      --out-repo "$REPO" --local-dir "$OUT" --prompts-file "$CAP" \
      > /workspace/floop_${B}_rank$R.log 2>&1 < /dev/null &
  done
  sleep 60
  while pgrep -f 'qwen_lightning_extraction --rank' > /dev/null; do sleep 120; done
  echo "[fashion_loop $(date -u +%F_%H:%M:%S)] cycle $B complete"
  rm -f "$CAP" /workspace/floop_f.tsv /workspace/floop_m.tsv
  B=$((B + 1))
done
echo "[fashion_loop $(date -u +%F_%H:%M:%S)] STOP_DF seen; exiting after $B cycles"
