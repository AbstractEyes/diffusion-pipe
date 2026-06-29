#!/usr/bin/env bash
# Build the ~20k fair fashion set -> AbstractPhil/qwen-deepfashion (predominantly female + male substrate).
# Phase 1: real DeepFashion (~12,015 female). Phase 2: synthetic female expansion (rare garments).
# Phase 3: synthetic male substrate. All --domain fashion, 2-GPU, resumable (same --local-dir).
# Stop with: touch /workspace/STOP_DF. Survives session culls (setsid).
exec >> /workspace/deepfashion_run.log 2>&1
cd /workspace
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep ^HF_TOKEN= | cut -d= -f2-)
export HF_HOME=/workspace/hf
REPO="${DF_REPO:-AbstractPhil/qwen-deepfashion}"
OUT=/workspace/qwen_df_out
SRC=AbstractPhil/diffusion-pretrain-set-ft1
FEMALE_SYNTH="${FEMALE_SYNTH:-3000}"
MALE_SYNTH="${MALE_SYNTH:-5000}"
LIMIT="${DF_LIMIT:-0}"     # >0 = smoke (cap per phase)

python3 -c "from huggingface_hub import create_repo; create_repo('$REPO', repo_type='dataset', exist_ok=True); print('repo ready: $REPO')"

run2() {  # $1=phase label; rest=extra flags
  local phase="$1"; shift
  [ -f /workspace/STOP_DF ] && { echo "[df] STOP_DF set, skip $phase"; return; }
  echo "[df $(date -u +%F_%H:%M:%S)] phase=$phase flags: $*"
  for R in 0 1; do
    CUDA_VISIBLE_DEVICES=$R setsid python3 -m qwen_extraction.qwen_lightning_extraction \
      --rank $R --world-size 2 --domain fashion --offload off --batch-size 16 \
      --out-repo "$REPO" --local-dir "$OUT" ${LIMIT:+--limit $LIMIT} "$@" \
      > /workspace/df_${phase}_rank$R.log 2>&1 < /dev/null &
  done
  sleep 30
  while pgrep -f 'qwen_lightning_extraction --rank' > /dev/null; do sleep 60; done
  echo "[df $(date -u +%F_%H:%M:%S)] phase=$phase complete"
}

# Phase 1: real DeepFashion captions (captions_source_json -> deepfashion_caption), ~12,015 female
run2 deepfashion --source-dataset "$SRC" --source-config deepfashion \
  --source-column captions_source_json --source-json-path deepfashion_caption

# Phase 2: synthetic female expansion (up-weights rare garments DeepFashion lacks)
python3 -m qwen_extraction.fashion_captions --gender female --count "$FEMALE_SYNTH" --batch 0 --out /workspace/df_female.tsv
run2 synthfemale --prompts-file /workspace/df_female.tsv

# Phase 3: synthetic male substrate (male-appropriate outfits)
python3 -m qwen_extraction.fashion_captions --gender male --count "$MALE_SYNTH" --batch 0 --out /workspace/df_male.tsv
run2 synthmale --prompts-file /workspace/df_male.tsv

echo "[df $(date -u +%F_%H:%M:%S)] ALL PHASES DONE -> $REPO"
touch /workspace/DF_ALL_DONE
