# Running the Qwen-Image Lightning extraction on a GPU pod

Generates synthetic character images from FFHQ llava prompts with race-bias mitigation +
age constraint, and writes an HF Image parquet dataset (Hub-viewable + diffusion-pipe
ingestible) to `AbstractPhil/qwen-synth-characters`. See `qwen_lightning_extraction.py`
header and `prompt_policy.py` for the policy.

## Deps (the diffusion-pipe pod already has these)
`diffusers>=0.38, torch, huggingface_hub, pyarrow, datasets, pillow, numpy`.
Optional for 48GB cards: `bitsandbytes` (int8) — but prefer bigger cards + `--offload off`.

## Throughput note (why card size matters)
Qwen-Image is **~20B (~44GB bf16 pipeline)**.
- On **>=80GB** cards (A100-80G / H100): it fits **resident** → use `--offload off` (fastest);
  raise `--batch-size` (try 16-32 @1024², less for tall/wide). H100 also has native FP8 if we
  later want it.
- On a **48GB A40** it does NOT fit resident → `--offload on` (cpu offload), but that re-stages
  the 40GB DiT every batch → ~28 s/img (~60h for 40,848 on 4 GPUs). Avoid for the full set.
- `--offload auto` (default) tries resident, falls back to offload on OOM.

## Setup on a fresh pod
```bash
# 1. get the code (clone the fork's branch, or scp the qwen_extraction/ dir + utils/)
cd /workspace && git clone -b feat/parquet-hf-dataset-backend <repo-or-scp> diffusion-pipe
cd /workspace/diffusion-pipe
# 2. env
export HF_TOKEN=hf_xxx            # needs write access to AbstractPhil/qwen-synth-characters
export HF_HOME=/workspace/hf      # persist the ~85GB model download on /workspace
```

## Smoke test (1 GPU, 64 imgs, no upload)
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m qwen_extraction.qwen_lightning_extraction \
  --rank 0 --world-size 1 --limit 64 --batch-size 8 --offload off \
  --no-upload --local-dir /workspace/qwen_smoke
# inspect: python3 /workspace/inspect_shard.py /workspace/qwen_smoke/rank0/shard_r0_00000.parquet
```
Check: per-image time (target <2 s/img resident), race spread, images decode + non-black.

## Full run (4 GPUs -> AbstractPhil/qwen-synth-characters)
```bash
export HF_TOKEN=hf_xxx HF_HOME=/workspace/hf
for R in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$R setsid python3 -m qwen_extraction.qwen_lightning_extraction \
    --rank $R --world-size 4 --out-repo AbstractPhil/qwen-synth-characters \
    --local-dir /workspace/qwen_synth_out --batch-size 16 --offload off \
    > /workspace/qwen_rank$R.log 2>&1 < /dev/null &
done
```
Resumable: re-run the same command; `--resume` (default) skips ids already in completed shards.
Each rank owns `data/rank{R}/` in the repo so the 4 ranks never collide. PNG (lossless),
~350MB shards, streaming upload with retry.

## After generation
1. Confirm the Hub dataset viewer renders thumbnails (image is an HF Image feature).
2. Train with `examples/qwen_synth_characters_dataset.toml`.
3. Run the **separate strong age-verification filter + child-tag snip** before using as a
   diffusion-pretrain input (this script only prompt-constrains age 25-35 + de-minors + records
   `age_band` metadata; it does not filter — that is a deliberate later step).
4. Optional re-imposition: build a race-balanced repeat manifest from the emitted `race` column
   via `utils/subject_bucket.dampened_repeats`, then set `bucket_manifest=` in the dataset TOML.

## Actual run config used (RTX PRO 6000 Blackwell, 96GB x2)
2 GPUs, `--offload off` (resident, ~1.7 s/img/GPU @ batch 16). Qwen-Image is ~20B/~56GB — fits
resident on 96GB; needs torch w/ sm_120 (cap (12,0)). Deploy code via base64-over-ssh (scp was
flaky): `B64=$(base64 -w0 <f); ssh "echo $B64|base64 -d>dest"`. HF_TOKEN read from /proc/1/environ.

## Synthetic expansion (grow the set beyond the 40,848 FFHQ prompts)
`synth_captions.py` makes diverse combinatorial character captions; `--prompts-file` feeds them in;
the same augment layer adds demographics/expression/attributes. The pod-side orchestrator
`synth_loop.sh` waits for the FFHQ run to finish, then loops 10k-image batches forever (ids
`synth_b{N}_*`, appended to the same repo/local-dir), until `touch /workspace/STOP_SYNTH` or pod death.
```bash
# one synthetic batch by hand:
python3 qwen_extraction/synth_captions.py --count 10000 --batch 0 --out /workspace/caps0.tsv
for R in 0 1; do CUDA_VISIBLE_DEVICES=$R setsid python3 -m qwen_extraction.qwen_lightning_extraction \
  --rank $R --world-size 2 --prompts-file /workspace/caps0.tsv --offload off --batch-size 16 \
  --out-repo AbstractPhil/qwen-synth-characters --local-dir /workspace/qwen_synth_out \
  > /workspace/cap0_r$R.log 2>&1 </dev/null & done
# or the autonomous loop: SYNTH_COUNT=10000 setsid bash qwen_extraction/synth_loop.sh &
```

## Anti-flooding & policy knobs
`--no-dedup` disables the MinHash/LSH sentence-similarity resampler; `--dedup-threshold` tunes it.
Expression/hair/eye/makeup/jewelry/quality weights + the race distribution live in
`prompt_policy.py` (PromptAugmentConfig + the module-level taxonomies); `POLICY_VERSION` stamps each row.
