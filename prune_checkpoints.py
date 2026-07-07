#!/usr/bin/env python3
"""Prune accumulating DeepSpeed resume checkpoints from a training output dir.

diffusion-pipe's `checkpoint_every_n_minutes` (plus each `save_every_n_steps`/`save_every_n_epochs`
save) writes a `global_stepN/` DeepSpeed RESUME checkpoint (optimizer + model shards, often ~0.5-2 GB
each) and does NOT prune them. Over a long run they accumulate -- e.g. ~130 over a ~65 h run -- and can
fill the volume with no warning (a network-FS `df` won't even show it; see below). You only ever resume
from the LATEST, so this keeps the newest `--keep` and deletes older ones.

It NEVER touches the exported `adapter_model.safetensors` files or `epoch*`/`step*` adapter dirs -- your
actual trained weights are safe (only the `global_step*` DeepSpeed resume state is pruned).

    # one-shot: keep the newest 3 resume checkpoints under an output dir, delete older
    python prune_checkpoints.py /path/to/output_dir --keep 3

    # run alongside a long training job: prune every 15 min until a done-marker file appears
    python prune_checkpoints.py /path/to/output_dir --keep 3 --interval 900 --until /path/to/DONE

Note: on a network filesystem (e.g. RunPod's MooseFS), `df` reports the CLUSTER capacity, not your
pod's quota, so it is useless for spotting a fill-up -- use `du -sh <dir>` to see real usage.
"""
import argparse
import glob
import os
import re
import shutil
import time

_STEP = re.compile(r"global_step(\d+)")


def _step_num(d):
    m = _STEP.search(os.path.basename(d))
    return int(m.group(1)) if m else -1


def prune_global_steps(output_dir, keep):
    """Delete all but the newest `keep` global_step* dirs under output_dir (searched recursively).
    keep <= 0 deletes ALL. Returns (found, removed). Pure file ops -> unit-testable, no torch."""
    dirs = [d for d in glob.glob(os.path.join(output_dir, "**", "global_step*"), recursive=True)
            if os.path.isdir(d)]
    dirs.sort(key=_step_num)
    victims = dirs[:-keep] if keep > 0 else list(dirs)
    for d in victims:
        shutil.rmtree(d, ignore_errors=True)
    return len(dirs), len(victims)


def main():
    ap = argparse.ArgumentParser(description="Prune old DeepSpeed resume checkpoints (global_step*).")
    ap.add_argument("output_dir", help="training output_dir (searched recursively for global_step*)")
    ap.add_argument("--keep", type=int, default=3, help="newest N resume checkpoints to keep (default 3)")
    ap.add_argument("--interval", type=int, default=0,
                    help="if >0, loop, pruning every N seconds (daemon mode)")
    ap.add_argument("--until", default=None,
                    help="marker file whose presence stops the --interval loop")
    args = ap.parse_args()
    while True:
        found, removed = prune_global_steps(args.output_dir, args.keep)
        if removed:
            print(f"[prune] {args.output_dir}: {found} global_step dirs -> removed {removed}, "
                  f"kept newest {args.keep}", flush=True)
        if args.interval <= 0:
            break
        if args.until and os.path.exists(args.until):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
