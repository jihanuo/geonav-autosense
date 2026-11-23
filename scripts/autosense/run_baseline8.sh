#!/bin/bash
# Paired baseline: original workflow runtime, same model/map/episode/step budget.
set -u
ROOT=/data3/jihanuo/others/geonav
REPO=$ROOT/geonav-official
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO
export DEEPSEEK_API_KEY=$(tr -d '\r\n' < /tmp/dskey)
cd "$REPO"
LOG="$ROOT/logs/baseline8_$(date +%Y%m%d_%H%M%S).log"
echo "LOG=$LOG"
"$ROOT/env/bin/python" "$ROOT/run_deepseek_test.py" \
  --mode eval --runtime_mode workflow \
  --split val_seen --map_name birmingham_block_1 --episode_id 11 --ann_id 1 \
  --start_idx 0 --num_episodes 1 \
  --map_size 240 --map_meters 410 \
  --gps_noise_scale 0.0 --success_dist 20 \
  --eval_max_timestep 8 \
  --output_dir "$REPO/results/" 2>&1 | tee "$LOG"
echo "BASELINE8_DONE"
