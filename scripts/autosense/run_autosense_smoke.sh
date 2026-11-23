#!/bin/bash
# Smoke test: one CityNav episode in autonomous mode with a DeepSeek backend.
# The model is NOT the official qwen3-vl-plus, so numbers are for engineering
# validation only and must not be compared with the paper.
set -u
ROOT=/data3/jihanuo/others/geonav
REPO=$ROOT/geonav-official
export HF_HUB_OFFLINE=1
export PYTHONPATH=$REPO
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  if [ -s /tmp/dskey ]; then
    export DEEPSEEK_API_KEY=$(tr -d '\r\n' < /tmp/dskey)
  else
    echo "缺少 DEEPSEEK_API_KEY" >&2
    exit 2
  fi
fi
mkdir -p "$ROOT/logs"
cd "$REPO"
LOG="$ROOT/logs/autosense_smoke_$(date +%Y%m%d_%H%M%S).log"
echo "LOG=$LOG"
"$ROOT/env/bin/python" "$ROOT/run_deepseek_test.py" \
  --mode eval --runtime_mode autonomous \
  --split val_seen --map_name birmingham_block_1 --episode_id 11 --ann_id 1 \
  --start_idx 0 --num_episodes 1 \
  --map_size 240 --map_meters 410 \
  --gps_noise_scale 0.0 --success_dist 20 \
  --eval_max_timestep 8 --max_tools_per_step 2 \
  --max_total_llm_calls 40 --max_episode_seconds 900 \
  --output_dir "$REPO/results/" 2>&1 | tee "$LOG"
echo "AUTOSENSE_SMOKE_EXIT=${PIPESTATUS[0]}"
