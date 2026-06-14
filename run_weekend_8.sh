#!/usr/bin/env bash
# Weekend 8 runner — one model per FRESH process.
#
# Each model runs in a fresh process so framework state and accelerator memory
# are released between models. Neural models default to CPU for stability; set
# WEEKEND8_NEURAL_DEVICE=mps explicitly to opt into Apple MPS.
#
# Usage:
#   bash run_weekend_8.sh            # full run, all models, CPU by default
#   bash run_weekend_8.sh --quick    # 2-trial smoke test of the whole pipeline
#
# Results merge into tuning_weekend_8/best_configs.json and
# tuning_weekend_8/tuned_leaderboard.csv across processes (append/replace,
# not overwrite), so the separate runs assemble one complete leaderboard.

set -euo pipefail

# Run from this script's own directory so relative paths (data/, images/,
# tuning_weekend_8/) resolve regardless of where the user launches it.
cd "$(dirname "$0")"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--quick") ]]; then
  echo "Usage: bash run_weekend_8.sh [--quick]" >&2
  exit 2
fi
QUICK_ARG=""
if [[ ${1:-} == "--quick" ]]; then
  QUICK_ARG="--quick"
fi
PY=python               # adjust if your interpreter is python3
SCRIPT=weekend_8_tuning_v9.py

echo "=============================================="
echo "Weekend 8 — per-process runner"
echo "QUICK mode: ${1:-off}"
echo "=============================================="

# 1) Baselines once (cheap, no model) — fresh process.
if [[ -n "$QUICK_ARG" ]]; then
  $PY "$SCRIPT" --baselines "$QUICK_ARG"
else
  $PY "$SCRIPT" --baselines
fi

# 2) Cheap models — each its own process. (CPU-bound; fast regardless.)
for m in sarima xgboost lightgbm; do
  echo ""
  echo ">>>>>> starting fresh process for: $m"
  if [[ -n "$QUICK_ARG" ]]; then
    $PY "$SCRIPT" --models "$m" "$QUICK_ARG"
  else
    $PY "$SCRIPT" --models "$m"
  fi
done

# 3) Neural nets — each in its own fresh process.
for m in lstm nbeats; do
  echo ""
  echo ">>>>>> starting fresh process for: $m"
  if [[ -n "$QUICK_ARG" ]]; then
    $PY "$SCRIPT" --models "$m" "$QUICK_ARG"
  else
    $PY "$SCRIPT" --models "$m"
  fi
done

echo ""
echo "=============================================="
if [[ ${1:-} != "--quick" ]]; then
  $PY weekend_8_analysis_v2.py
  echo "All models and analysis complete."
else
  echo "Quick smoke test complete; production artifacts were not modified."
fi
echo "=============================================="
