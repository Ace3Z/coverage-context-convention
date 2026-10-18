#!/usr/bin/env bash
# The GPU runs behind results/encoder/: the 24 selection runs (selection.json), then the 39 grid runs (summary.json).
# They run one after another on the visible GPU. To spread them over several GPUs, give each process its own
# CUDA_VISIBLE_DEVICES and a share of the loops. A selection run took 12 to 70 minutes and a grid run 13 to 152
# minutes on the GPUs of the committed runs. GPU training is not bit-for-bit repeatable, so new runs differ slightly
# from the committed ones.
# Usage (repository root, after pip install -r requirements-neural.txt):
#   bash scripts/encoder_grid.sh [select|grid|all] [RUNS_DIR]   (default: all, encoder_runs)
# At the end it rewrites results/encoder/ (selection.json, summary.json, the prediction tables) and the encoder table
# and numbers in report/generated/ from the new runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
what=${1:-all} W=${2:-encoder_runs}

if [[ $what != grid ]]; then
  # Selection (PLAN S9 and D3): sentence arm, whole sentence, seeds 1-2, with and without Zupitza, two learning rates,
  # 1, 3 or 5 epochs. Only the john_ws dev numbers enter the rule (oevl.select_config).
  for seed in 1 2; do for lr in 1e-4 3e-4; do for ep in 1 3 5; do for z in z noz; do
    ex=""; [[ $z == noz ]] && ex=zupitza_grammar   # "" trains on all five works
    R=$W/select/${z}_lr${lr}_ep${ep}/sentence_s$seed; mkdir -p "$R"
    python -m oevl.encoder_experiment --arms sentence --seed $seed --max-bytes 2560 --batch 32 --micro 8 \
      --lr $lr --epochs $ep --exclude-train "$ex" --out "$R" > "$R/train.log" 2>&1
  done; done; done; done
  python -m oevl.select_config --runs "$W/select"
fi

if [[ $what != select ]]; then
  # Grid: the chosen setting (lr 3e-4, 5 epochs, no Zupitza), seeds 1-3, at three byte caps. The word arm has no
  # context to cut, so it runs at the full cap only. local2 and local5 never exceed 103 bytes, so their runs at the
  # three caps get identical inputs (encoder_summary pools them as replicates).
  jobs=()
  for cap in 128 256 2560; do for arm in left sentence local2 local5; do for seed in 1 2 3; do
    jobs+=("$cap $arm $seed"); done; done; done
  for seed in 1 2 3; do jobs+=("2560 word $seed"); done
  for job in "${jobs[@]}"; do
    read -r cap arm seed <<< "$job"
    cfg=cap$cap; [[ $cap == 2560 ]] && cfg=capfull
    R=$W/grid/$cfg/${arm}_s$seed; S=$W/select/noz_lr3e-4_ep5/sentence_s$seed
    if [[ $cfg/$arm == capfull/sentence && -s $S/metrics.json ]]; then
      mkdir -p "$R"; cp -R "$S/." "$R/"   # same command apart from --out: the committed grid reused seeds 1-2
      continue
    fi
    mkdir -p "$R"
    python -m oevl.encoder_experiment --arms $arm --seed $seed --max-bytes $cap --batch 32 --micro 8 \
      --lr 3e-4 --epochs 5 --exclude-train zupitza_grammar --out "$R" > "$R/train.log" 2>&1
  done
  python -m oevl.encoder_summary --runs "$W/grid"
fi
