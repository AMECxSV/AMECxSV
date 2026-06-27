#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/tidyvoice_raw_scores/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
SHARDS="${SHARDS:-6}"
TORCH_THREADS="${TORCH_THREADS:-3}"
INTER_OP_THREADS="${INTER_OP_THREADS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

EXTRACTION_PROTOCOL="${EXTRACTION_PROTOCOL:-protocols/tidyvoice_dev.csv}"
CALIBRATION_PROTOCOL="${CALIBRATION_PROTOCOL:-protocols/tidyvoice_calibration.csv}"
TEST_PROTOCOL="${TEST_PROTOCOL:-protocols/tidyvoice_test.csv}"
AUDIO_ROOT="${AUDIO_ROOT:-data/tidyvoice/TidyVoiceX_ASV/TidyVoiceX_Dev}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-data/embeddings}"
SCORES_DIR="${SCORES_DIR:-data/scores}"

MODELS=(
  speechbrain_ecapa_tdnn_voxceleb
  wespeaker_resnet34_cnceleb
  funasr_campplus_cn_3k
  funasr_eres2netv2_cn_200k
  hf_wavlm_base_sv_voxceleb1
  hf_wavlm_base_plus_sv_voxceleb1
)

mkdir -p "$LOG_DIR" "$SCORES_DIR"

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_DIR/run.log"
}

run_sharded_extraction() {
  local model="$1"
  local pids=()

  log "extract_start model=$model device=$DEVICE shards=$SHARDS torch_threads=$TORCH_THREADS"
  for shard in $(seq 0 $((SHARDS - 1))); do
    local shard_log="$LOG_DIR/extract_${model}_shard${shard}.log"
    (
      "$PYTHON_BIN" data_prep/extract_protocol_embeddings.py \
        --protocol "$EXTRACTION_PROTOCOL" \
        --audio-root "$AUDIO_ROOT" \
        --output-root "$EMBEDDINGS_ROOT" \
        --model "$model" \
        --device "$DEVICE" \
        --num-shards "$SHARDS" \
        --shard-index "$shard" \
        --torch-threads "$TORCH_THREADS" \
        --inter-op-threads "$INTER_OP_THREADS" \
        --progress-every "$PROGRESS_EVERY"
    ) >"$shard_log" 2>&1 &
    local pid="$!"
    pids+=("$pid")
    log "extract_shard_started model=$model shard=$shard pid=$pid log=$shard_log"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    log "extract_failed model=$model"
    return "$status"
  fi
  log "extract_done model=$model"
}

run_scores() {
  local model="$1"

  local calibration_output="$SCORES_DIR/tidyvoice_calibration_${model}.csv"
  local test_output="$SCORES_DIR/tidyvoice_test_${model}.csv"

  log "score_start model=$model split=calibration output=$calibration_output"
  "$PYTHON_BIN" data_prep/score_trial_protocol.py \
    --protocol "$CALIBRATION_PROTOCOL" \
    --embeddings-root "$EMBEDDINGS_ROOT" \
    --model "$model" \
    --output "$calibration_output" \
    --progress-every 1000000 \
    >"$LOG_DIR/score_calibration_${model}.log" 2>&1
  log "score_done model=$model split=calibration output=$calibration_output"

  log "score_start model=$model split=test output=$test_output"
  "$PYTHON_BIN" data_prep/score_trial_protocol.py \
    --protocol "$TEST_PROTOCOL" \
    --embeddings-root "$EMBEDDINGS_ROOT" \
    --model "$model" \
    --output "$test_output" \
    --progress-every 1000000 \
    >"$LOG_DIR/score_test_${model}.log" 2>&1
  log "score_done model=$model split=test output=$test_output"
}

log "run_start run_id=$RUN_ID log_dir=$LOG_DIR"
log "config device=$DEVICE shards=$SHARDS torch_threads=$TORCH_THREADS inter_op_threads=$INTER_OP_THREADS python=$PYTHON_BIN"
log "protocols extraction=$EXTRACTION_PROTOCOL calibration=$CALIBRATION_PROTOCOL test=$TEST_PROTOCOL"

for model in "${MODELS[@]}"; do
  run_sharded_extraction "$model"
  run_scores "$model"
done

log "run_done"
