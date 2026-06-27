#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/voxceleb1b_data_processing/$RUN_ID}"
CONDA_ENV="${CONDA_ENV:-amecxsv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCORING_PYTHON_BIN="${SCORING_PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cpu}"
SHARDS="${SHARDS:-6}"
TORCH_THREADS="${TORCH_THREADS:-3}"
INTER_OP_THREADS="${INTER_OP_THREADS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

SOURCE_PROTOCOL="${SOURCE_PROTOCOL:-protocols/voxceleb1b.csv}"
CALIBRATION_PROTOCOL="${CALIBRATION_PROTOCOL:-protocols/voxceleb1b_calibration.csv}"
TEST_PROTOCOL="${TEST_PROTOCOL:-protocols/voxceleb1b_test.csv}"
UNUSED_PROTOCOL="${UNUSED_PROTOCOL:-protocols/voxceleb1b_unused_cross_split.csv}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-protocols/voxceleb1b_speaker_split_manifest.csv}"
SPLIT_SUMMARY="${SPLIT_SUMMARY:-protocols/voxceleb1b_split_summary.json}"
SPLIT_SEED="${SPLIT_SEED:-voxceleb1b_speaker_split_v1}"
CALIBRATION_RATIO="${CALIBRATION_RATIO:-0.60}"

AUDIO_ROOT="${AUDIO_ROOT:-data/voxceleb1}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-data/embeddings}"
SCORES_DIR="${SCORES_DIR:-data/scores}"
SCORE_PREFIX="${SCORE_PREFIX:-voxceleb1b}"
WIDE_OUTPUT="${WIDE_OUTPUT:-data/voxceleb1b_trials.parquet}"
C9_OUTPUT_DIR="${C9_OUTPUT_DIR:-data/c9}"
C9_PREFIX="${C9_PREFIX:-voxceleb1b_c9}"

RUN_SPLIT="${RUN_SPLIT:-1}"
RUN_EXTRACTION="${RUN_EXTRACTION:-1}"
RUN_SCORES="${RUN_SCORES:-1}"
RUN_WIDE="${RUN_WIDE:-1}"
RUN_C9="${RUN_C9:-1}"

DEFAULT_MODELS=(
  speechbrain_ecapa_tdnn_voxceleb
  wespeaker_resnet34_cnceleb
  funasr_campplus_cn_3k
  funasr_eres2netv2_cn_200k
  hf_wavlm_base_sv_voxceleb1
  hf_wavlm_base_plus_sv_voxceleb1
)

if [[ -n "${MODEL_LIST:-}" ]]; then
  MODEL_LIST_NORMALIZED="${MODEL_LIST//,/ }"
  read -r -a MODELS <<< "$MODEL_LIST_NORMALIZED"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$LOG_DIR" "$SCORES_DIR" "$(dirname "$WIDE_OUTPUT")" "$C9_OUTPUT_DIR"

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_DIR/run.log"
}

run_split() {
  log "split_start source=$SOURCE_PROTOCOL calibration=$CALIBRATION_PROTOCOL test=$TEST_PROTOCOL"
  "$PYTHON_BIN" data_prep/split_trial_protocol.py \
    --protocol "$SOURCE_PROTOCOL" \
    --calibration-ratio "$CALIBRATION_RATIO" \
    --seed "$SPLIT_SEED" \
    --manifest-output "$SPLIT_MANIFEST" \
    --calibration-output "$CALIBRATION_PROTOCOL" \
    --test-output "$TEST_PROTOCOL" \
    --unused-output "$UNUSED_PROTOCOL" \
    --summary-output "$SPLIT_SUMMARY" \
    >"$LOG_DIR/split.log" 2>&1
  log "split_done summary=$SPLIT_SUMMARY"
}

run_sharded_extraction() {
  local model="$1"
  local pids=()

  log "extract_start model=$model protocol=$SOURCE_PROTOCOL device=$DEVICE shards=$SHARDS"
  for shard in $(seq 0 $((SHARDS - 1))); do
    local shard_log="$LOG_DIR/extract_${model}_shard${shard}.log"
    (
      "$PYTHON_BIN" data_prep/extract_protocol_embeddings.py \
        --protocol "$SOURCE_PROTOCOL" \
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

  local wait_status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      wait_status=1
    fi
  done
  if [[ "$wait_status" -ne 0 ]]; then
    log "extract_failed model=$model"
    return "$wait_status"
  fi
  log "extract_done model=$model"
}

run_scores_for_model() {
  local model="$1"
  local calibration_output="$SCORES_DIR/${SCORE_PREFIX}_calibration_${model}.parquet"
  local test_output="$SCORES_DIR/${SCORE_PREFIX}_test_${model}.parquet"

  log "score_start model=$model split=calibration output=$calibration_output"
  "$SCORING_PYTHON_BIN" data_prep/score_trial_protocol.py \
    --protocol "$CALIBRATION_PROTOCOL" \
    --embeddings-root "$EMBEDDINGS_ROOT" \
    --model "$model" \
    --output "$calibration_output" \
    --output-format parquet \
    --progress-every 1000000 \
    >"$LOG_DIR/score_calibration_${model}.log" 2>&1
  log "score_done model=$model split=calibration output=$calibration_output"

  log "score_start model=$model split=test output=$test_output"
  "$SCORING_PYTHON_BIN" data_prep/score_trial_protocol.py \
    --protocol "$TEST_PROTOCOL" \
    --embeddings-root "$EMBEDDINGS_ROOT" \
    --model "$model" \
    --output "$test_output" \
    --output-format parquet \
    --progress-every 1000000 \
    >"$LOG_DIR/score_test_${model}.log" 2>&1
  log "score_done model=$model split=test output=$test_output"
}

run_wide_table() {
  log "wide_start output=$WIDE_OUTPUT"
  "$SCORING_PYTHON_BIN" data_prep/build_c0_c5_trial_table.py \
    --scores-dir "$SCORES_DIR" \
    --score-glob "${SCORE_PREFIX}_*.parquet" \
    --output "$WIDE_OUTPUT" \
    --output-format parquet \
    --center-protocol "$CALIBRATION_PROTOCOL" \
    --dataset-name voxceleb1b \
    --overwrite \
    >"$LOG_DIR/build_wide_table.log" 2>&1
  log "wide_done output=$WIDE_OUTPUT"
}

run_c9_dataset() {
  log "c9_start output_dir=$C9_OUTPUT_DIR prefix=$C9_PREFIX"
  "$SCORING_PYTHON_BIN" data_prep/build_c9_multi_enroll_dataset.py \
    --scores-dir "$SCORES_DIR" \
    --score-prefix "$SCORE_PREFIX" \
    --score-format parquet \
    --dataset-name voxceleb1b \
    --dataset-prefix "$C9_PREFIX" \
    --output-dir "$C9_OUTPUT_DIR" \
    --seed voxceleb1b_c9_multi_enroll_v1 \
    --overwrite \
    >"$LOG_DIR/build_c9.log" 2>&1
  log "c9_done output_dir=$C9_OUTPUT_DIR prefix=$C9_PREFIX"
}

log "run_start run_id=$RUN_ID log_dir=$LOG_DIR"
log "config python=$PYTHON_BIN scoring_python=$SCORING_PYTHON_BIN device=$DEVICE shards=$SHARDS torch_threads=$TORCH_THREADS"
log "steps split=$RUN_SPLIT extraction=$RUN_EXTRACTION scores=$RUN_SCORES wide=$RUN_WIDE c9=$RUN_C9"
log "models=${MODELS[*]}"

if [[ "$RUN_SPLIT" == "1" ]]; then
  run_split
fi

for model in "${MODELS[@]}"; do
  if [[ "$RUN_EXTRACTION" == "1" ]]; then
    run_sharded_extraction "$model"
  fi
  if [[ "$RUN_SCORES" == "1" ]]; then
    run_scores_for_model "$model"
  fi
done

if [[ "$RUN_WIDE" == "1" ]]; then
  run_wide_table
fi

if [[ "$RUN_C9" == "1" ]]; then
  run_c9_dataset
fi

log "run_done"
