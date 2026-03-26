#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/llm-align/liuchonghan/hf_cache}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-$(find "$HF_CACHE_ROOT/models--google--metricx-24-hybrid-xl-v2p6/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:-$(find "$HF_CACHE_ROOT/models--google--mt5-xl/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)}"

OUTPUT_DIR="${OUTPUT_DIR:-/llm-align/liuchonghan/metricx_scores_v3_r1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-1536}"
LOG_FILE="${LOG_FILE:-/llm-align/liuchonghan/metricx_scores_v3_r1_run.log}"

INPUT_DIR_1="${INPUT_DIR_1:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_1}"
INPUT_DIR_2="${INPUT_DIR_2:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_2_t0.9}"
INPUT_DIR_3="${INPUT_DIR_3:-/llm-align/duyimin/multi_lang/v3_r1/result_top_lang_train_data_sample_3_t0.9}"

if [[ -z "$MODEL_SNAPSHOT" || ! -d "$MODEL_SNAPSHOT" ]]; then
  echo "MetricX model snapshot not found under $HF_CACHE_ROOT" >&2
  exit 1
fi

if [[ -z "$TOKENIZER_SNAPSHOT" || ! -d "$TOKENIZER_SNAPSHOT" ]]; then
  echo "mT5 tokenizer snapshot not found under $HF_CACHE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

cd "$REPO_DIR"

echo "repo_dir=$REPO_DIR"
echo "model_snapshot=$MODEL_SNAPSHOT"
echo "tokenizer_snapshot=$TOKENIZER_SNAPSHOT"
echo "output_dir=$OUTPUT_DIR"
echo "batch_size=$BATCH_SIZE"
echo "max_input_length=$MAX_INPUT_LENGTH"
echo "log_file=$LOG_FILE"
echo "input_dirs:"
echo "  $INPUT_DIR_1"
echo "  $INPUT_DIR_2"
echo "  $INPUT_DIR_3"

nohup python3 -m metricx24.score_dialog_outputs \
  --tokenizer "$TOKENIZER_SNAPSHOT" \
  --model_name_or_path "$MODEL_SNAPSHOT" \
  --input_dirs "$INPUT_DIR_1" "$INPUT_DIR_2" "$INPUT_DIR_3" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_input_length "$MAX_INPUT_LENGTH" \
  --qe > "$LOG_FILE" 2>&1 &

PID=$!
echo "started_pid=$PID"
echo "tail_log=tail -f $LOG_FILE"
echo "summary_file=$OUTPUT_DIR/summary_by_dir.json"
