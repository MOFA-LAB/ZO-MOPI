#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODEL="${MODEL:-facebook/opt-1.3b}"
TASK="${TASK:-SST2}"
MODE="${MODE:-ft}"
GPU="${GPU:-3}"
BS="${BS:-16}"
EPS="${EPS:-1e-3}"
LR="${LR:-1e-7}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
STEPS="${STEPS:-20000}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-20000}"
TRAIN="${TRAIN:-1000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-float16}"
MAX_TIME="${MAX_TIME:-0}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-gemma}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

RANK="${RANK:-2}"
STEP_INTERVAL="${STEP_INTERVAL:-100}"
OPT="${OPT:-sgd}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="lozo"

EXTRA_ARGS=""
if [ "$MODE" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE_OUTPUT_DIR}" = "True" ] || [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

LOAD_FLAG=""
if [ "$LOAD_MODE" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" = "float16" ]; then
    LOAD_FLAG="--load_float16"
fi

TASK_ARGS="--train_as_classification"
case "$TASK" in
    CB)
        DEV=100
        ;;
    Copa)
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD|DROP|SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

mkdir -p "${LLM_DIR}/result"

TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-vi${STEP_INTERVAL}-r${RANK}-opt${OPT}-${RUN_SUFFIX}"
OUTPUT_DIR="${LLM_DIR}/result/${TASK}-${MODEL_NAME}-${TAG}"
WANDB_NAME="${WANDB_NAME:-${TASK}-${MODEL_NAME}-${TAG}}"

echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "GPU(s): ${GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Trainer: ${TRAINER}"
echo "Optimizer: ${OPT}"
echo "LR scheduler: ${LR_SCHEDULER_TYPE}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Rank: ${RANK}"
echo "Step interval: ${STEP_INTERVAL}"
echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"

cd "${LLM_DIR}"

WANDB_PROJECT="${WANDB_PROJECT_NAME}" WANDB_NAME="${WANDB_NAME}" CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model_name "${MODEL}" \
    --task_name "${TASK}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --train_set_seed "${SEED}" \
    --num_train "${TRAIN}" \
    --num_eval "${EVAL}" \
    --logging_steps 1 \
    --max_steps "${STEPS}" \
    --trainer "${TRAINER}" \
    ${LOAD_FLAG} \
    --zo_optimizer "${OPT}" \
    --adam_eps 1e-6 \
    --max_time "${MAX_TIME}" \
    --learning_rate "${LR}" \
    --zo_eps "${EPS}" \
    --per_device_train_batch_size "${BS}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --eval_strategy steps \
    --save_strategy steps \
    --save_total_limit 1 \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --step_interval "${STEP_INTERVAL}" \
    --rank_r "${RANK}" \
    --report_to "${REPORT_TO}" \
    ${OVERWRITE_FLAG} \
    ${TASK_ARGS} \
    ${EXTRA_ARGS} \
    "$@"
