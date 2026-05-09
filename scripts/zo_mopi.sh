#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


MODEL="${MODEL:-facebook/opt-1.3b}"
TASK="${TASK:-SQuAD}"
MODE="${MODE:-ft}"
GPU="${GPU:-3}"
BS="${BS:-16}"
EPS="${EPS:-1e-3}"
LR="${LR:-5e-3}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
STEPS="${STEPS:-8000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-8000}"
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

# ZO-Muon specific hyperparameters
RANK="${RANK:-64}"
STEP_INTERVAL="${STEP_INTERVAL:-1}"
K_START="${K_START:-32}"
RESET_V_ON_P_REFRESH="${RESET_V_ON_P_REFRESH:-True}"
PHASE2_STEPS="${PHASE2_STEPS:-20000}"
BETA="${BETA:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MULTIPLE_SAMPLE="${MULTIPLE_SAMPLE:-True}"
PERTURBATION_MODE="${PERTURBATION_MODE:-two_side}"
PHASE2_TYPE="${PHASE2_TYPE:-muon}"
ONE_D_LR="${ONE_D_LR:-1e-7}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="zomuon"

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
# TASK_ARGS=""
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

TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-vi${STEP_INTERVAL}-r${RANK}-k${K_START}-p2${PHASE2_STEPS}-ns${NUM_SAMPLES}-b${BETA}-p${PERTURBATION_MODE}-t${PHASE2_TYPE}-${RUN_SUFFIX}"
OUTPUT_DIR="${LLM_DIR}/result/${TASK}-${MODEL_NAME}-${TAG}"
WANDB_NAME="${WANDB_NAME:-${TASK}-${MODEL_NAME}-${TAG}}"

echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "GPU(s): ${GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Trainer: ${TRAINER}"
echo "LR scheduler: ${LR_SCHEDULER_TYPE}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Rank: ${RANK}"
echo "Step interval: ${STEP_INTERVAL}"
echo "K start: ${K_START}"
echo "Reset V on P refresh: ${RESET_V_ON_P_REFRESH}"
echo "Phase2 steps: ${PHASE2_STEPS}"
echo "Phase2 type: ${PHASE2_TYPE}"
echo "Perturbation mode: ${PERTURBATION_MODE}"
echo "Multiple sample: ${MULTIPLE_SAMPLE}"
echo "Num samples: ${NUM_SAMPLES}"
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
    --zo_optimizer "${PHASE2_TYPE}" \
    --multiple_sample "${MULTIPLE_SAMPLE}" \
    --max_time "${MAX_TIME}" \
    --adam_eps 1e-6 \
    --num_samples "${NUM_SAMPLES}" \
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
    --k_start "${K_START}" \
    --reset_v_on_p_refresh "${RESET_V_ON_P_REFRESH}" \
    --phase2_steps "${PHASE2_STEPS}" \
    --zo_muon_beta "${BETA}" \
    --zo_perturbation_mode "${PERTURBATION_MODE}" \
    --one_d_lr "${ONE_D_LR}" \
    --report_to "${REPORT_TO}" \
    ${OVERWRITE_FLAG} \
    ${TASK_ARGS} \
    ${EXTRA_ARGS} \
    "$@"
