#!/usr/bin/env bash
set -euo pipefail

GPUS=(1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

BASE_PATH=/home/hungpv/projects/lab/DTW-v2

CKPT_TYPE="gemma"
CKPT_NAME="gemma-3-1b-pt"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"

TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen3-4B-Instruct-2507"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_TYPE}/${TEACHER_MODEL_NAME}"

DATA_DIR="${BASE_PATH}/data/openmath"
TASK="dual_space_kd"

BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACC=${GRAD_ACC:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
LR=${LR:-0.001}
TOTAL_ITERS=${TOTAL_ITERS:-10000}
TRAIN_NUM=${TRAIN_NUM:-50000}

DTW_RATE=${DTW_RATE:-0.2}
CE_RATE=${CE_RATE:-0.5}
KD_RATE=${KD_RATE:-0.5}
KD_TEMP=${KD_TEMP:-2.0}
DTW_GAMMA=${DTW_GAMMA:-2.0}

LORA_RANK=${LORA_RANK:-256}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}

MAX_LENGTH=${MAX_LENGTH:-512}
PRECISION=${PRECISION:-bf16}
CRITERION="dual_space_kd"
KD_OBJ=${KD_OBJ:-adaptive_kl}

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=${PROJECTOR_LR:-0.001}

SETTING=criterion=${CRITERION}+dtw__steps=${TOTAL_ITERS}__teacher=${TEACHER_MODEL_TYPE}__kd^rate=${KD_RATE}__dtw^rate=${DTW_RATE}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1

SEED=10

mkdir -p "${SAVE_PATH}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --task ${TASK}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 2"
OPTS+=" --train-num ${TRAIN_NUM}"
OPTS+=" --do-train"
OPTS+=" --eval-gen"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --lr ${LR}"
OPTS+=" --total-iters ${TOTAL_ITERS}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --dtw-rate ${DTW_RATE}"
OPTS+=" --ce-rate ${CE_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --dtw-gamma ${DTW_GAMMA}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 512"
OPTS+=" --gradient-checkpointing"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${LORA_RANK}"
OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --seed ${SEED}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --wandb --wandb-project OpenMath-Distill --wandb-run-name gemma3_dskd_dtw"

if [[ ${PRECISION} == "bf16" ]]; then
    OPTS+=" --deepspeed"
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ ${PRECISION} == "fp16" ]]; then
    OPTS+=" --deepspeed"
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
fi

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"

export NCCL_DEBUG=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

echo "Launching: ${CMD}"
${CMD} |& tee "${SAVE_PATH}/train.log"

