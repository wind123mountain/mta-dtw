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
CKPT_NAME="gemma-3-270m"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"

TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen3-4B-Instruct-2507"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_TYPE}/${TEACHER_MODEL_NAME}"

DATA_DIR="${BASE_PATH}/data/openmath"
TASK="dual_space_kd_with_cma"

BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACC=${GRAD_ACC:-32}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
LR=${LR:-0.001}
EPOCH=${EPOCH:-3}

DTW_RATE=${DTW_RATE:-0.2}
CE_RATE=${CE_RATE:-0.5}
KD_RATE=${KD_RATE:-0.5}
KD_TEMP=${KD_TEMP:-2.0}
DTW_GAMMA=${DTW_GAMMA:-2.0}

MAX_LENGTH=${MAX_LENGTH:-1024}
PRECISION=${PRECISION:-bf16}
CRITERION="dual_space_kd_with_cma"
KD_OBJ=${KD_OBJ:-skewed_reverse_kl}

PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=${PROJECTOR_LR:-0.001}

SETTING=criterion=${CRITERION}__epoch=${EPOCH}__teacher=${TEACHER_MODEL_TYPE}__kd^rate=${KD_RATE}__dtw^rate=${DTW_RATE}__ce^rate=${CE_RATE}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1

SEED=${SEED:-10}

mkdir -p "${SAVE_PATH}"

OPTS=""
OPTS+=" --dtw-gamma ${DTW_GAMMA}"
# OPTS+=" --wandb --wandb-project OpenMath-Distill --wandb-run-name gemma3_270m_dskd_full"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
OPTS+=" --task ${TASK}"
OPTS+=" --criterion ${CRITERION}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 2"
OPTS+=" --dev-num 1000"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --lr ${LR}"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --dtw-rate ${DTW_RATE}"
OPTS+=" --ce-rate ${CE_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 512"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --seed ${SEED}"

OPTS+=" --deepspeed"
if [[ ${PRECISION} == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ ${PRECISION} == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ ${PRECISION} == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"

export NCCL_DEBUG=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

echo "Launching: ${CMD}"
${CMD} >> "${SAVE_PATH}/train.log" 2>&1 &


