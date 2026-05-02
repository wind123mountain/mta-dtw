#!/bin/bash
GPUS=(0)
WORK_DIR="/home/hungpv/projects/lab/DTW-v2"
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")

#CKPT_PATH=${1}
CKPT_PATH=/home/hungpv/projects/lab/DSKDv2/outputs/gpt2/gpt2-base/dskd_v2_eta/reverse_kl-bf16__teacher_qwen1.5__kd^rate0.5__kd^temp2.0__epoch10__bsz4x4x2x1_32__lr0.0005__proj^lr0.0005/epoch10_step3570_loss5.0984_rougel25.4130
BATCH_SIZE=${2-32}

for seed in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} dolly ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} sinst ${BATCH_SIZE} $seed
done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
# done

