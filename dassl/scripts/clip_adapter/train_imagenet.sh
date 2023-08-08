#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=CLIP_Adapter

DATASET=$1
SEED=$2
SHOTS=$3
CUDA_VISIBLE_DEVICES=$4

CFG=mom_lr35e-3_B4_ep40


DIR=/data4/kchanwo/clipall/clipall/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi