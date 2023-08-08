#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=CLIPALL

DATASET=$1
SEED=$2
SHOTS=$3
CUDA_VISIBLE_DEVICES=$4
BATCH=$5
EP=$6
# SAVE=$7
CFG=mom_lr2e-3_B${BATCH}_ep${EP}


DIR=/data4/kchanwo/clipall/clipall/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
# DIR=/data4/kchanwo/clipall/clipall/output/${DATASET}/${TRAINER}/${SAVE}/seed${SEED}
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