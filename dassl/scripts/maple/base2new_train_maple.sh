#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=MaPLe

DATASET=$1
SEED=$2
CUDA_VISIBLE_DEVICES=$3

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16


DIR=/data4/kchanwo/clipall/clipall/output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi