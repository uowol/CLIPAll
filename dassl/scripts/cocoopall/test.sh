#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=CoCoOpALL

DATASET=$1
SEED=$2
SHOTS=$3
CUDA_VISIBLE_DEVICES=$4

CFG=vit_b16_c4_ep10_batch1_ctxv1
LOADEP=10


DIR=/data4/kchanwo/clipall/clipall/output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
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
    --model-dir /data4/kchanwo/clipall/clipall/output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only
fi