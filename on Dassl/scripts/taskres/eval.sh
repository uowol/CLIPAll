#!/bin/bash

# custom config
DATA=/data4/kchanwo/clipall/datasets/
TRAINER=CLIPall
SHOTS=16
NCTX=16
CSC=False
CTP=end
SCALE=0.5

DATASET=$1
CFG=$2
CUDA_VISIBLE_DEVICES=$3
SAVE=$4
OUTDIR="/data4/kchanwo/clipall/taskres/output/evaluation/${DATASET}/${CFG}_${SHOTS}shots"
MODELDIR="/data4/kchanwo/clipall/taskres/output/FINAL/debug/imagenet/${SAVE}"

for SEED in 1
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${OUTDIR}/seed${SEED} \
    --model-dir ${MODELDIR}/ \
    --load-epoch 60 \
    --eval-only \
    TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE}
done