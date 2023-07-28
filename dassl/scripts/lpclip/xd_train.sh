#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=LinearProbingCLIP

DATASET=$1
SEED=$2

CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16
SHOTS=$3

DIR=/data4/kchanwo/clipall/maple/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS}