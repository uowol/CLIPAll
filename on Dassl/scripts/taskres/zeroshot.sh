#!/bin/bash

# custom config
DATA=/data4/kchanwo/clipall/datasets/
# TRAINER=ZeroshotCLIP
TRAINER=ZeroshotCLIP2
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir /data4/kchanwo/clipall/maple/output/${TRAINER}/${CFG}/${DATASET} \
--eval-only