#!/bin/bash

# custom config
DATA=/data4/kchanwo/clipall/datasets/
TRAINER=ZeroshotCLIP
# TRAINER=ZeroshotCLIP2

DATASET=$1
CUDA_VISIBLE_DEVICES=$2

CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir /data4/kchanwo/clipall/clipall/output/${TRAINER}/${CFG}/${DATASET} \
--eval-only