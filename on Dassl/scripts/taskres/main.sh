#!/bin/bash

# custom config
DATA=/data4/kchanwo/clipall/datasets/
# DATA=/path/to/datasets
TRAINER=CLIPall # CLIPall TaskRes

DATASET=$1
CFG=$2      # config file
ENHANCE=$3  # path to enhanced base weights
SHOTS=$4    # number of shots (1, 2, 4, 8, 16)
SCALE=$5    # scaling factor
CUDA_VISIBLE_DEVICES=$6
SAVE=$7

for SEED in 2
do
    # DIR=/data4/kchanwo/clipall/taskres/output/FINAL/debug/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
    DIR=/data4/kchanwo/clipall/taskres/output/FINAL/debug/${DATASET}/${SAVE}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --enhanced-base ${ENHANCE} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.TaskRes.RESIDUAL_SCALE ${SCALE}
    fi
done