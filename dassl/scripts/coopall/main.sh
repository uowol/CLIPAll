#!/bin/bash

#cd ../..

# custom config
DATA="/data4/kchanwo/clipall/datasets/"
TRAINER=CoOpALL

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
SEED=$7

DIR=/data4/kchanwo/clipall/clipall/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir /data4/kchanwo/clipall/clipall/output/${DATASET}/CoOp/vit_b16_16shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed1 \
    --load-epoch 200 \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
