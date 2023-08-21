CUDA_VISIBLE_DEVICES=$1
ALGORITHM=$2
DATASET=$3
SEED=$4
BATCH_SIZE=$5
LR=$6
T=$7
# LR2=$5
# LR3=$6
# LR4=$7

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/${ALGORITHM}/${DATASET}_T${T}_b${BATCH_SIZE}_lr${LR} \
    --algorithm ${ALGORITHM} \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${BATCH_SIZE}, \"lr\": ${LR} }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs ${T} \
    --dplclip_path /data4/kchanwo/clipall/train_results/DPLCLIP/${DATASET}_T${T}_b${BATCH_SIZE}_lr${LR}/IID_best.pkl