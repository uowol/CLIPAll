DATASET='PACS'
SEED=$1
CUDA_VISIBLE_DEVICES=$2
BATCH_SIZE=$3
LR=$4
T=$5
# LR2=$5
# LR3=$6
# LR4=$7

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T0_b${BATCH_SIZE}_lr${LR} \
    --algorithm CLIPALL \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${BATCH_SIZE}, \"lr\": ${LR}, \"momentum\": 0.9 }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs ${T} \
# &&
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
#     --data_dir /data4/kchanwo/clipall/datasets/ \
#     --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T1_b32_lr${NAME} \
#     --algorithm CLIPALL \
#     --dataset ${DATASET} \
#     --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": 32, \"lr\": ${LR2} }" \
#     --trial_seed ${SEED} \
#     --seed ${SEED} \
#     --test_envs 1 \
# &&
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
#     --data_dir /data4/kchanwo/clipall/datasets/ \
#     --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T2_b32_lr${NAME} \
#     --algorithm CLIPALL \
#     --dataset ${DATASET} \
#     --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": 32, \"lr\": ${LR3} }" \
#     --trial_seed ${SEED} \
#     --seed ${SEED} \
#     --test_envs 2 \
# &&
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
#     --data_dir /data4/kchanwo/clipall/datasets/ \
#     --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T3_b4_lr${NAME} \
#     --algorithm CLIPALL \
#     --dataset ${DATASET} \
#     --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": 4, \"lr\": ${LR4} }" \
#     --trial_seed ${SEED} \
#     --seed ${SEED} \
#     --test_envs 3
