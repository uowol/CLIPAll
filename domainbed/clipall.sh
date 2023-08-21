DATASET=$1
NAME=$2
SEED=$3
CUDA_VISIBLE_DEVICES=$4
B1=$5
B2=$6
B3=$7
B4=$8

## NAME LIST
# [None] 	best hyper parameter
# [1]	    b4_lr1e-3
# [2]	    b64_lr1e-3 
# [3]	    b64_lr3e-3 <- except
# [4]	    b32_lr1e-3 
# [5]	    b32_lr2e-3 <- except 

## EXAMPLE
# bash clipall.sh VLCS b4_lr1e-3 1 4 2
# bash clipall.sh VLCS b32_lr1e-3 1 32 2
# bash clipall.sh OfficeHome b64_lr1e-3 1 64 2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T0_${NAME} \
    --algorithm CLIPALL \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${B1} }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs 0 \
&&
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T1_${NAME} \
    --algorithm CLIPALL \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${B2} }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs 1 \
&&
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T2_${NAME} \
    --algorithm CLIPALL \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${B3} }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs 2 \
&&
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m domainbed.scripts.train \
    --data_dir /data4/kchanwo/clipall/datasets/ \
    --output_dir /data4/kchanwo/clipall/train_results/ViTB16/CLIPALL_${DATASET}_T3_${NAME} \
    --algorithm CLIPALL \
    --dataset ${DATASET} \
    --hparams "{\"clip_backbone\": \"ViT-B/16\", \"batch_size\": ${B4} }" \
    --trial_seed ${SEED} \
    --seed ${SEED} \
    --test_envs 3
