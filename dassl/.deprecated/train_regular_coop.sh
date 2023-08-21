SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=coop

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet 
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16 end 16 16 False ${SEED}
done

# CUDA_VISIBLE_DEVICES=2 bash ~/CLIPAll/dassl/scripts/coop/main.sh stanford_cars mom_lr2e-3_B32_ep100 end 16 16 False 1
# CUDA_VISIBLE_DEVICES=2 bash ~/CLIPAll/dassl/scripts/coop/main.sh imagenet mom_lr2e-3_B256_ep40 end 16 16 False 1