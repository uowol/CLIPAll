SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=coop

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet 
do
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep50 end 16 1 False ${SEED}
    # # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep50 end 16 1 True ${SEED}
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep100 end 16 2 False ${SEED}
    # # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep100 end 16 2 True ${SEED}
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep100 end 16 4 False ${SEED}
    # # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16_ep100 end 16 4 True ${SEED}
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16 end 16 8 False ${SEED}
    # # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16 end 16 8 True ${SEED}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16 end 16 16 False ${SEED}
    # CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/main.sh ${DATASET} vit_b16 end 16 16 True ${SEED}
done

# CUDA_VISIBLE_DEVICES=0 bash ~/CLIPAll/dassl/scripts/coopall/main.sh eurosat vit_b16 end 16 16 False 1