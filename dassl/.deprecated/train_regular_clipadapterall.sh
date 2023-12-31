# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=clip_adapterall

###################### 16-Shot ######################
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/train.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}

bash ~/CLIPAll/dassl/scripts/${TRAINER}/train_imagenet.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/clip_adapterall/train.sh eurosat 1 16 1
