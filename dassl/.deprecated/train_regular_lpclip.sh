# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=lpclip

###################### 16-Shot ######################
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train_dg.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/lpclip/xd_train.sh stanford_cars 1 16 0
