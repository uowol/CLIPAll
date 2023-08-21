# SEED=$1
CUDA_VISIBLE_DEVICES=$1
TRAINER=cocoopall

###################### 16-Shot ######################
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh stanford_cars 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh oxford_flowers 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh caltech101 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh dtd 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh fgvc_aircraft 1 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh oxford_pets 1 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh food101 1 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh ucf101 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh eurosat 1 16 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh sun397 1 16 ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh imagenet 1 16 ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/cocoopall/train.sh stanford_cars 1 16 3
