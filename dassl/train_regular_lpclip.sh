# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=lpclip

# ###################### 1-Shot ######################
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 1 ${CUDA_VISIBLE_DEVICES}

# ###################### 2-Shot ######################
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 2 ${CUDA_VISIBLE_DEVICES}

# ###################### 4-Shot ######################
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 4 ${CUDA_VISIBLE_DEVICES}

# ###################### 8-Shot ###################### 여기서부터는 보통 200 epoch를 사용한다.
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
# bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 8 ${CUDA_VISIBLE_DEVICES}

###################### 16-Shot ######################
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}

bash ~/CLIPAll/dassl/scripts/${TRAINER}/xd_train.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
