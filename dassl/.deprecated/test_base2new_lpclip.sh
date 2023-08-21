# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=lpclip

bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh stanford_cars ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh oxford_flowers ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh caltech101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh dtd ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh fgvc_aircraft ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh oxford_pets ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh food101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh ucf101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh eurosat ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test.sh sun397 ${SEED} ${CUDA_VISIBLE_DEVICES}

bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_dg.sh imagenet ${SEED} ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/lpclip/base2new_train.sh stanford_cars 1 0
