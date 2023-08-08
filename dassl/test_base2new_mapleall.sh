# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=mapleall

bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh stanford_cars ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh oxford_flowers ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh caltech101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh dtd ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh fgvc_aircraft ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh oxford_pets ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh food101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh ucf101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh eurosat ${SEED} ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple.sh sun397 ${SEED} ${CUDA_VISIBLE_DEVICES}

bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_test_maple_dg.sh imagenet ${SEED} ${CUDA_VISIBLE_DEVICES}

# bash ~/CLIPAll/dassl/scripts/mapleall/base2new_test_maple.sh eurosat 1 4
