# conda activate maple
# cd CLIPall/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=$3

# bash scripts/${TRAINER}/base2new_test.sh stanford_cars ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh oxford_flowers ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh caltech101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh dtd ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh fgvc_aircraft ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh oxford_pets ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh food101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh ucf101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/${TRAINER}/base2new_test.sh eurosat ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh sun397 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
# bash scripts/${TRAINER}/base2new_test.sh imagenet ${SEED} ${CUDA_VISIBLE_DEVICES} 256 40
