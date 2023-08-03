# conda activate maple
# cd CLIPall/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2

bash scripts/clipall/base2new_train.sh stanford_cars ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh oxford_flowers ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh caltech101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh dtd ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh fgvc_aircraft ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh oxford_pets ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh food101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh ucf101 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh eurosat ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh sun397 ${SEED} ${CUDA_VISIBLE_DEVICES} 32 100
bash scripts/clipall/base2new_train.sh imagenet ${SEED} ${CUDA_VISIBLE_DEVICES} 256 40
