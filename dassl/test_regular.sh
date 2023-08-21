# conda activate maple
# cd CLIPAll/dassl/

SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=lfa

###################### 16-Shot ######################
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 32 100

bash ~/CLIPAll/dassl/scripts/${TRAINER}/test_cp.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
