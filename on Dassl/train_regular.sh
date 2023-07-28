# conda activate maple
# cd vscode/TaskRes/

# DATASET=$1
METHOD=$1
CUDA_VISIBLE_DEVICES=$2
SAVE=$3

###################### 1-Shot ######################
# bash scripts/taskres/main.sh stanford_cars ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh oxford_flowers ${METHOD}_lr2e-3_B256_ep100 none 1 1.0 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh caltech101 ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh dtd ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh fgvc_aircraft ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh oxford_pets ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh food101 ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh ucf101 ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh eurosat ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots
# bash scripts/taskres/main.sh sun397 ${METHOD}_lr2e-3_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots

###################### 2-Shot ######################
# bash scripts/taskres/main.sh stanford_cars ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh oxford_flowers ${METHOD}_lr2e-3_B256_ep100 none 2 1.0 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh caltech101 ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh dtd ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh fgvc_aircraft ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh oxford_pets ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh food101 ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh ucf101 ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh eurosat ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots
# bash scripts/taskres/main.sh sun397 ${METHOD}_lr2e-3_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots

###################### 4-Shot ######################
# bash scripts/taskres/main.sh stanford_cars ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh oxford_flowers ${METHOD}_lr2e-3_B256_ep100 none 4 1.0 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh caltech101 ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh dtd ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh fgvc_aircraft ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh oxford_pets ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh food101 ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh ucf101 ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh eurosat ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots
# bash scripts/taskres/main.sh sun397 ${METHOD}_lr2e-3_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots

###################### 8-Shot ######################
# bash scripts/taskres/main.sh stanford_cars ${METHOD}_lr2e-3_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots
# bash scripts/taskres/main.sh oxford_flowers ${METHOD}_lr2e-3_B256_ep200 none 8 1.0 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots
bash scripts/taskres/main.sh caltech101 ${METHOD}_lr2e-3_B32_ep100 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_8shots
bash scripts/taskres/main.sh dtd ${METHOD}_lr2e-3_B32_ep100 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_8shots
bash scripts/taskres/main.sh fgvc_aircraft ${METHOD}_lr2e-3_B32_ep100 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_8shots
# bash scripts/taskres/main.sh oxford_pets ${METHOD}_lr2e-3_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots
# bash scripts/taskres/main.sh food101 ${METHOD}_lr2e-3_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots
# bash scripts/taskres/main.sh ucf101 ${METHOD}_lr2e-3_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots
bash scripts/taskres/main.sh eurosat ${METHOD}_lr2e-3_B32_ep100 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_8shots
# bash scripts/taskres/main.sh sun397 ${METHOD}_lr2e-3_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots

###################### 16-Shot ######################
# bash scripts/taskres/main.sh stanford_cars ${METHOD}_lr2e-3_B256_ep200 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_16shots
# bash scripts/taskres/main.sh oxford_flowers ${METHOD}_lr2e-3_B256_ep200 none 16 1.0 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_16shots
bash scripts/taskres/main.sh caltech101 ${METHOD}_lr2e-3_B32_ep100 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_16shots
bash scripts/taskres/main.sh dtd ${METHOD}_lr2e-3_B32_ep100 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_16shots
bash scripts/taskres/main.sh fgvc_aircraft ${METHOD}_lr2e-3_B32_ep100 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_16shots
# bash scripts/taskres/main.sh oxford_pets ${METHOD}_lr2e-3_B256_ep200 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_16shots
# bash scripts/taskres/main.sh food101 ${METHOD}_lr2e-3_B256_ep200 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_16shots
# bash scripts/taskres/main.sh ucf101 ${METHOD}_lr2e-3_B256_ep200 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_16shots
bash scripts/taskres/main.sh eurosat ${METHOD}_lr2e-3_B32_ep100 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B32_16shots
# bash scripts/taskres/main.sh sun397 ${METHOD}_lr2e-3_B256_ep200 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_16shots