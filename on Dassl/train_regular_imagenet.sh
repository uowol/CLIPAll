# conda activate maple
# cd vscode/TaskRes/

# DATASET=$1
METHOD=$1
CUDA_VISIBLE_DEVICES=$2
SAVE=$3

# ###################### 1-Shot ######################
# bash scripts/taskres/main.sh imagenet ${METHOD}_lr2e-4_B256_ep100 none 1 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_1shots

# ###################### 2-Shot ######################
# bash scripts/taskres/main.sh imagenet ${METHOD}_lr2e-4_B256_ep100 none 2 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_2shots

# ###################### 4-Shot ######################
# bash scripts/taskres/main.sh imagenet ${METHOD}_lr2e-4_B256_ep100 none 4 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep100B256_4shots

# ###################### 8-Shot ######################
# bash scripts/taskres/main.sh imagenet ${METHOD}_lr2e-4_B256_ep200 none 8 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep200B256_8shots

###################### 16-Shot ######################
bash scripts/taskres/main.sh imagenet ${METHOD}_lr2e-3_B256_ep40 none 16 0.5 ${CUDA_VISIBLE_DEVICES} ${SAVE}Ep40B256_16shots