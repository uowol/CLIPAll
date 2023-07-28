# conda activate maple
# cd vscode/TaskRes/

SEED=$1
CUDA_VISIBLE_DEVICES=$2

# ###################### 1-Shot ######################
bash scripts/clipall/train.sh imagenet ${SEED} 1 ${CUDA_VISIBLE_DEVICES}
# ###################### 2-Shot ######################
bash scripts/clipall/train.sh imagenet ${SEED} 2 ${CUDA_VISIBLE_DEVICES}
# ###################### 4-Shot ######################
bash scripts/clipall/train.sh imagenet ${SEED} 4 ${CUDA_VISIBLE_DEVICES}
# ###################### 8-Shot ######################
bash scripts/clipall/train.sh imagenet ${SEED} 8 ${CUDA_VISIBLE_DEVICES}
###################### 16-Shot ######################
bash scripts/clipall/train.sh imagenet ${SEED} 16 ${CUDA_VISIBLE_DEVICES}