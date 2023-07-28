# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
CFG=$1
CUDA_VISIBLE_DEVICES=$2
SAVE=$3

bash scripts/taskres/eval.sh imagenetv2 generalization_rn50 ${CFG} ${CUDA_VISIBLE_DEVICES} ${SAVE}
bash scripts/taskres/eval.sh imagenet_sketch generalization_rn50 ${CFG} ${CUDA_VISIBLE_DEVICES} ${SAVE}
bash scripts/taskres/eval.sh imagenet_a generalization_rn50 ${CFG} ${CUDA_VISIBLE_DEVICES} ${SAVE}
bash scripts/taskres/eval.sh imagenet_r generalization_50 ${CFG} ${CUDA_VISIBLE_DEVICES} ${SAVE}