# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
SEED=$1
CUDA_VISIBLE_DEVICES=$2

bash scripts/clipall/test.sh imagenetv2 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash scripts/clipall/test.sh imagenet_sketch ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash scripts/clipall/test.sh imagenet_a ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash scripts/clipall/test.sh imagenet_r ${SEED} 16 ${CUDA_VISIBLE_DEVICES}