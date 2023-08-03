# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
SEED=$1
CUDA_VISIBLE_DEVICES=$2

bash scripts/clipall/test.sh imagenetv2 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh imagenet_sketch ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh imagenet_a ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh imagenet_r ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40
bash scripts/clipall/test.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES} 256 40