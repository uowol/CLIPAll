# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
SEED=$1
CUDA_VISIBLE_DEVICES=$2

bash scripts/mapleall/xd_test_maple.sh imagenetv2 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh imagenet_sketch ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh imagenet_a ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh imagenet_r ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh oxford_flowers ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh caltech101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh dtd ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh fgvc_aircraft ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh stanford_cars ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh oxford_pets ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh ucf101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh eurosat ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh food101 ${SEED} ${CUDA_VISIBLE_DEVICES}
bash scripts/mapleall/xd_test_maple.sh sun397 ${SEED} ${CUDA_VISIBLE_DEVICES}