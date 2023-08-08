CUDA_VISIBLE_DEVICES=$1
TRAINER=coopall

for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/eval.sh ${DATASET} mom_lr2e-3_B256_ep40
done
