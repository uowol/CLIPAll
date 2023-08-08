SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=coopall

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101 imagenet
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} bash ~/CLIPAll/dassl/scripts/${TRAINER}/base2new_train.sh ${DATASET} ${CUDA_VISIBLE_DEVICES} end 16 False ${SEED}
done
