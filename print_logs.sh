DIR="/data4/kchanwo/clipall/clipall/output/"
TRAINER=$1
CFG=$2
# NUM_SHOTS=$4
TYPE=$3 # fewshot, base2novel, crossdataset
# DATASET=$4

for DATASET in imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    for NUM_SHOTS in 16 #1 2 4 8 16
    do
        python print_logs.py \
        --dir ${DIR} \
        --dataset ${DATASET} \
        --trainer ${TRAINER} \
        --cfg ${CFG} \
        --num_shots ${NUM_SHOTS} \
        --type ${TYPE}
    done
    read
done