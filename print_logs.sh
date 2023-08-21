DIR="/data4/kchanwo/clipall/clipall/output/"
TYPE=$1 # fewshot, base2novel, crossdataset
TRAINER=$2
BATCH=$3
EP=$4
CFG=mom_lr2e-3_B${BATCH}_ep${EP}


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