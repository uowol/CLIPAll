DIR="/data4/kchanwo/clipall/maple/output/"
DATASET=$1
TRAINER=$2
CFG=$3
NUM_SHOTS=$4
TYPE=$5 # fewshot, base2novel, crossdataset

python print_logs.py \
--dir ${DIR} \
--dataset ${DATASET} \
--trainer ${TRAINER} \
--cfg ${CFG} \
--num_shots ${NUM_SHOTS} \
--type ${TYPE}
