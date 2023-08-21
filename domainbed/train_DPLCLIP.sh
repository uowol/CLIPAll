CUDA_VISIBLE_DEVICES=$1
DATASET=$2
LR='1e-3'
for SEED in 1 2 3
do
    for TEST_ENV in 0 1 2 3
    do
        bash train.sh ${CUDA_VISIBLE_DEVICES} DPLCLIP ${DATASET} ${SEED} 32 ${LR} ${TEST_ENV}
    done
done