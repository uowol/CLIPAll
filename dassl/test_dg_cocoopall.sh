SEED=$1
CUDA_VISIBLE_DEVICES=$2
TRAINER=cocoopall

###################### 16-Shot ######################
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh imagenetv2 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh imagenet_sketch ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh imagenet_a ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh imagenet_r ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh stanford_cars ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh oxford_flowers ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh caltech101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh dtd ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh fgvc_aircraft ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh oxford_pets ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh food101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh ucf101 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh eurosat ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
bash ~/CLIPAll/dassl/scripts/${TRAINER}/test.sh sun397 ${SEED} 16 ${CUDA_VISIBLE_DEVICES}
