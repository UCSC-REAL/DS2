#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" 
DATASET_LIST=('tulu_300k') 



OUTPUT_DIR="../scoring_output/"
mkdir -p $OUTPUT_DIR

for DATASET_NAME in "${DATASET_LIST[@]}"; do

    : > "$LOG_FILE"
    LOG_FILE=${OUTPUT_DIR}/${DATASET_NAME}.log

    echo "Scoring ${DATASET_NAME} dataset using model ${MODEL_NAME} on $NUM_GPUS GPUs" | tee -a "$LOG_FILE"
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --dynamo_backend no \
        --main_process_port 29501 \
        scoring.py \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE"
    
    sleep 10 # for release the port 29500
done




