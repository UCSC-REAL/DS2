#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1

BATCH_SIZE_PER_GPU=1


MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" # 'google/gemma-2b-it' "meta-llama/Meta-Llama-3.1-8B-Instruct" 'mistralai/Mistral-7B-Instruct-v0.3'

# DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list
DATASET_LIST=('test') 



OUTPUT_DIR="scoring_output/"
mkdir -p $OUTPUT_DIR

for DATASET_NAME in "${DATASET_LIST[@]}"; do

    LOG_FILE=${OUTPUT_DIR}/${DATASET_NAME}.log
    : > "$LOG_FILE"

    echo "Scoring ${DATASET_NAME} dataset using model ${MODEL_NAME} on $NUM_GPUS GPUs" | tee -a "$LOG_FILE"
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --dynamo_backend no \
        scoring_test.py \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE"
    
    sleep 10 # for release the port 29500
done




