#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4

BATCH_SIZE_PER_GPU=1


# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" # 'google/gemma-2-2b-it' "meta-llama/Meta-Llama-3.1-8B-Instruct" 'mistralai/Mistral-7B-Instruct-v0.3'
# MODEL_NAME="microsoft/Phi-3.5-MoE-instruct"
MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
# MODEL_NAME='google/gemma-2-2b-it'
# DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list

DATASET_LIST=('dolly') 

OUTPUT_DIR='../labeling-info-local-with-perplexity/'
LOG_FILE='../labeling_local.log'
mkdir -p $OUTPUT_DIR

## clear the log file
: > "$LOG_FILE"


for DATASET_NAME in "${DATASET_LIST[@]}"; do
    echo "Labeling ${DATASET_NAME} dataset using model ${MODEL_NAME} on $NUM_GPUS GPUs" | tee -a "$LOG_FILE"
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --dynamo_backend no \
        labeling_json.py \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE"
    
    sleep 10 # for release the port 29500
done


############################# check the LOG_FILE ############################

# nohup bash labeling_datasets_local.sh > /dev/null 2>&1 & 

