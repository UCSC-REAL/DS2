#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8

BATCH_SIZE_PER_GPU=1


MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct" # 'google/gemma-2b-it' "meta-llama/Meta-Llama-3.1-8B-Instruct" 'mistralai/Mistral-7B-Instruct-v0.3'

# DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list

DATASET_LIST=('flan_v2') 



OUTPUT_DIR="/mnt/azureml/crunch/outputs/"
mkdir -p $OUTPUT_DIR

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


# root_dir="score_curationopen-instruct" 
# # data_dir="${root_dir}/logs/${MODEL_NAME}/${DATASET_NAME}"
# data_dir="${root_dir}/zzzz-test/"


AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/output/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
azcopy copy "$OUTPUT_DIR/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive


# azcopy copy  "$AZURE_STORAGE_CONTAINER_URL"  "$data_dir" --recursive # --from-to BlobLocal


# azcopy copy  "https://afminternshipuksouth.blob.core.windows.net/jinlong/output/?sp=racwdlmeop&st=2024-08-16T20:24:00Z&se=2025-03-07T05:24:00Z&sv=2022-11-02&sr=c&sig=%2F3S088nsqvxn%2BtRGen7utkkxJHxgQ5NwlSQlWufI390%3D" ./labeling-info/ --recursive


