
### Scoring models
MODEL_NAME=('meta/llama-3.1-8b-instruct')

##TULU subsets
DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') 

for DATASET_NAME in "${DATASET_LIST[@]}"; do
    LOG_FILE=" logs/${DATASET_NAME}.log"
    echo "Scoring ${DATASET_NAME} dataset using API Call model ${MODEL_NAME}" | tee -a "$LOG_FILE"

    python3 scoring_api.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE" &
done

