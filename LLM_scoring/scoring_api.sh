
### Scoring models
MODEL_NAME=('meta/llama-3.1-8b-instruct')

##TULU subsets
DATASET_LIST=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca' 'sharegpt' 'wizardlm' 'open_orca' 'science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded')

for DATASET_NAME in "${DATASET_LIST[@]}"; do
    LOG_FILE=" logs/${DATASET_NAME}.log"
    echo "Scoring ${DATASET_NAME} dataset using API CALL model ${MODEL_NAME}" | tee -a "$LOG_FILE"

    python3 scoring_gpt_api.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE" &
done

