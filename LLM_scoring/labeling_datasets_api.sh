

#### Full sub-dataset for reference
# DATASET_LIST=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca' 'sharegpt' 'wizardlm' 'open_orca' 'science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded')

# DATASET_LIST=('sharegpt')
# DATASET_LIST=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca')
# DATASET_LIST=('wizardlm' 'open_orca' 'science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded')


### Full api call models
# MODEL_NAME=('meta/llama-3.1-8b-instruct' 'google/gemma-2-9b-it' 'mistralai/mistral-7b-instruct-v0.3' "mistralai/mixtral-8x22b-instruct-v0.1" "nv-mistralai/mistral-nemo-12b-instruct" "meta/llama-3.1-405b-instruct") #

MODEL_NAME=('meta/llama-3.1-8b-instruct')
DATASET_LIST=('flan_v2')

for DATASET_NAME in "${DATASET_LIST[@]}"; do
    LOG_FILE="./data_refine/logs/${DATASET_NAME}.log"
    echo "Labeling ${DATASET_NAME} dataset using API CALL model ${MODEL_NAME}" | tee -a "$LOG_FILE"

    python3 data_refine/labeling_api.py \
        --model_name $MODEL_NAME \
        --dataset_name $DATASET_NAME 2>&1 | tee -a "$LOG_FILE" &
done

# nohup bash ./data_refine/labeling_multiple_datasets_api.sh > labeling_api_call_llama.log &
# nohup bash ./data_refine/labeling_multiple_datasets_api.sh > labeling_api_call_mistral.log &
# nohup bash ./data_refine/labeling_multiple_datasets_api.sh > labeling_api_call_gemma.log &