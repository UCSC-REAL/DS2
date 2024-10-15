
# datasets=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca' 'wizardlm' 'open_orca' 'science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded') #sharegpt

datasets=('gpt4_alpaca')
# datasets=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca' 'wizardlm' 'open_orca' )
## ignore the following dataset now
# datasets=('science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded') #sharegpt

for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python3 filter_dataset.py  --dataset_name $dataset
done



# python3 ./tools/diagnose_rlhf.py  --config ./config/hh_rlhf_harmless-base.py