
# datasets=('flan_v2' 'cot' 'oasst1' 'lima' 'gpt4_alpaca' 'code_alpaca' 'wizardlm' 'open_orca' 'science.evidence_inference' 'science.qasper_truncated_4000' 'science.scifact_json' 'science.scitldr_aic' 'science.scierc_ner' 'science.scierc_relation' 'hard_coded') #sharegpt


# CUDA_VISIBLE_DEVICES=3

# # datasets=('flan_v2') 
# datasets=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca' 'all_train') 
# labeling_model=('mistralai/Mistral-7B-Instruct-v0.3') #'meta-llama/Meta-Llama-3.1-8B-Instruct'


# for dataset in "${datasets[@]}"
# do
#     python3 ./tools/diagnose_tulu.py  --config ./config/tulu_template\.py --dataset_name $dataset --labeling_model $labeling_model
# done



# datasets=('dolly' 'flan_v2' 'oasst1' 'wizardlm' 'stanford_alpaca' 'all_train')
datasets=('all_train') 

# labeling_model='mistralai/Mistral-7B-Instruct-v0.3' 
# labeling_model="gpt-4o-mini"
# labeling_model='meta-llama/Meta-Llama-3.1-8B-Instruct'

# labeling_models=('meta-llama/Meta-Llama-3.1-8B-Instruct' "gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3')

labeling_models=("gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3')

gpus=(0 1 2 3)  # GPU list

for idx in ${!labeling_models[@]}; do
  dataset=${datasets[0]}
  labeling_model=${labeling_models[$idx]}
  gpu=${gpus[$((idx % 4))]}  # 分配 GPU，循环使用 0,1,2,3

  echo "#### processing dataset: ${dataset}"
  echo "#### processing labeling model: ${labeling_model}"


  CUDA_VISIBLE_DEVICES=$gpu python3 diagnose_tulu.py \
    --config ./config/tulu_template.py \
    --dataset_name $dataset \
    --labeling_model $labeling_model &

done
wait  


# python3 ./tools/diagnose_rlhf.py  --config ./config/hh_rlhf_harmless-base.py


# python3 ./tools/diagnose_tulu.py  --config ./config/tulu_template\.py --dataset_name dolly --labeling_model meta-llama/Meta-Llama-3.1-8B-Instruct
