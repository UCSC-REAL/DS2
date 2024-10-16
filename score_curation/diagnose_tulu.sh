
# datasets=('dolly' 'flan_v2' 'oasst1' 'wizardlm' 'stanford_alpaca' 'all_train')
datasets=('all_train') 

# labeling_models=('meta-llama/Meta-Llama-3.1-8B-Instruct' "gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3')

labeling_models=("gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3')

gpus=(0 1 2 3)  # GPU list

for idx in ${!labeling_models[@]}; do
  dataset=${datasets[0]}
  labeling_model=${labeling_models[$idx]}
  gpu=${gpus[$((idx % 4))]}  # allocate gpu

  echo "#### processing dataset: ${dataset}"
  echo "#### processing labeling model: ${labeling_model}"


  CUDA_VISIBLE_DEVICES=$gpu python3 diagnose_tulu.py \
    --config ./config/tulu_template.py \
    --dataset_name $dataset \
    --labeling_model $labeling_model &

done
wait  


# python3 diagnose_tulu.py  --config ./config/tulu_template\.py --dataset_name dolly --labeling_model meta-llama/Meta-Llama-3.1-8B-Instruct
