
datasets=('tulu_300k')
rating_models=('meta-llama/Meta-Llama-3.1-8B-Instruct' "gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3')
score_root_path="../scoring_output/"
output_dir="../score_curation_results/"

gpus=(0 1 2 3)  # GPU list

for idx in ${!rating_models[@]}; do
  dataset=${datasets[0]}
  rating_model=${rating_models[$idx]}
  gpu=${gpus[$((idx % 4))]}  

  echo "*** processing dataset: ${dataset} ***"
  echo "*** processing rating model: ${rating_model} ***"


  CUDA_VISIBLE_DEVICES=$gpu python3 diagnose.py \
    --config tulu_template.py \
    --dataset_name $dataset \
    --score_root_path $score_root_path \
    --output_dir $output_dir \
    --rating_model $rating_model &

done

