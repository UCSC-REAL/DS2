

# data_types=('label-filtered-2k' 'label-filtered-5k' 'label-filtered-20k' 'label-filtered-40k')

# data_types=('diversity-filtered-2k' 'diversity-filtered-5k' 'diversity-filtered-20k' 'diversity-filtered-40k')


# for data_type in ${data_types[@]}; do

#     AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/data_scale_result/models/meta-llama/Meta-Llama-3.1-8B-Instruct/all_train/meta-llama/Meta-Llama-3.1-8B/lora_merged_${data_type}?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
#     azcopy copy "$AZURE_STORAGE_CONTAINER_URL" "./data_scale_models/"  --recursive

# done



###########################################################################################


data_types=('label-filtered-2k' 'label-filtered-5k' 'label-filtered-20k' 'label-filtered-40k')

# GPU ID 列表
gpu_ids=(0 1 2 3)

# Evaluating llama 7B model using chain-of-thought
for i in "${!data_types[@]}"; do
  data_type="${data_types[$i]}"
  
  # 获取当前循环中的 GPU ID，使用取模操作来确保循环分配 GPU
  gpu_id=${gpu_ids[$((i % ${#gpu_ids[@]}))]}

  mode_path="/home/azureuser/cloudfiles/code/Users/jinlong.pang/LADR_LLM_alignment_data_refinement/open-instruct/data_scale_models/lora_merged_${data_type}"

  CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -m eval.truthfulqa.run_eval \
    --data_dir raw_data/eval/truthfulqa \
    --save_dir results/trutufulqa/llama/ \
    --model_name_or_path ${mode_path} \
    --tokenizer_name_or_path ${mode_path} \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --load_in_8bit > zzzzzz_llama_truthfulqa_${data_type}.log &


done