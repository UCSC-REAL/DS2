# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.

# 模型路径
# good_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_good_labels/'
# bad_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_labels/'
# llama_7B_model='meta-llama/Llama-2-7b-hf'


# # 评估结果保存目录
# declare -A save_dirs
# save_dirs=(
#   ["good"]='results/truthfulqa/llama2-7B-'
#   ["bad"]='results/truthfulqa/llama2-7B-random'
#   ["normal"]='results/truthfulqa/llama2-7B-normal'
# )

# # 模型和tokenizer路径
# declare -A models
# models=(
#   ["good"]=$good_labels_finetuned_model
#   ["bad"]=$bad_labels_finetuned_model
#   ["normal"]=$llama_7B_model
# )

# # CUDA 设备
# declare -A cuda_devices
# cuda_devices=(
#   ["good"]=0
#   ["bad"]=1
#   ["normal"]=2
# )

# # 运行评估
# for key in "${!models[@]}"; do
#   CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir ${save_dirs[$key]} \
#     --model_name_or_path ${models[$key]} \
#     --tokenizer_name_or_path ${models[$key]} \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit > zzz_llama_truthfulqa_${key}.log &
# done





# Evaluating llama 7B model, getting the truth and info scores and multiple choice accuracy

# To get the truth and info scores, the original TruthfulQA paper trained 2 judge models based on GPT3 curie engine.
# If you have or trained those judges, you can specify the `gpt_truth_model_name`` and `gpt_info_model_name`,
# which are the names of the GPT models trained following https://github.com/sylinrl/TruthfulQA#fine-tuning-gpt-3-for-evaluation
# But recently Openai has deprecated the GPT3 curie engine, so we also provide the option to use the HF models as the judges.
# The two models provided here are trained based on the llama2 7B model.
# We have the training details in https://github.com/allenai/truthfulqa_reeval, and verified these two models have similar performance as the original GPT3 judges.

# export CUDA_VISIBLE_DEVICES=2

# data_type='filtered' ##filtered random
# labeling_model='meta/llama-3.1-8b-instruct'
# dataset_name='flan_v2'
# model_path="output/tulu_flan_v2_7B_lora_merged_filtered_meta/llama-3.1-8b-instruct/"

########################################################################
eval_dataset_name='truthfulqa'

train_dataset_name='stanford_alpaca'
labeling_model='meta-llama/Meta-Llama-3.1-8B-Instruct'
base_model='meta-llama/Llama-2-7b-hf'

filtered_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_filtered_${labeling_model}" ## 6.6k
random_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_random_${labeling_model}"  # 6.6k
diversity_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_diversity-filtered_${labeling_model}"
label_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_label-filtered_${labeling_model}"

# 模型和tokenizer路径
declare -A models
models=(
  ["filtered"]=$filtered_finetuned_model
  ["random"]=$random_finetuned_model
#   ["full"]=$full_finetuned_model
  # ["label"]=$label_finetuned_model
#   ["base"]=$base_model
# ['diversity']=$diversity_finetuned_model
)

declare -A save_dirs
save_dirs=(
  ["filtered"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/filtered
  ["random"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/random
  ["full"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/full
["label"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/label
  ["base"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/base
  ["diversity"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/diversity
)



# CUDA 设备
declare -A cuda_devices
cuda_devices=(
  ["filtered"]=2
  ["random"]=3
  ["full"]=2
    ["label"]=3
  ["base"]=3
  ["diversity"]=2
)


# 运行评估
for key in "${!models[@]}"; do

  echo "Log file for ${key}: ./logs/llama_${eval_dataset_name}_${key}.log"
  
  CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.truthfulqa.run_eval \
    --data_dir raw_data/eval/truthfulqa \
    --save_dir ${save_dirs[$key]} \
    --model_name_or_path ${models[$key]} \
    --tokenizer_name_or_path ${models[$key]} \
    --metrics truth info mc \
    --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 20 \
    --load_in_8bit > ./logs/llama_${eval_dataset_name}_${key}.log &


done





# python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir results/trutufulqa/llama2-7B-filtered \
#     --model_name_or_path $model_path \
#     --tokenizer_name_or_path $model_path \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit



# # # Evaluating Tulu 7B model using chat format, getting the truth and info scores and multiple choice accuracy
# python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir results/trutufulqa/tulu2-7B/ \
#     --model_name_or_path ../checkpoints/tulu2/7B/ \
#     --tokenizer_name_or_path ../checkpoints/tulu2/7B/ \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using chat format, getting the truth and info scores and multiple choice accuracy
# python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir results/trutufulqa/llama2-chat-7B \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --metrics truth info mc \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt, getting the truth and info scores
# # Multiple choice accuracy is not supported for chatgpt, since we cannot get the probabilities from chatgpt
# python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir results/trutufulqa/chatgpt \
#     --openai_engine gpt-3.5-turbo-0301 \
#     --metrics truth info \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20

# # Evaluating gpt-4, getting the truth and info scores
# # Multiple choice accuracy is not supported for gpt-4, since we cannot get the probabilities from gpt-4
# python -m eval.truthfulqa.run_eval \
#     --data_dir raw_data/eval/truthfulqa \
#     --save_dir results/trutufulqa/gpt4 \
#     --openai_engine gpt-4-0314 \
#     --metrics truth info \
#     --preset qa \
#     --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#     --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#     --eval_batch_size 20