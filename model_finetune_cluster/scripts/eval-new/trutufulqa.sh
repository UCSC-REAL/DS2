
########################################################################
eval_dataset_name='truthfulqa'

train_dataset_name=$1
labeling_model=$2
base_model=$3
# models=$4
# save_dirs=$5
# cuda_devices=$6

# 恢复传递的数组
eval "$4"
eval "$5"
eval "$6"

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