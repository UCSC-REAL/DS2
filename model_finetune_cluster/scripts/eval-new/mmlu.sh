
eval_dataset_name='mmlu'

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

for key in "${!models[@]}"; do

  echo "Log file for ${key}: ./logs/llama_${eval_dataset_name}_${key}.log"


    CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir raw_data/eval/mmlu \
    --save_dir ${save_dirs[$key]} \
    --model_name_or_path ${models[$key]} \
    --tokenizer_name_or_path  ${models[$key]} \
    --eval_batch_size 16 > ./logs/llama_${eval_dataset_name}_${key}.log &

done


################################################################################################################
# compare the data size


# eval_dataset_name='mmlu'

# # 设置基准类型，选择 'filtered' 或 'random'
# base_type='random'  # 或者设置为 'random'

# # 定义数据集大小
# sizes=('3k' '15k' '25k' '35k')

# # 初始化 CUDA 设备数组
# declare -A cuda_devices
# gpu_index=0

# # 动态生成 cuda_devices 数组
# for size in "${sizes[@]}"; do
#     data_type="${base_type}-${size}"
#     cuda_devices[$data_type]=$gpu_index
#     gpu_index=$(( (gpu_index + 1) % 4 ))  # 假设有 4 个 GPU，循环使用它们
# done

# # 初始化 data_types 数组
# data_types=("${!cuda_devices[@]}")

# # 定义模型路径
# declare -A models
# for data_type in "${data_types[@]}"; do
#     models[$data_type]="output/tulu_flan_v2_7B_lora_merged_${data_type}_meta/llama-3.1-8b-instruct/"
# done

# # 定义保存路径
# declare -A save_dirs
# for data_type in "${data_types[@]}"; do
#     save_dirs[$data_type]="results/${eval_dataset_name}/llama2-7B-${data_type}"
# done

# # 运行评估
# for key in "${!models[@]}"; do
#     CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir ${save_dirs[$key]} \
#     --model_name_or_path ${models[$key]} \
#     --tokenizer_name_or_path ${models[$key]} \
#     --eval_batch_size 16 > zzz_llama_${eval_dataset_name}_${key}.log &
# done




# Evaluating llama 7B model using 0 shot directly


# #### random baseline
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-random \
#     --model_name_or_path ./output/tulu_v2_7B_lora_merged_random\
#     --tokenizer_name_or_path ./output/tulu_v2_7B_lora_merged_random \
#     --eval_batch_size 4 \
#     --load_in_8bit

# #### filtered version
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-filtered \
#     --model_name_or_path ./output/tulu_v2_7B_lora_merged_filtered \
#     --tokenizer_name_or_path ./output/tulu_v2_7B_lora_merged_filtered \
#     --eval_batch_size 4 \
#     --load_in_8bit

# good_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_good_labels/'
# bad_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_bad_labels/'
# llama_7B_model='meta-llama/Llama-2-7b-hf'
# #### full data version
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B \
#     --model_name_or_path $llama_7B_model \
#     --tokenizer_name_or_path  $llama_7B_model \
#     --eval_batch_size 16 \
#     --load_in_8bit




# # Evaluating llama 7B model using 0 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-0shot \
#     --model_name_or_path ../hf_llama_models/7B \
#     --tokenizer_name_or_path ../hf_llama_models/7B \
#     --eval_batch_size 4 \
#     --load_in_8bit

####################################################################################################################################

# # Evaluating llama 7B model using 5 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-5shot \
#     --model_name_or_path ../hf_llama_models/7B \
#     --tokenizer_name_or_path ../hf_llama_models/7B \
#     --eval_batch_size 4 \
#     --load_in_8bit


# # Evaluating Tulu 7B model using 0 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-0shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating Tulu 7B model using 5 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-5shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using 0-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating llama2 chat model using 5-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating gpt4 using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # Evaluating gpt4 using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir raw_data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20