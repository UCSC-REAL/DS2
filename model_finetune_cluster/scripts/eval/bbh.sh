
###################################################################################################
eval_dataset_name='bbh'

train_dataset_name=$1
labeling_model=$2
base_model=$3
models=$4
save_dirs=$5
cuda_devices=$6

# 恢复传递的数组
eval "$4"
eval "$5"
eval "$6"

# sleep 4h

for key in "${!models[@]}"; do

  echo "Log file for ${key}: ./logs/llama_${eval_dataset_name}_${key}.log"

  CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.bbh.run_eval \
    --data_dir raw_data/eval/bbh \
    --save_dir ${save_dirs[$key]} \
    --model ${models[$key]}  \
    --tokenizer ${models[$key]} \
    --max_num_examples_per_task 40 \
    --use_vllm > ./logs/llama_${eval_dataset_name}_${key}.log &


done


###################################################################################################

# eval_dataset_name='bbh'

# # 定义数据集大小
# sizes=('3k' '15k' '25k' '35k')

# # 初始化 CUDA 设备数组
# declare -A cuda_devices
# gpu_index=0
# base_type='random' # 循环处理 'filtered' 和 'random' 两种情况


# # 更新 CUDA 设备数组
# for size in "${sizes[@]}"; do
#     data_type="${base_type}-${size}"
#     cuda_devices[$data_type]=$gpu_index
#     gpu_index=$(( (gpu_index + 1) % 4 ))  # 假设有 4 个 GPU，循环使用它们
# done

# # 初始化 data_types 数组
# data_types=("${!cuda_devices[@]}")

# # 初始化模型路径
# declare -A models
# for data_type in "${data_types[@]}"; do
#     models[$data_type]="output/tulu_flan_v2_7B_lora_merged_${data_type}_meta/llama-3.1-8b-instruct/"
# done

# # 定义保存路径
# declare -A save_dirs
# for data_type in "${data_types[@]}"; do
#     save_dirs[$data_type]="results/${eval_dataset_name}/llama2-7B-${data_type}"
# done

# # 运行评估任务
# for key in "${!models[@]}"; do
#     CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.bbh.run_eval \
#         --data_dir raw_data/eval/bbh \
#         --save_dir ${save_dirs[$key]} \
#         --model ${models[$key]}  \
#         --tokenizer ${models[$key]} \
#         --max_num_examples_per_task 40 > zzz_llama_${eval_dataset_name}_${key}.log &
# done





###################################################################################################

# export CUDA_VISIBLE_DEVICES=1

# good_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_good_labels/'
# bad_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_bad_labels/'
# llama_7B_model='meta-llama/Llama-2-7b-hf'

# #nohup bash ./scripts/eval/bbh.sh > zzz_llama_bbh_good.log &
# # nohup bash ./scripts/eval/bbh.sh > zzz_llama_bbh_bad.log &
# # nohup bash ./scripts/eval/bbh.sh > zzz_llama_bbh_normal.log &
# CUDA_VISIBLE_DEVICES=3 nohup python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/llama-7B-cot-normal/ \
#     --model $bad_labels_finetuned_model \
#     --tokenizer $bad_labels_finetuned_model \
#     --max_num_examples_per_task 40 > zzz_llama_bbh_bad_labels.log &

###################################################################################################
# evaluating llama 7B model using chain-of-thought
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/llama-7B-cot/ \
#     --model ../hf_llama_models/7B \
#     --tokenizer ../hf_llama_models/7B \
#     --max_num_examples_per_task 40 \
#     --use_vllm


# # evaluating llama 7B model using direct answering (no chain-of-thought)
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/llama-7B-no-cot/ \
#     --model ../hf_llama_models/7B \
#     --tokenizer ../hf_llama_models/7B \
#     --max_num_examples_per_task 40 \
#     --use_vllm \
#     --no_cot


# # evaluating tulu 7B model using chain-of-thought and chat format
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/tulu-7B-cot/ \
#     --model ../checkpoint/tulu_7B \
#     --tokenizer ../checkpoints/tulu_7B \
#     --max_num_examples_per_task 40 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/llama2-chat-7B-cot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --max_num_examples_per_task 40 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # evaluating gpt-3.5-turbo-0301 using chain-of-thought
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/chatgpt-cot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40


# # evaluating gpt-3.5-turbo-0301 using direct answering (no chain-of-thought)
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/chatgpt-no-cot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40 \
#     --no_cot


# # evaluating gpt-4 using chain-of-thought
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/gpt4-cot/ \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40


# # evaluating gpt-4 using direct answering (no chain-of-thought)
# python -m eval.bbh.run_eval \
#     --data_dir raw_data/eval/bbh \
#     --save_dir results/bbh/gpt4-no-cot/ \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40 \
#     --no_cot