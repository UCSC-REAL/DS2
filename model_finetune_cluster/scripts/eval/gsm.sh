
eval_dataset_name='gsm'

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

  CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.gsm.run_eval \
      --data_dir raw_data/eval/gsm/ \
      --max_num_examples 200 \
      --save_dir ${save_dirs[$key]} \
      --model ${models[$key]} \
      --tokenizer ${models[$key]} \
      --n_shot 8 > ./logs/llama_${eval_dataset_name}_${key}.log &
done


# nohup bash ./scripts/eval/gsm.sh > zzz_llama_gsm_normal.log &
# nohup bash ./scripts/eval/gsm.sh > zzz_llama_gsm_bad.log &
# nohup bash ./scripts/eval/gsm.sh > zzz_llama_gsm_good.log &

###################################################################################################################
### evaluate the impact of data size


# eval_dataset_name='gsm'

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


# for key in "${!models[@]}"; do
#   CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.gsm.run_eval \
#       --data_dir data/eval/gsm/ \
#       --max_num_examples 200 \
#       --save_dir ${save_dirs[$key]} \
#       --model ${models[$key]} \
#       --tokenizer ${models[$key]} \
#       --n_shot 8 > zzz_llama_${eval_dataset_name}_${key}.log &
# done

###################################################################################################################



# export CUDA_VISIBLE_DEVICES=1

# # Evaluating llama 7B model using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-good \
#     --model $good_labels_finetuned_model \
#     --tokenizer $good_labels_finetuned_model \
#     --n_shot 8






# # Evaluating llama 7B model using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-cot-8shot \
#     --model ../hf_llama_models/7B \
#     --tokenizer ../hf_llama_models/7B \
#     --n_shot 8 \
#     --use_vllm


# # Evaluating llama 7B model using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-8shot \
#     --model ../hf_llama_models/7B \
#     --tokenizer ../hf_llama_models/7B \
#     --n_shot 8 \
#     --no_cot \
#     --use_vllm


# # Evaluating tulu 7B model using chain-of-thought and chat format
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/tulu-7B-cot-8shot \
#     --model ../checkpoints/tulu_7B \
#     --tokenizer ../checkpoints/tulu_7B \
#     --n_shot 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # Evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama2-chat-7B-cot-8shot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --n_shot 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#     --use_vllm


# # Evaluating chatgpt using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating chatgpt using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot


# # Evaluating gpt4 using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating gpt4 using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-no-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot
