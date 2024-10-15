# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
# export CUDA_VISIBLE_DEVICES=2

# good_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_good_labels/'
# bad_labels_finetuned_model='./output/tulu_flan_v2_7B_lora_merged_filtered_bad_labels/'
# llama_7B_model='meta-llama/Llama-2-7b-hf'

# CUDA_VISIBLE_DEVICES=3 nohup python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-goldp-bad \
#     --model $bad_labels_finetuned_model \
#     --tokenizer $bad_labels_finetuned_model \
#     --eval_batch_size 20 \
#     --load_in_8bit > zzz_llama_tydiqa_bad_labels.log &


################################################################################################

eval_dataset_name='tydiqa'

train_dataset_name='stanford_alpaca'
labeling_model='meta-llama/Meta-Llama-3.1-8B-Instruct'

# base_model='meta-llama/Llama-2-7b-hf'

base_model="meta-llama/Meta-Llama-3.1-8B"


# output/tulu_${dataset_name}_7B_lora_merged_${data_type}_${labeling_model}/ template

filtered_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_filtered_${labeling_model}" ## 6.6k
random_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_random_${labeling_model}"  # 6.6k
diversity_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_diversity-filtered_${labeling_model}"
label_finetuned_model="output-backup/tulu_${train_dataset_name}_7B_lora_merged_label-filtered_${labeling_model}"

full_finetuned_model="output/tulu_v2_7B_lora_merged_full_data"

# 模型和tokenizer路径
declare -A models
models=(
  # ["filtered"]=$filtered_finetuned_model
  # ["random"]=$random_finetuned_model
#   ["full"]=$full_finetuned_model
  # ["label"]=$label_finetuned_model
  ["base"]=$base_model
# ['diversity']=$diversity_finetuned_model
)



declare -A save_dirs
save_dirs=(
  ["filtered"]=results/${eval_dataset_name}/llama2-7B-filtered
  ["random"]=results/${eval_dataset_name}/llama2-7B-random
  ["full"]=results/${eval_dataset_name}/llama2-7B-full
["label"]=results/${eval_dataset_name}/llama2-7B-label
  ["base"]=results/${eval_dataset_name}/llama2-7B-base
  ["diversity"]=results/${eval_dataset_name}/llama2-7B-diversity
)



# CUDA 设备
declare -A cuda_devices
cuda_devices=(
  ["filtered"]=3
  ["random"]=2
  ["full"]=2
    ["label"]=3
  ["base"]=3
  ["diversity"]=0
)

# sleep 7h

for key in "${!models[@]}"; do
  CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.tydiqa.run_eval \
    --data_dir raw_data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir ${save_dirs[$key]} \
    --model ${models[$key]} \
    --tokenizer ${models[$key]} \
    --eval_batch_size 20 \
    --load_in_8bit > ./logs/llama_${eval_dataset_name}_${key}.log &
done
####################################################################################


# eval_dataset_name='tydiqa'

# # 设置基准类型，选择 'filtered' 或 'random'
# base_type='filtered'  # 或者设置为 'random'

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
#   CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir ${save_dirs[$key]} \
#     --model ${models[$key]} \
#     --tokenizer ${models[$key]} \
#     --eval_batch_size 20 \
#     --load_in_8bit > zzz_llama_${eval_dataset_name}_${key}.log &
# done

####################################################################################

# llama_7B_model='meta-llama/Llama-2-7b-hf'

# CUDA_VISIBLE_DEVICES=3 nohup python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-base \
#     --model $llama_7B_model \
#     --tokenizer $llama_7B_model \
#     --eval_batch_size 20 \
#     --load_in_8bit  > zzz_llama_tydiqa_base.log &



# Evaluating llama 7B model, with gold passage provided
# By default, we use 1-shot setting, and 100 examples per language
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-goldp \
#     --model ../hf_llama_model/7B \
#     --tokenizer ../hf_llama_model/7B \
#     --eval_batch_size 20 \
#     --load_in_8bit


# # Evaluating llama 7B model, with no context provided (closed-book QA)
# # By default, we use 1-shot setting, and 100 examples per language
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-no-context \
#     --model ../hf_llama_model/7B \
#     --tokenizer ../hf_llama_model/7B \
#     --eval_batch_size 40 \
#     --load_in_8bit \
#     --no_context  

# # Evaluating Tulu 7B model, with gold passage provided
# # For Tulu, we use chat format.
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/tulu-7B-goldp \
#     --model ../checkpoints/tulu_7B \
#     --tokenizer ../checkpoints/tulu_7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating Tulu 7B model, with no context provided (closed-book QA)
# # For Tulu, we use chat format.
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/tulu-7B-no-context \
#     --model ../checkpoints/tulu_7B \
#     --tokenizer ../checkpoints/tulu_7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --no_context \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model, with gold passage provided
# # For llama2 chat model, we use chat format.
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama2-chat-7B-goldp \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating llama2 chat model, with no context provided (closed-book QA)
# # For llama2 chat model, we use chat format.
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama2-chat-7B-no-context \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --no_context \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt, with gold passage provided
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/chatgpt-goldp-1shot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt, with no context provided (closed-book QA)
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/chatgpt-no-context-1shot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --no_context 


# # Evaluating gpt4, with gold passage provided
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/gpt4-goldp-1shot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20


# # Evaluating gpt4, with no context provided (closed-book QA)
# python -m eval.tydiqa.run_eval \
#     --data_dir raw_data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/gpt4-no-context-1shot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --no_context 