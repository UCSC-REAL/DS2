# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=False

# use vllm for generation
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --tokenizer_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --save_dir results/alpaca_farm/tulu_v1_7B/ \
#     --eval_batch_size 20 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

eval_dataset_name='alpaca'


filtered_finetuned_model="output/tulu_flan_v2_7B_lora_merged_filtered_meta/llama-3.1-8b-instruct/"
random_finetuned_model="output/tulu_flan_v2_7B_lora_merged_random_meta/llama-3.1-8b-instruct/"
label_finetuned_model="output/tulu_flan_v2_7B_lora_merged_label-filtered_meta/llama-3.1-8b-instruct/"
llama_7B_model='meta-llama/Llama-2-7b-hf'
full_finetuned_model="output/tulu_v2_7B_lora_merged_full_data"

# 模型和tokenizer路径
declare -A models
models=(
  ["filtered"]=$filtered_finetuned_model
  ["random"]=$random_finetuned_model
  ["full"]=$full_finetuned_model
  ["label"]=$label_finetuned_model
  ["base"]=$llama_7B_model
)

declare -A save_dirs
save_dirs=(
  ["filtered"]=results/${eval_dataset_name}/llama2-7B-filtered
  ["random"]=results/${eval_dataset_name}/llama2-7B-random
  ["full"]=results/${eval_dataset_name}/llama2-7B-full
["label"]=results/${eval_dataset_name}/llama2-7B-label
  ["base"]=results/${eval_dataset_name}/llama2-7B-base
)



# CUDA 设备
declare -A cuda_devices
cuda_devices=(
  ["filtered"]=0
  ["random"]=1
  ["full"]=2
  ["base"]=3
    ["label"]=3
)


for key in "${!models[@]}"; do
  CUDA_VISIBLE_DEVICES=${cuda_devices[$key]} nohup python -m eval.alpaca_farm.run_eval \
    --model_name_or_path ${models[$key]} \
    --tokenizer_name_or_path ${models[$key]} \
    --save_dir ${save_dirs[$key]} \
    --eval_batch_size 20 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --load_in_8bit > zzz_llama_${eval_dataset_name}_${key}.log &
done


# use normal huggingface generation function
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --tokenizer_name_or_path ../checkpoints/tulu_v1_7B/ \
#     --save_dir results/alpaca_farm/tulu_v1_7B/ \
#     --eval_batch_size 20 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --load_in_8bit
