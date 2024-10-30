# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=False

eval_dataset_name='alpaca'


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
