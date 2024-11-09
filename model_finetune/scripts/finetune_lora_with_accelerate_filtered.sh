# export CUDA_VISIBLE_DEVICES=0,1,2,3

# MODEL_SIZE=7B
# NUM_GPUS=4
# BATCH_SIZE_PER_GPU=1
# TOTAL_BATCH_SIZE=128


# data_type='random-3k' ##filtered random label-filtered 
# labeling_model='meta/llama-3.1-8b-instruct'
# dataset_name='flan_v2'

# base_model='meta-llama/Llama-2-7b-hf'
# # base_model='meta-llama/Meta-Llama-3.1-8B-Instruct'

# # root_path="score_curation/data/${labeling_model}/"
# train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"


# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# # echo "Training data: ${dataset_name}"
# echo "Training data path: ${train_data}"



# ### Lora training
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path $base_model \
#     --use_lora \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     --tokenizer_name $base_model \
#     --use_slow_tokenizer \
#     --train_file $train_data \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 16 \
#     --checkpointing_steps epoch \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 5 \
#     --output_dir output/tulu_${dataset_name}_${MODEL_SIZE}_lora_${data_type}_${labeling_model}/ \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1 &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path $base_model \
#     --lora_model_name_or_path output/tulu_${dataset_name}_${MODEL_SIZE}_lora_${data_type}_${labeling_model}/ \
#     --output_dir output/tulu_${dataset_name}_${MODEL_SIZE}_lora_merged_${data_type}_${labeling_model}/ \
#     --save_tokenizer

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --lora_model_name_or_path output/tulu_flan_v2_7B_lora_filtered/ \
#     --output_dir output/tulu_flan_v2_7B_lora_merged_filtered/ \
#     --save_tokenizer

# --use_deepspeed \
# --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \

# nohup bash ./scripts/finetune_lora_with_accelerate_filtered.sh > zzz_lora_finetune_filtered.log &

##############################################################################################################
## for data size parallel

export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128


# DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list
dataset_name='stanford_alpaca'


labeling_model='meta-llama/Meta-Llama-3.1-8B-Instruct'
base_model='meta-llama/Llama-2-7b-hf'

# base_model='meta-llama/Meta-Llama-3.1-8B-Instruct'

# data_types=('filtered-35k') #'filtered-25k' 'filtered-15k' 'filtered-35k'

# data_types=('diversity-filtered')

data_types=('filtered' 'random')


for data_type in "${data_types[@]}"
do
    # train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"

    train_data="./selected_data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"

    GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
    echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
    echo "Training data path: ${train_data}"

    ### Lora training
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        open_instruct/finetune.py \
        --model_name_or_path $base_model \
        --use_lora \
        --lora_rank 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --tokenizer_name $base_model \
        --use_slow_tokenizer \
        --train_file $train_data \
        --max_seq_length 2048 \
        --preprocessing_num_workers 16 \
        --checkpointing_steps epoch \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate 1e-4 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0. \
        --num_train_epochs 5 \
        --output_dir output/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
        --with_tracking \
        --report_to tensorboard \
        --logging_steps 1

    python open_instruct/merge_lora.py \
        --base_model_name_or_path $base_model \
        --lora_model_name_or_path output/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
        --output_dir output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}/ \
        --save_tokenizer

    rm -rf output/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/
done
