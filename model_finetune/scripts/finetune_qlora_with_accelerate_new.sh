export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8


train_dataset_name=$1
labeling_model=$2
base_model=$3
data_type=$4
$cluster_root_path=$5
TOTAL_BATCH_SIZE=$6  #128
BATCH_SIZE_PER_GPU=$7
max_seq_length=$8


mkdir -p output/models/


# train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"
echo "Processing data type: $data_type"

train_data="selected_data/${labeling_model}/${train_dataset_name}/${data_type}_dataset.json"

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Training data path: ${train_data}"

# Lora training
accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path $base_model \
    --gradient_checkpointing \
    --use_qlora \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name $base_model  \
    --use_slow_tokenizer \
    --train_file $train_data \
    --max_seq_length $max_seq_length \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path $base_model \
    --lora_model_name_or_path output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
    --output_dir output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}/ \
    --qlora \
    --save_tokenizer

sleep 30s

rm -rf output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/


# accelerate launch \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     open_instruct/finetune.py \
#     --model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --gradient_checkpointing \
#     --use_qlora \
#     --use_lora \
#     --use_flash_attn \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     --tokenizer_name ../hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/tulu_v2/tulu_v2_data.jsonl \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-4 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 5 \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_qlora/ \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1 &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_qlora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --qlora \
#     --save_tokenizer
