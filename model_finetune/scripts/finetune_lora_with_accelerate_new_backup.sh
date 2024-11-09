
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128


train_dataset_name=$1
labeling_model=$2
base_model=$3
data_types=$4
$cluster_root_path=$5

# data_types=('filtered-35k') #'filtered-25k' 'filtered-15k' 'filtered-35k')

# data_types=('diversity-filtered')



for data_type in "${data_types[@]}"
do
    # train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"

    train_data="./selected_data/${labeling_model}/${train_dataset_name}/${data_type}_dataset.json"

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
        --output_dir ${cluster_root_path}/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
        --with_tracking \
        --report_to tensorboard \
        --logging_steps 1

    python open_instruct/merge_lora.py \
        --base_model_name_or_path $base_model \
        --lora_model_name_or_path ${cluster_root_path}/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
        --output_dir ${cluster_root_path}/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}/ \
        --save_tokenizer

    sleep 30s

    rm -rf ${cluster_root_path}/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/
done
