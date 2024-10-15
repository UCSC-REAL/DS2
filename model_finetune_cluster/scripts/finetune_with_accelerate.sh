export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32

data_type='random'
labeling_model='gemma'
dataset_name='flan_v2'

# root_path="score_curation/data/${labeling_model}/"
train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# echo "Training data: ${dataset_name}"
echo "Training data path: ${train_data}"


# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.


############################################ ./scripts/finetune_with_accelerate.sh
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --train_file $train_data \
    --use_slow_tokenizer \
    --max_seq_length 1024 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/tulu_v2_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 

# # --------------------------- original version --------------------------- 
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
        # --use_deepspeed \
        # --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name meta-llama/Llama-2-7b-hf \
#     --use_slow_tokenizer \
#     --train_file score_curation/data/gemma/flan_v2/filtered_dataset.json  \
#     --max_seq_length 8192 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 2 \
#     --output_dir output/tulu_v2_${MODEL_SIZE}/ \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1


python3  open_instruct/finetune.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--tokenizer_name meta-llama/Llama-2-7b-hf \
--use_slow_tokenizer \
--train_file score_curation/data/gemma/flan_v2/filtered_dataset.json \
--output_dir output/tulu_v2_7B/ \
--max_seq_length 4096 \
--preprocessing_num_workers 128 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--learning_rate 2e-5 \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0. \
--num_train_epochs 2 \
--output_dir output/tulu_v2_7B/ \
--with_tracking \
--report_to tensorboard \
--logging_steps 1 \
--low_cpu_mem_usage