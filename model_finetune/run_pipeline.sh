#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
SEED=42 

RAW_DATASET_LIST=('tulu_300k') # data source
rating_model="meta-llama/Meta-Llama-3.1-8B-Instruct" #"gpt-4o-mini" 'mistralai/Mistral-7B-Instruct-v0.3'

declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="128 1 2048"  # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
base_models["meta-llama/Llama-3.2-3B"]="32 1 128"  # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length

# data types represent the generated subsets by baselines
data_types=('ds2_10k')  


#############################################################
######## model finetuning on selected training data ######### 
#############################################################

cluster_root_path="../model_output" 
mkdir -p $cluster_root_path

# for base_model in "${!base_models[@]}"
# do
#     IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
#     TOTAL_BATCH_SIZE=${params[0]}
#     BATCH_SIZE_PER_GPU=${params[1]}
#     max_seq_length=${params[2]}


#     for raw_dataset_name in "${RAW_DATASET_LIST[@]}"
#     do

#         for data_type in "${data_types[@]}"
#         do

#             if [[ $data_type == "base" ]]; then
#                 echo "Skipping base model finetune"
#                 continue     
#             fi

#             mkdir -p $cluster_root_path/models/
#             train_data="../selected_data/${rating_model}/${raw_dataset_name}/${data_type}_dataset.json"

#             GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
#             echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
#             echo "Training data path: ${train_data}"

#             ### Lora training
#             accelerate launch \
#                 --mixed_precision bf16 \
#                 --num_machines 1 \
#                 --num_processes $NUM_GPUS \
#                 finetune.py \
#                 --model_name_or_path $base_model \
#                 --use_lora \
#                 --lora_rank 64 \
#                 --lora_alpha 16 \
#                 --seed $SEED \
#                 --lora_dropout 0.1 \
#                 --tokenizer_name $base_model \
#                 --use_slow_tokenizer \
#                 --train_file $train_data \
#                 --max_seq_length $max_seq_length \
#                 --preprocessing_num_workers 16 \
#                 --checkpointing_steps epoch \
#                 --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#                 --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#                 --learning_rate 1e-4 \
#                 --lr_scheduler_type linear \
#                 --warmup_ratio 0.03 \
#                 --weight_decay 0. \
#                 --num_train_epochs 5 \
#                 --output_dir $cluster_root_path/models/${rating_model}/${raw_dataset_name}/${base_model}/lora_${data_type}/ \
#                 --with_tracking \
#                 --report_to tensorboard \
#                 --logging_steps 1

#             python merge_lora.py \
#                 --base_model_name_or_path $base_model \
#                 --lora_model_name_or_path $cluster_root_path/models/${rating_model}/${raw_dataset_name}/${base_model}/lora_${data_type}/ \
#                 --output_dir $cluster_root_path/models/${rating_model}/${raw_dataset_name}/${base_model}/lora_merged_${data_type}/ \
#                 --save_tokenizer

#             sleep 10s

#             rm -rf $cluster_root_path/models/${rating_model}/${raw_dataset_name}/${base_model}/lora_${data_type}

#         done
#     done
# done


wait

############################################################
###############  finetuned model evaluation ################
############################################################

echo "starting evaluating finetuned models..."

for base_model in "${!base_models[@]}"; do

    for raw_dataset_name in "${RAW_DATASET_LIST[@]}"; do

        for data_type in "${data_types[@]}"; do

            model_name_or_path=$cluster_root_path/models/${rating_model}/${raw_dataset_name}/${base_model}/lora_merged_${data_type}

            if [[ $data_type == "base" ]]; then
                echo "base model evaluation"
                model_name_or_path=$base_model
            fi

            echo "###### Processing data type:: ${data_type}"

            #### MMLU: factual knowledge            
            eval_dataset_name='mmlu'
            local_save_dir=${cluster_root_path}/results/${rating_model}/${raw_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=0 python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir raw_data/eval/mmlu \
            --save_dir ${local_save_dir} \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path  $model_name_or_path \
            --eval_batch_size 8  &

            ##### GSM8k: reasoning            
            eval_dataset_name='gsm'
            local_save_dir=${cluster_root_path}/results/${rating_model}/${raw_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=1 python -m eval.gsm.run_eval \
                --data_dir raw_data/eval/gsm/ \
                --max_num_examples 200 \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path \
                --tokenizer_name_or_path $model_name_or_path \
                --n_shot 8 \
                --use_vllm &

            ###### BBH: reasoning
            eval_dataset_name='bbh'
            local_save_dir=${cluster_root_path}/results/${rating_model}/${raw_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=2 python -m eval.bbh.run_eval \
                --data_dir raw_data/eval/bbh \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path  \
                --tokenizer_name_or_path $model_name_or_path \
                --max_num_examples_per_task 40 \
                --use_vllm &

            ##### truthfulness            
            eval_dataset_name='truthfulqa'
            local_save_dir=${cluster_root_path}/results/${rating_model}/${raw_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=3 python -m eval.truthfulqa.run_eval \
                --data_dir raw_data/eval/truthfulqa \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path \
                --tokenizer_name_or_path $model_name_or_path \
                --metrics truth info mc \
                --preset qa \
                --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
                --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
                --eval_batch_size 20 \
                --load_in_8bit &


            ###### multilinguality            
            eval_dataset_name='tydiqa'
            local_save_dir=${cluster_root_path}/results/${rating_model}/${raw_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=4 python -m eval.tydiqa.run_eval \
                --data_dir raw_data/eval/tydiqa/ \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path \
                --tokenizer_name_or_path $model_name_or_path \
                --eval_batch_size 20 \
                --load_in_8bit &

            wait

        done

    done
done

sleep 10s

for base_model in "${!base_models[@]}"; do
    for raw_dataset_name in "${RAW_DATASET_LIST[@]}"; do

        for data_type in "${data_types[@]}"; do        
        echo "*** Processing rating model:: ${rating_model} ***"
        echo "*** Processing Base model:: ${base_model} ***"
        echo "*** Processing training dataset:: ${raw_dataset_name} ***"
        echo "*** Processing data type:: ${data_type} ***"

        python3 read_results.py --root_result_path "${cluster_root_path}/results" --raw_dataset $raw_dataset_name --base_model $base_model --rating_model $rating_model --baseline_tag $data_type

        done

    done
done 
