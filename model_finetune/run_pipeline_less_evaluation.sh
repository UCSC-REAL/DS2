#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
# TRAIN_DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca' 'all_train') # full data list

TRAIN_DATASET_LIST=('all_train') 

#########################################
############ labeling_models #############

labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# labeling_model="gpt-4o-mini"
# labeling_model='mistralai/Mistral-7B-Instruct-v0.3'

############ labeling_models #############
#########################################


######################################
############ base_models #############

declare -A base_models
# base_models["meta-llama/Llama-2-7b-hf"]="128 4 4096"  
base_models["meta-llama/Meta-Llama-3.1-8B"]="128 2 2048" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
# base_models["mistralai/Mistral-7B-v0.3"]="128 4 2048"


# base_models["meta-llama/Llama-2-13b-hf"]="128 1 2048"
# base_models["meta-llama/Meta-Llama-3.1-70B"]="8 1 1024"


############ base_models #############
######################################


## # data_types used for ablation study, which determines the finetuned model

# data_types=( 'completion' 'perplexity' 'knn' 'less' 'full' 'random' 'label-filtered' 'diversity-filtered' 'filtered' ) #baselines


# data_types=('less-mmlu-half')
# data_types=('less' 'less-bbh' 'less-gsm' 'less-mmlu' 'less-truthfulqa' 'less-tydiqa')
# data_types=('less-bbh-half' 'less-gsm-half' 'less-mmlu-half' 'less-truthfulqa-half' 'less-tydiqa-half')
data_types=('less-bbh-qa-pair')
# data_types=('test')
#############################################################
######## model finetuning on selected training data ######### 
#############################################################


echo "###### All data types here:: ${data_types[@]}"
echo "###### All training datasets here:: ${TRAIN_DATASET_LIST[@]}"


cluster_root_path="/mnt/server0-A/jinlong/less-output" ##  

mkdir -p $cluster_root_path

# for base_model in "${!base_models[@]}"
# do
#     IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
#     TOTAL_BATCH_SIZE=${params[0]}
#     BATCH_SIZE_PER_GPU=${params[1]}
#     max_seq_length=${params[2]}


#     for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"
#     do

#         echo "###### Processing training dataset :: ${train_dataset_name}"


#         for data_type in "${data_types[@]}"
#         do
#             echo "###### Processing data type:: ${data_type}"

#             if [[ $data_type == "base" ]]; then
#                 echo "Skipping base model finetune"
#                 continue     
#             fi

#             mkdir -p $cluster_root_path/models/

#             # train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"
#             echo "Processing data type: $data_type"

#             train_data="selected_data/${labeling_model}/${train_dataset_name}/${data_type}_dataset.json"

#             echo "train data path: $train_data"

#             GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
#             echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
#             echo "Training data path: ${train_data}"

#             ### Lora training                 
#             # 
#             # --use_qlora \ 
#             # --use_deepspeed \
#             # --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#             accelerate launch \
#                 --mixed_precision bf16 \
#                 --num_machines 1 \
#                 --num_processes $NUM_GPUS \
#                 --fsdp_sharding_strategy FULL_SHARD \
#                 --fsdp_offload_params true \
#                 open_instruct/finetune.py \
#                 --model_name_or_path $base_model \
#                 --gradient_checkpointing \
#                 --use_lora \
#                 --lora_rank 64 \
#                 --lora_alpha 16 \
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
#                 --output_dir $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
#                 --with_tracking \
#                 --report_to tensorboard \
#                 --logging_steps 1
#                 # --use_qlora 

#             python open_instruct/merge_lora.py \
#                 --base_model_name_or_path $base_model \
#                 --lora_model_name_or_path $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
#                 --output_dir $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}/ \
#                 --save_tokenizer
#                 # --qlora

#             sleep 10s

#             rm -rf $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}

#         done
#     done
# done


# wait
# sleep 10s

# ############################################################
# ######## ####  finetuned model  evaluation ######## #### 
# ###########################################################

for base_model in "${!base_models[@]}"; do

    for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do

        for data_type in "${data_types[@]}"; do

            model_name_or_path=$cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}

            if [[ $data_type == "base" ]]; then
                echo "base model evaluation"
                model_name_or_path=$base_model
            fi

            echo "###### Processing data type:: ${data_type}"

            ##MMLU: factual knowledge
            ### ./scripts/eval/mmlu.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"
            
            eval_dataset_name='mmlu'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=7 python -m eval.mmlu.run_eval \
            --ntrain 5 \
            --data_dir raw_data/eval/mmlu \
            --save_dir ${local_save_dir} \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path  $model_name_or_path \
            --eval_batch_size 4  \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format &

            # ### reasoning
            #### ./scripts/eval/gsm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
            eval_dataset_name='gsm'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=2,5 python -m eval.gsm.run_eval \
                --data_dir raw_data/eval/gsm/ \
                --max_num_examples 200 \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path \
                --tokenizer_name_or_path $model_name_or_path \
                --n_shot 8 \
                --use_vllm &

            ### BBH: 
            #### ./scripts/eval/bbh.sh "$train_dataset_name" "$labeling_model" "$base_model"  "${!models[@]}" "${!save_dirs[@]}" "$cuda_devices"
        
            eval_dataset_name='bbh'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=1 python -m eval.bbh.run_eval \
                --data_dir raw_data/eval/bbh \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path  \
                --tokenizer_name_or_path $model_name_or_path \
                --max_num_examples_per_task 40 \
                --use_vllm &

            ### truthfulness
            #### ./scripts/eval/truthfulqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
            eval_dataset_name='truthfulqa'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

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


            # # ### multilinguality
            # # ./scripts/eval/tydiqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
            eval_dataset_name='tydiqa'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

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
    echo "###### Processing Base model :: ${labeling_model}"
    echo "###### Processing Labeling model :: ${base_model}"
    
    for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do
        echo "###### Processing training dataset :: ${train_dataset_name}"

        for data_type in "${data_types[@]}"; do        
            python3 read_results.py --root_result_path "${cluster_root_path}/results" --train_dataset $train_dataset_name --base_model $base_model --labeling_model $labeling_model --baseline_tag $data_type

        done

    done
done 
