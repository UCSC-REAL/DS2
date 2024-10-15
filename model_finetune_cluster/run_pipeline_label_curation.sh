#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
# TRAIN_DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca' 'all_train') # full data list

TRAIN_DATASET_LIST=('all_train') 

#########################################
############ labeling_models #############

# labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# labeling_model="gpt-4o-mini"
labeling_model='mistralai/Mistral-7B-Instruct-v0.3'

############ labeling_models #############
#########################################


######################################
############ base_models #############

declare -A base_models
# base_models["meta-llama/Llama-2-7b-hf"]="128 1 4096"
base_models["meta-llama/Meta-Llama-3.1-8B"]="64 1 2048"
# base_models["mistralai/Mistral-7B-v0.3"]="128 1 2048"

# base_models["meta-llama/Llama-2-13b-hf"]="8 1 2048"

############ base_models #############
######################################


## # data_types used for ablation study, which determines the finetuned model

# data_types=('filtered' 'completion' 'random' 'label-filtered' 'diversity-filtered'  'perplexity' 'knn' 'less' 'full' 'combined') #baselines

# data_types=('combined') #baselines


# data_types=('label-filtered-cured' 'diversity-filtered-cured' 'filtered-cured') #baselines

confidence_prob=0.3


# data_types=("label-filtered-cured-${confidence_prob}" "diversity-filtered-cured-${confidence_prob}" "filtered-cured-${confidence_prob}") #baselines

# "label-filtered-cured-${confidence_prob}"
data_types=("label-filtered-cured-${confidence_prob}") #baselines

#############################################################
######## model finetuning on selected training data ######### 
#############################################################

echo "###### All data types here:: ${data_types[@]}"
echo "###### All training datasets here:: ${TRAIN_DATASET_LIST[@]}"


cluster_root_path="output" ## . for local

mkdir -p $cluster_root_path

for base_model in "${!base_models[@]}"
do
    IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
    TOTAL_BATCH_SIZE=${params[0]}
    BATCH_SIZE_PER_GPU=${params[1]}
    max_seq_length=${params[2]}


    for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"
    do

        echo "###### Processing training dataset :: ${train_dataset_name}"


        for data_type in "${data_types[@]}"
        do
            echo "###### Processing data type:: ${data_type}"

            if [[ $data_type == "base" ]]; then
                echo "Skipping base model finetune"
                continue  # 跳过 item2，继续下一个 item
            fi

            mkdir -p $cluster_root_path/models/

            # train_data="score_curation/data/${labeling_model}/${dataset_name}/${data_type}_dataset.json"
            echo "Processing data type: $data_type"

            train_data="new_train_data/${labeling_model}/${train_dataset_name}/${data_type}_dataset.json"

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
                --max_seq_length $max_seq_length \
                --preprocessing_num_workers 16 \
                --checkpointing_steps epoch \
                --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
                --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
                --learning_rate 1e-4 \
                --lr_scheduler_type linear \
                --warmup_ratio 0.03 \
                --weight_decay 0. \
                --num_train_epochs 5 \
                --output_dir $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
                --with_tracking \
                --report_to tensorboard \
                --logging_steps 1

            python open_instruct/merge_lora.py \
                --base_model_name_or_path $base_model \
                --lora_model_name_or_path $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/ \
                --output_dir $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}/ \
                --save_tokenizer

            sleep 10s

            rm -rf $cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_${data_type}/

        done
    done
done

sleep 10s


AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/curation_result/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
azcopy copy "$cluster_root_path/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive


echo "starting evaluating finetuned models..."



# # ############################################################
# # ######## ####  finetuned model  evaluation ######## #### 
# # ###########################################################
for base_model in "${!base_models[@]}"; do

    for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do

        for data_type in "${data_types[@]}"; do

            model_name_or_path=$cluster_root_path/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_${data_type}

            if [[ $data_type == "base" ]]; then
                echo "base model evaluation"
                model_name_or_path=$base_model
            fi

            echo "###### Processing data type:: ${data_type}"

            ####MMLU: factual knowledge
            #### ./scripts/eval-new/mmlu.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"
            
            eval_dataset_name='mmlu'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=0 python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir raw_data/eval/mmlu \
            --save_dir ${local_save_dir} \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name_or_path  $model_name_or_path \
            --eval_batch_size 8  &

            # # ### reasoning
            # # ./scripts/eval-new/gsm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
            eval_dataset_name='gsm'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=2 python -m eval.gsm.run_eval \
                --data_dir raw_data/eval/gsm/ \
                --max_num_examples 200 \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path \
                --tokenizer_name_or_path $model_name_or_path \
                --n_shot 8 \
                --use_vllm &

            # BBH: 
            ## ./scripts/eval-new/bbh.sh "$train_dataset_name" "$labeling_model" "$base_model"  "${!models[@]}" "${!save_dirs[@]}" "$cuda_devices"
        
            eval_dataset_name='bbh'
            local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$data_type

            CUDA_VISIBLE_DEVICES=1 python -m eval.bbh.run_eval \
                --data_dir raw_data/eval/bbh \
                --save_dir ${local_save_dir} \
                --model_name_or_path $model_name_or_path  \
                --tokenizer_name_or_path $model_name_or_path \
                --max_num_examples_per_task 40 \
                --use_vllm &

            # # ### truthfulness
            # # ./scripts/eval-new/truthfulqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
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


            # # # # ### multilinguality
            # # # # ./scripts/eval-new/tydiqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
            
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
    for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do

        for data_type in "${data_types[@]}"; do        
        echo "###### Processing Base model :: ${labeling_model}"
        echo "###### Processing Labeling model :: ${base_model}"
        echo "###### Processing training dataset :: ${train_dataset_name}"
        echo "###### Processing data type :: ${data_type}"

        python3 read_results.py --root_result_path "${cluster_root_path}/results" --train_dataset $train_dataset_name --base_model $base_model --labeling_model $labeling_model --baseline_tag $data_type

        done

    done
done 

AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/curation_result/results/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
azcopy copy "$cluster_root_path/results/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive
