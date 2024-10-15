#!/bin/bash

# TRAIN_DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca' 'all_train') # full data list

TRAIN_DATASET_LIST=('all_train') 

### labeling models
# labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# labeling_model="gpt-4o-mini"
labeling_model='mistralai/Mistral-7B-Instruct-v0.3'


### base_models 
# base_model='meta-llama/Llama-2-7b-hf'
# base_model="meta-llama/Meta-Llama-3.1-8B"
# base_model="meta-llama/Llama-2-13b-hf"
base_model='mistralai/Mistral-7B-v0.3'

## # data_types used for ablation study, which determines the finetuned model

#  data_types=('filtered-35k' 'filtered-25k' 'filtered-15k' 'filtered-3k')
# data_types=('base' 'filtered' 'random' 'completion' 'label-filtered' 'diversity-filtered' 'perplexity' 'knn' 'less' 'full' ) #baselines

data_types=('filtered')

#########################################################################################################################################################
## system root path
cluster_root_path="output" ## . for local
# cluster_root_path="" ## . for local

mkdir -p $cluster_root_path

echo "###### All data types here:: ${data_types[@]}"
echo "###### All training datasets here:: ${TRAIN_DATASET_LIST[@]}"


# ############################################################
# ######## ####  finetuned model  evaluation ######## #### 
# ###########################################################
for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do

    ### random baseline
    random_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_random"

    ##baseline: single use the high quality labels
    label_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_label-filtered"

    ##baseline: completion
    completion_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_completion"


    ## baseline: high-quality labels with diversity
    diversity_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_diversity-filtered"

    ### our method
    filtered_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_filtered" 

    ## full data baseline
    full_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_full"

    ## perplexity baseline
    perplexity_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_perplexity"

    knn_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_knn"

    knn_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_knn"

    less_finetuned_model="models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_less"

    ## model & tokenizer
    declare -A models=(
        ["filtered"]=$filtered_finetuned_model
        ["random"]=$random_finetuned_model
        ["full"]=$full_finetuned_model
        ["label-filtered"]=$label_finetuned_model
        ["base"]=$base_model
        ['diversity-filtered']=$diversity_finetuned_model
        ['completion']=$completion_finetuned_model
        ['perplexity']=$perplexity_finetuned_model
        ['knn']=$knn_finetuned_model
        ['less']=$less_finetuned_model
    )

    for key in "${data_types[@]}"; do


        echo "###### Processing data type:: ${key}"

        # ### MMLU: factual knowledge
        # # ./scripts/eval-new/mmlu.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"
        
        eval_dataset_name='mmlu'
        local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$key

        CUDA_VISIBLE_DEVICES=0 python -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir raw_data/eval/mmlu \
        --save_dir ${local_save_dir} \
        --model_name_or_path ${models[$key]} \
        --tokenizer_name_or_path  ${models[$key]} \
        --eval_batch_size 8  &

        # # ### reasoning
        # # ./scripts/eval-new/gsm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        
        eval_dataset_name='gsm'
        local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$key

        CUDA_VISIBLE_DEVICES=2 python -m eval.gsm.run_eval \
            --data_dir raw_data/eval/gsm/ \
            --max_num_examples 200 \
            --save_dir ${local_save_dir} \
            --model_name_or_path ${models[$key]} \
            --tokenizer_name_or_path ${models[$key]} \
            --n_shot 8 \
            --use_vllm &

        ## BBH: 
        # ./scripts/eval-new/bbh.sh "$train_dataset_name" "$labeling_model" "$base_model"  "${!models[@]}" "${!save_dirs[@]}" "$cuda_devices"
       
        eval_dataset_name='bbh'
        local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$key

        CUDA_VISIBLE_DEVICES=1 python -m eval.bbh.run_eval \
            --data_dir raw_data/eval/bbh \
            --save_dir ${local_save_dir} \
            --model_name_or_path ${models[$key]}  \
            --tokenizer_name_or_path ${models[$key]} \
            --max_num_examples_per_task 40 \
            --use_vllm &




        # # # # ### truthfulness
        # # # # ./scripts/eval-new/truthfulqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        
        eval_dataset_name='truthfulqa'
        local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$key

        CUDA_VISIBLE_DEVICES=3 python -m eval.truthfulqa.run_eval \
            --data_dir raw_data/eval/truthfulqa \
            --save_dir ${local_save_dir} \
            --model_name_or_path ${models[$key]} \
            --tokenizer_name_or_path ${models[$key]} \
            --metrics truth info mc \
            --preset qa \
            --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
            --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
            --eval_batch_size 20 \
            --load_in_8bit &


        # # # # ### multilinguality
        # # # # ./scripts/eval-new/tydiqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        
        eval_dataset_name='tydiqa'
        local_save_dir=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/$key

        CUDA_VISIBLE_DEVICES=4 python -m eval.tydiqa.run_eval \
            --data_dir raw_data/eval/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir ${local_save_dir} \
            --model_name_or_path ${models[$key]} \
            --tokenizer_name_or_path ${models[$key]} \
            --eval_batch_size 20 \
            --load_in_8bit &

        # # ### code evaluation
        # # # ./scripts/eval-new/codex_humaneval.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        # eval_dataset_name='codex_humaneval'

        # CUDA_VISIBLE_DEVICES=5  python -m eval.codex_humaneval.run_eval \
        # --data_file raw_data/eval/codex_humaneval/HumanEval.jsonl.gz \
        # --eval_pass_at_ks 1 5 10 20 \
        # --unbiased_sampling_size_n 20 \
        # --temperature 0.1 \
        # --save_dir ${save_dirs[$key]} \
        # --model ${models[$key]} \
        # --tokenizer ${models[$key]} &

        # # ### need OpenAI API key
        # # # ./scripts/eval-new/alpaca_farm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        # eval_dataset_name='alpaca_farm'

        # CUDA_VISIBLE_DEVICES=6 python -m eval.alpaca_farm.run_eval \
        #     --model_name_or_path ${models[$key]} \
        #     --tokenizer_name_or_path ${models[$key]} \
        #     --save_dir ${save_dirs[$key]} \
        #     --eval_batch_size 20 \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        #     --load_in_8bit &

        wait

    done




    ############################################################
    ######## ####  save the files to containers ######## #### 
    ###########################################################

    files_in_output_dir=$(ls "$cluster_root_path")

    # 打印文件列表
    echo "Files in output directory:"
    echo "$files_in_output_dir"


    ######################################################### 
    # copy to storage account

    AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/evaluation_result/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
    azcopy copy "$cluster_root_path/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive

done

#########################################################
# copy to local

# data_dir="../finetune_result/"
# azcopy copy  "$AZURE_STORAGE_CONTAINER_URL"  "$data_dir" --recursive # --from-to BlobLocal




# azcopy copy  "https://afminternshipuksouth.blob.core.windows.net/jinlong/finetune_result/?sp=racwdlmeop&st=2024-08-16T20:24:00Z&se=2025-03-07T05:24:00Z&sv=2022-11-02&sr=c&sig=%2F3S088nsqvxn%2BtRGen7utkkxJHxgQ5NwlSQlWufI390%3D" ./ --recursive

# azcopy copy  "https://afminternshipuksouth.blob.core.windows.net/jinlong/output/?sp=racwdlmeop&st=2024-08-16T20:24:00Z&se=2025-03-07T05:24:00Z&sv=2022-11-02&sr=c&sig=%2F3S088nsqvxn%2BtRGen7utkkxJHxgQ5NwlSQlWufI390%3D" ./perplexity/ --recursive
