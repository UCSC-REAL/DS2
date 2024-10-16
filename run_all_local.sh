#!/bin/bash

# TRAIN_DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list

TRAIN_DATASET_LIST=('oasst1' 'stanford_alpaca') 

### labeling models
labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"

### base_models 
base_model='meta-llama/Llama-2-7b-hf'

## # data_types used for ablation study, which determines the finetuned model

#  data_types=('filtered-35k' 'filtered-25k' 'filtered-15k' 'filtered-3k')
# data_types=('filtered' 'random' 'label-filtered' 'diversity-filtered' 'full') #baselines

data_types=('full')


# ## system root path
# cluster_root_path="/mnt/azureml/crunch/outputs" ## . for local
# # cluster_root_path="" ## . for local

# mkdir -p $cluster_root_path


for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"
do

    #############################################################
    ######## model finetuning on selected training data ######### 
    #############################################################

    ./scripts/finetune_lora_with_accelerate_new.sh  "$train_dataset_name" "$labeling_model" "$base_model" "$data_types"

    # AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/finetune_result/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
    # azcopy copy "$cluster_root_path/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive

done


# # ############################################################
# # ######## ####  finetuned model  evaluation ######## #### 
# # ###########################################################
# for train_dataset_name in "${TRAIN_DATASET_LIST[@]}"; do

#     ### random baseline
#     random_finetuned_model="output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_random"

#     ##baseline: single use the high quality labels
#     label_finetuned_model="output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_label-filtered"


#     ## baseline: high-quality labels with diversity
#     diversity_finetuned_model="output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_diversity-filtered"

#     ### our method
#     filtered_finetuned_model="output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_filtered" 

#     ## full data baseline
#     full_finetuned_model="output/models/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_full"

#     ## model & tokenizer
#     declare -A models=(
#         ["filtered"]=$filtered_finetuned_model
#         # ["random"]=$random_finetuned_model
#         # ["full"]=$full_finetuned_model
#         # ["label"]=$label_finetuned_model
#         # ["base"]=$base_model
#         # ['diversity']=$diversity_finetuned_model
#     )


#     declare -A save_dirs=(
#         ["filtered"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/filtered
#         ["random"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/random
#         ["full"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/full
#         ["label"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/label
#         ["base"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/base
#         ["diversity"]=${cluster_root_path}/results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/diversity
#     )



#     # #  
#     # model_declaration=$(declare -p models)
#     # save_dirs_declaration=$(declare -p save_dirs)
#     # cuda_devices_declaration=$(declare -p cuda_devices)


#     for key in "${!models[@]}"; do

#         # ### MMLU: factual knowledge
#         # ./scripts/eval-new/mmlu.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"
        
#         eval_dataset_name='mmlu'

#         CUDA_VISIBLE_DEVICES=0 python -m eval.mmlu.run_eval \
#         --ntrain 0 \
#         --data_dir raw_data/eval/mmlu \
#         --save_dir ${save_dirs[$key]} \
#         --model_name_or_path ${models[$key]} \
#         --tokenizer_name_or_path  ${models[$key]} \
#         --eval_batch_size 16  &

#         ### BBH: 
#         # ./scripts/eval-new/bbh.sh "$train_dataset_name" "$labeling_model" "$base_model"  "${!models[@]}" "${!save_dirs[@]}" "$cuda_devices"
       
#         eval_dataset_name='bbh'

#         CUDA_VISIBLE_DEVICES=1 python -m eval.bbh.run_eval \
#             --data_dir raw_data/eval/bbh \
#             --save_dir ${save_dirs[$key]} \
#             --model ${models[$key]}  \
#             --tokenizer ${models[$key]} \
#             --max_num_examples_per_task 40 \
#             --use_vllm &



#         # # ### reasoning
#         # # ./scripts/eval-new/gsm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        
#         eval_dataset_name='gsm'

#         CUDA_VISIBLE_DEVICES=2 python -m eval.gsm.run_eval \
#             --data_dir raw_data/eval/gsm/ \
#             --max_num_examples 200 \
#             --save_dir ${save_dirs[$key]} \
#             --model ${models[$key]} \
#             --tokenizer ${models[$key]} \
#             --n_shot 8 &


#         # # # ### truthfulness
#         # # # ./scripts/eval-new/truthfulqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
        
#         eval_dataset_name='truthfulqa'
        
#         CUDA_VISIBLE_DEVICES=3 python -m eval.truthfulqa.run_eval \
#             --data_dir raw_data/eval/truthfulqa \
#             --save_dir ${save_dirs[$key]} \
#             --model_name_or_path ${models[$key]} \
#             --tokenizer_name_or_path ${models[$key]} \
#             --metrics truth info mc \
#             --preset qa \
#             --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
#             --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
#             --eval_batch_size 20 \
#             --load_in_8bit &


#         # # # ### multilinguality
#         # # # ./scripts/eval-new/tydiqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
#         eval_dataset_name='tydiqa'

#         CUDA_VISIBLE_DEVICES=4 python -m eval.tydiqa.run_eval \
#             --data_dir raw_data/eval/tydiqa/ \
#             --n_shot 1 \
#             --max_num_examples_per_lang 100 \
#             --max_context_length 512 \
#             --save_dir ${save_dirs[$key]} \
#             --model ${models[$key]} \
#             --tokenizer ${models[$key]} \
#             --eval_batch_size 20 \
#             --load_in_8bit &

#         # # ### code evaluation
#         # # # ./scripts/eval-new/codex_humaneval.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
#         # eval_dataset_name='codex_humaneval'

#         # CUDA_VISIBLE_DEVICES=5  python -m eval.codex_humaneval.run_eval \
#         # --data_file raw_data/eval/codex_humaneval/HumanEval.jsonl.gz \
#         # --eval_pass_at_ks 1 5 10 20 \
#         # --unbiased_sampling_size_n 20 \
#         # --temperature 0.1 \
#         # --save_dir ${save_dirs[$key]} \
#         # --model ${models[$key]} \
#         # --tokenizer ${models[$key]} &

#         # # ### need OpenAI API key
#         # # # ./scripts/eval-new/alpaca_farm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$model_declaration" "$save_dirs_declaration" "$cuda_devices_declaration"
#         # eval_dataset_name='alpaca_farm'

#         # CUDA_VISIBLE_DEVICES=6 python -m eval.alpaca_farm.run_eval \
#         #     --model_name_or_path ${models[$key]} \
#         #     --tokenizer_name_or_path ${models[$key]} \
#         #     --save_dir ${save_dirs[$key]} \
#         #     --eval_batch_size 20 \
#         #     --use_chat_format \
#         #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#         #     --load_in_8bit &

#         wait

#     done




#     ############################################################
#     ######## ####  save the files to containers ######## #### 
#     ###########################################################

#     files_in_output_dir=$(ls "$cluster_root_path")

#     #   
#     echo "Files in output directory:"
#     echo "$files_in_output_dir"


#     ######################################################### 
#     # copy to storage account

#     AZURE_STORAGE_CONTAINER_URL="https://afminternshipuksouth.blob.core.windows.net/jinlong/finetune_result/?sp=racwdlmeop&st=2024-08-24T00:58:39Z&se=2025-04-03T08:58:39Z&sv=2022-11-02&sr=c&sig=rbf41XiVlLJw76zeillA%2FRMAjgGMo2lQHO3m3RW5Ho8%3D"
#     azcopy copy "$cluster_root_path/*" "$AZURE_STORAGE_CONTAINER_URL" --recursive

# done

# #########################################################
# # copy to local

# data_dir="../finetune_result/"
# azcopy copy  "$AZURE_STORAGE_CONTAINER_URL"  "$data_dir" --recursive # --from-to BlobLocal




# azcopy copy  "https://afminternshipuksouth.blob.core.windows.net/jinlong/finetune_result/?sp=racwdlmeop&st=2024-08-16T20:24:00Z&se=2025-03-07T05:24:00Z&sv=2022-11-02&sr=c&sig=%2F3S088nsqvxn%2BtRGen7utkkxJHxgQ5NwlSQlWufI390%3D" ../finetune_result/ --recursive

