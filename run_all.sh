#!/bin/bash

# TRAIN_DATASET_LIST=('flan_v2' 'oasst1' 'wizardlm' 'dolly' 'stanford_alpaca') # full data list
train_dataset_name='dolly'

### labeling models
labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"

### base_models 
base_model='meta-llama/Llama-2-7b-hf'

## # data_types used for ablation study
#  data_types=('filtered-35k' 'filtered-25k' 'filtered-15k' 'filtered-3k')
# data_types=('filtered' 'random' 'label-filtered' 'diversity-filtered' 'full') #baselines

data_types=('full')


#############################################################
######## model finetuning on selected training data ######### 
#############################################################

./scripts/finetune_lora_with_accelerate_new.sh  "$train_dataset_name" "$labeling_model" "$base_model" "$data_types"





############################################################
######## ####  finetuned model  evaluation ######## #### 
###########################################################
 

# ### random baseline
# random_finetuned_model="output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_random"

# ##baseline: single use the high quality labels
# label_finetuned_model="output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_label-filtered"


# ## baseline: high-quality labels with diversity
# diversity_finetuned_model="output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_diversity-filtered"

# ### our method
# filtered_finetuned_model="output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_filtered" 

# ## full data baseline
# full_finetuned_model="output/${labeling_model}/${train_dataset_name}/${base_model}/lora_merged_full"

# ## model & tokenizer
# declare -A models
# models=(
#     ["filtered"]=$filtered_finetuned_model
#     ["random"]=$random_finetuned_model
#     # ["full"]=$full_finetuned_model
#     # ["label"]=$label_finetuned_model
#     # ["base"]=$base_model
#     # ['diversity']=$diversity_finetuned_model
# )

# declare -A save_dirs
# save_dirs=(
#     ["filtered"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/filtered
#     ["random"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/random
#     ["full"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/full
#     ["label"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/label
#     ["base"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/base
#     ["diversity"]=results/${labeling_model}/${train_dataset_name}/${eval_dataset_name}/${base_model}/diversity
# )

# # CUDA gpu devices
# declare -A cuda_devices
# cuda_devices=(
#     ["filtered"]=2
#     ["random"]=3
#     ["full"]=2
#     ["label"]=3
#     ["base"]=3
#     ["diversity"]=2
# )


# ### factual knowledge
# ./scripts/eval-new/mmlu.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"

# ### reasoning
# ./scripts/eval-new/bbh.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"

# ### reasoning
# ./scripts/eval-new/gsm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"

# ### truthfulness
# ./scripts/eval-new/truthfulqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"

# ### multilinguality
# ./scripts/eval-new/tydiqa.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"


# ### code evaluation
# # ./scripts/eval-new/codex_humaneval.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"

# ### need OpenAI API key
# # ./scripts/eval-new/alpaca_farm.sh "$train_dataset_name" "$labeling_model" "$base_model" "$models" "$save_dirs" "$cuda_devices"