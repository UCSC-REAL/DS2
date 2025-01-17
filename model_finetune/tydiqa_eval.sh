export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

start_time=$(date +%s)




# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global-curve-positive-new1" "filtered-cured-50k-iter-split-token-ranking-sample")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global-curve-positive-new1")
# Train_DATASET_LIST=("filtered-cured-50k-full-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-random-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-new1")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-test")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-new1-shuffle" "filtered-cured-50k-active-split-global-positive-new2-shuffle")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-fixed-positive-shuffle")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-curve-positive-reverse")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-curve-positive-reverse-loss-sum")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-sample-positive-reverse")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-curve-smooth-positive-reverse")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-curve-positive-new1-fixed-base-loss")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-token-ranking-sample")

#### TODO evalution model #### 
# "filtered-cured-50k-active-split-token_ranking_sample" "filtered-cured-50k-active-split-token_ranking_sample-fixed-base-loss"
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss" "filtered-cured-50k-active-split-global-curve-positive-using-warmup" "filtered-cured-50k-active-split-global-curve-positive-fixed-base-loss-using-warmup")

# train_dataset_name="filtered-cured-50k-active-split-token_ranking_sample"
# train_dataset_name="filtered-cured-50k-active-split-token_ranking_sample-fixed-base-loss"
# train_dataset_name="filtered-cured-50k-active-split-global-half-positive-fixed-base-loss"
# train_dataset_name="filtered-cured-50k-active-split-global-curve-positive-using-warmup"
# train_dataset_name="filtered-cured-50k-active-split-global-curve-positive-fixed-base-loss-using-warmup"

Train_DATASET_LIST=("base")

# base_model=meta-llama/Llama-3.2-3B
base_model="meta-llama/Llama-3.1-8B-Instruct"

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-curve-positive-fixed-base-loss-using-warmup")

data_prop=0.6 # 0.3
eval_dataset_name='tydiqa'

model_path="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/data_prop_${data_prop}"
MODEL=hf #hf


for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
    echo "##### train_dataset_name: ${train_dataset_name}"

    # model_tags=("${train_dataset_name}_4")
    model_tags=("${train_dataset_name}")
    # model_tags=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")


    # for model_tag in ${model_tags[@]}; do
    for idx in ${!model_tags[@]}; do

        model_tag=${model_tags[$idx]} 

        if [[ $model_tag == 'base' ]]; then
            pretrained_model=$base_model
        else
            pretrained_model=${model_path}/lora_merged_${model_tag}
        fi

        echo "######## evaluation model: ${model_tag} #############"

        OUTPUT_PATH=token_selection_results/${data_prop}/${model_tag}

        mkdir -p $OUTPUT_PATH



        CUDA_VISIBLE_DEVICES=$idx python -m eval.tydiqa.run_eval \
            --data_dir raw_data/eval/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir $OUTPUT_PATH \
            --model_name_or_path $pretrained_model \
            --tokenizer_name_or_path $pretrained_model \
            --eval_batch_size 70 &
            # --load_in_8bit &

        
    done
    wait
done 

OUTPUT_PATH=token_selection_results/${data_prop}/


for train_dataset_name in "${Train_DATASET_LIST[@]}"; do

    # model_tags=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    model_tags=("${train_dataset_name}")

    for model_tag in "${model_tags[@]}"; do        
        # echo "###### Processing training dataset :: ${train_dataset_name}"
        # echo "###### Processing model_tag :: ${model_tag}"
        python3 read_results_token.py --root_result_path $OUTPUT_PATH --eval_dataset $eval_dataset_name --train_dataset $train_dataset_name --baseline_tag $model_tag 

    done

done



echo "all experiments finished!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"
