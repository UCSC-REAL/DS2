import os
import json
import pandas as pd
import fire
'''
{baseline_tag: 
    { 
        eval_dataset:{ }
                        
    }
}
'''

def main(
        root_result_path = 'results',
        train_dataset='all_train',
        baseline_tag = 'filtered', 
        eval_dataset = 'tydiqa'
        ):

    all_results = {}  

    ### full results print ####

    # base_model ='meta-llama/Llama-2-7b-hf'
    # base_model ="meta-llama/Meta-Llama-3.1-8B" 
    # base_model='mistralai/Mistral-7B-v0.3'


    # labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
    # labeling_model="gpt-4o-mini"
    # labeling_model='mistralai/Mistral-7B-Instruct-v0.3'


    # baseline_tags=['base','random',  'completion', 'perplexity',  'knn',  'label-filtered', 'diversity-filtered', 'less', 'filtered', 'full'] #baselines
    baseline_tags=[baseline_tag] #baselines

    # eval_dataset_lists = ['mmlu', 'truthfulqa', 'gsm', 'bbh', 'tydiqa']
    eval_dataset_lists = [eval_dataset]

    # Load results from JSON files
    for tag in baseline_tags:
        baseline_results = {}
        for eval_dataset in eval_dataset_lists:
            path = root_result_path + f'{tag}/metrics.json'
            try:
                with open(path, 'r') as f:
                    json_file = json.load(f)
                baseline_results[eval_dataset] = json_file
            except FileNotFoundError:
                print(f"Failed to find the file at {path}")
                baseline_results[eval_dataset] = None

        all_results[tag] = baseline_results

    # Extract relevant metrics and store in a DataFrame
    cur_results = {}
    for tag in baseline_tags:
        baseline_result = []
        for eval_dataset in eval_dataset_lists:
            if all_results[tag][eval_dataset] is None:
                value = 0
            else:
                if eval_dataset == 'mmlu':
                    value = round(all_results[tag][eval_dataset]['average_acc'] * 100, 1)
                elif eval_dataset == 'bbh':
                    value = round(all_results[tag][eval_dataset]['average_exact_match']* 100, 1)
                elif eval_dataset == 'tydiqa':
                    value = round(all_results[tag][eval_dataset]['average']['f1'], 1)
                elif eval_dataset == 'gsm':
                    value = round(all_results[tag][eval_dataset]['exact_match']* 100, 1)
                elif eval_dataset == 'truthfulqa':
                    # value = round(all_results[tag][eval_dataset]["truth-info acc"]* 100, 1)
                    value = round(all_results[tag][eval_dataset]["MC2"]* 100, 1)
                else:
                    print("unknown eval dat·aset!")


            baseline_result.append(value)
        cur_results[tag] = baseline_result

    # Convert cur_results to pandas DataFrame
    df_results = pd.DataFrame.from_dict(cur_results, orient='index', columns=eval_dataset_lists)

    # Calculate the average accuracy for each baseline
    df_results['average acc'] = df_results.mean(axis=1).round(1)
    # print(f"### base_model: {base_model}")
    # print(f"### labeling_model: {labeling_model}")


    # Ensure full display of the DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set display width
    pd.set_option('display.max_colwidth', None)  # Set max column width

    print(df_results)


if __name__ == '__main__':
    fire.Fire(main)
