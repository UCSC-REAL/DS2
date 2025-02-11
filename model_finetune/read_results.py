import os
import json
import pandas as pd
import fire


def main(
        root_result_path = 'results',
        raw_dataset='tulu_300k',
        base_model = "meta-llama/Meta-Llama-3.1-8B",
        rating_model='mistralai/Mistral-7B-Instruct-v0.3',
        baseline_tag = 'ds2_10k', 
        ):

    all_results = {}  
    baseline_tags=[baseline_tag] #baselines
    eval_dataset_lists = ['mmlu', 'truthfulqa', 'gsm', 'bbh', 'tydiqa']

    # Load results from JSON files
    for tag in baseline_tags:
        baseline_results = {}
        for eval_dataset in eval_dataset_lists:
            path = root_result_path + f'/{rating_model}/{raw_dataset}/{eval_dataset}/{base_model}/{tag}/metrics.json'
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
                    value = round(all_results[tag][eval_dataset]["truth-info acc"]* 100, 1)
                    # value = round(all_results[tag][eval_dataset]["MC2"]* 100, 1)
                else:
                    print("unknown eval dat·aset!")


            baseline_result.append(value)
        cur_results[tag] = baseline_result

    # Convert cur_results to pandas DataFrame
    df_results = pd.DataFrame.from_dict(cur_results, orient='index', columns=eval_dataset_lists)

    # Calculate the average accuracy for each baseline
    df_results['average acc'] = df_results.mean(axis=1).round(1)


    # Ensure full display of the DataFrame
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', 1000)  
    pd.set_option('display.max_colwidth', None) 

    print(df_results)


if __name__ == '__main__':
    fire.Fire(main)
