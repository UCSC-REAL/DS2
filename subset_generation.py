import torch 
from collections import Counter
import random
from datasets import load_dataset
import numpy as np
import math
import fire
import matplotlib.pyplot as plt
import seaborn as sns
import os

seed=3
random.seed(seed)
np.random.seed(seed)


def score_curating(reports, score_path, confidence_prob):
    corrupted_samples = [x[0] for x in reports.detection['score_error']]

    curated_sample = []
    curated_sample_scores = []
    for sample in reports.curation['score_curation']:  # (idx, score, confidence)
        if sample[2] >= confidence_prob:  
            curated_sample.append(sample[0])
            curated_sample_scores.append((int(sample[0]), int(sample[1]), round(sample[2],2)))

    print(f"Curated sample size: {len(curated_sample_scores)}")

    # Filter out some cured samples from corrupted instances
    curated_sample_set = set(curated_sample)
    corrupted_samples_total = [x for x in corrupted_samples if x not in curated_sample_set]

    print(f"Corrupted samples total: {len(corrupted_samples_total)}")

    # Change the original scores to the suggested score
    scores = torch.load(score_path + "output_scores_revised.pt")

    for sample_score in curated_sample_scores:
        scores[sample_score[0]] = sample_score[1]
        
    return scores

def extract_data(reports, scores, selected_subset_size, score_category):
    
    # Part 2 (feature-wise): Long-tail Diversity Score Sort
    rare_samples = reports.detection['rare_example'][:len(reports.detection['rare_example']) // 2]
    rare_samples_filtered = np.array(rare_samples)[:, :2]  # Use NumPy for faster operations

    print(f"Size of the remaining samples with high quality: {len(rare_samples_filtered)}")
    scores = np.array(scores)
    score_range = list(range(score_category-1, -1, -1))
    # Cache score indices to avoid repeated searches
    score_indices_cache = {score: np.where(scores == score)[0] for score in score_range}

    # Initialize list to store selected indices
    filtered_indices = []
    # Filter and sort samples by score
    for target_score in score_range:
        if len(filtered_indices) >= selected_subset_size:
            break

        # Get indices of current score
        score_indices = score_indices_cache[target_score]
        available_size = selected_subset_size - len(filtered_indices)

        # Add score indices if enough space, else sort and add top samples
        if available_size > len(score_indices):
            filtered_indices.extend(score_indices.tolist())
        else:
            # Filter and sort samples with the target score by score
            score_samples = rare_samples_filtered[np.isin(rare_samples_filtered[:, 0], score_indices)]
            if len(score_samples) > 0:  
                sorted_samples = score_samples[score_samples[:, 1].argsort()[::-1]][:available_size]
                filtered_indices.extend(sorted_samples[:, 0].astype(int).tolist())

    return filtered_indices


def print_score_heatmap(reports, dataset_name, save_path="figures/"):
    
    data = reports.diagnose['T']
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu")

    plt.title(f'Score transition matrix ({dataset_name})', fontsize=18)
    plt.xlabel('Scores', fontsize=18)
    plt.ylabel('Scores', fontsize=18)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.savefig(save_path + f"{dataset_name}_heatmap.pdf", format="pdf", bbox_inches="tight")
    

def main(
    dataset_name='tulu_300k',
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    selected_subset_size = 10000,
    confidence_prob = 0.5,
    root_score_path = "scoring_output",
    score_curation_path = "score_curation_results",
    output_dir = "selected_data",
    score_category = 6, #Number of score category
    ):

    score_path = root_score_path + f"/{model_name}/{dataset_name}/"
    report_path = score_curation_path + f"/{model_name}/{dataset_name}/{dataset_name}_report.pt"
    output_dir = output_dir + f"/{model_name}/{dataset_name}/"
    
    if dataset_name == 'tulu_300k':
        raw_dataset = load_dataset('jlpang888/tulu_300k')['train'] #300k data
    else:
        raise NotImplementedError

    # score curation reports
    reports = torch.load(report_path)
    print_score_heatmap(reports, dataset_name)

    curated_scores = score_curating(reports, score_path, confidence_prob)
    torch.save(curated_scores, score_path + f"output_scores_curated.pt")
    
    
    ## extract subset
    selected_indices = extract_data(reports, curated_scores, selected_subset_size, score_category)
    
    # Load the dataset and filter out samples by selected indices
    selected_dialogs = raw_dataset.select(selected_indices)
    selected_dialogs.to_json(output_dir + f"ds2_10k_dataset.json")

    print(f"Final dataset saved to {output_dir}ds2_10k_dataset.json")

if __name__ == '__main__':
    fire.Fire(main)
    
    
    