import torch 
import random
import numpy as np
from datasets import load_dataset
import fire 



def main(
        dataset_name='flan_v2',
        model_name='gemma',
        datasize=3000,
        threshold=300,
    ):  

    ## label curation reports

    report_file = f"./results/tulu_{dataset_name}/tulu_{dataset_name}_report.pt"
    reports = torch.load(report_file)


    '''Part 1 (label-wise): label curation'''


    ##  samples that can be cured
    cured_samples = []
    cured_sample_labels = []
    for sample in reports.curation['label_curation']: ##(idx, label, confidence)
        if sample[2] >= 0.75: #confidence prob;0.75
            cured_samples.append(sample[0])
            cured_sample_labels.append((sample[0], sample[1]))

    ### choose the data index that needed to be remove
    corrupted_samples = [x[0] for x in reports.detection['label_error']]

    #filter out some cured samples from corrupted instances
    cured_samples_set = set(cured_samples)
    corrupted_samples_total = [x for x in corrupted_samples if x not in cured_samples_set]


    # change the original labels to the suggested label
    root_path = f'/home/azureuser/cloudfiles/code/Users/jinlong.pang/LADR_LLM_alignment_data_refinement/labeling/data/{model_name}/'
    labels = torch.load(root_path + f"{dataset_name}/output_labels_revised.pt")

    for sample_label in cured_sample_labels:
        labels[sample_label[0]] = sample_label[1]
    print(f"Dataset {dataset_name} \n Label size: {len(labels)}")

    ## select high-quality samples based on the quality labels
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 打印结果
    for label, count in zip(unique_labels, counts):
        print(f"Label: {label}, Count: {count}")
    ###filter out the low-quality samples

    low_quality_label_idx = []
    for idx, label in enumerate(labels):
        if label<3:
            low_quality_label_idx.append(idx)
        # elif label == 3 and random.random() >= 0.5:
        #     low_quality_label_idx.append(idx)




    label_wise_filter_out_samples = set(low_quality_label_idx + corrupted_samples_total)



    '''Part-2 (feature-wise): handle the rare example'''

    rare_samples = reports.detection['rare_example'][:len(reports.detection['rare_example'])//2]
    rare_samples_filtered = [[sample[0], sample[1]] for sample in rare_samples if sample[0] not in set(label_wise_filter_out_samples)] 
    long_tail_scores = np.array(rare_samples_filtered)[:,1]


    bins = np.arange(0, max(long_tail_scores)+0.01, 0.01) # 定义区间边界

    # 计算每个区间的计数
    counts, _ = np.histogram(long_tail_scores, bins)


    threshold = threshold   ## the random 

    remaining_samples_indices = []

    for i in range(len(bins) - 1):
        indices_in_bin = np.where((long_tail_scores >= bins[i]) & (long_tail_scores < bins[i+1]))[0]
        
        if counts[i] > threshold:
            indices_in_bin = random.sample(list(indices_in_bin), threshold)
        
        remaining_samples_indices.extend(indices_in_bin)


    remaining_samples_idx = np.array(rare_samples_filtered, dtype=int)[remaining_samples_indices, 0]

    # long_tail_scores_filtered = long_tail_scores[remaining_samples_idx]
    # long_tail_scores_filtered = np.array(rare_samples_filtered)[remaining_samples_indices, 1]

    # 打印剩余的样本及其原始索引
    print("Size of the filtered dataset:", len(remaining_samples_idx))

    '''filter out the corrupted samples and reconstruct the dataset'''

    data_path = './data_refine/tulu_split_parquet/'
    dataset_path = data_path + f"{dataset_name}.parquet"

    data = load_dataset('parquet', data_files=dataset_path)
    filtered_dialogs = data['train'].select(remaining_samples_idx)

    filtered_labels = np.array(labels)[remaining_samples_idx].tolist()
    torch.save(filtered_labels, root_path + f"{dataset_name}/filtered_output_labels.pt")



    assert len(filtered_dialogs) == len(filtered_labels)

    # output_path = root_path + f"{dataset_name}/filtered_flan_v2.parquet"
    # filtered_dialogs.to_parquet(output_path)

    output_json_path = root_path + f"{dataset_name}/filtered_dataset.json"  ## the json form is for funetunning
    filtered_dialogs.to_json(output_json_path)


    ####################################################################################################################################################################################
    '''random baseline'''

    #full size

    #selected size

    full_data_size = len(data['train'])
    data_size = len(filtered_labels)
    print(f"full data size: {full_data_size}; random selected size: {data_size}!!!")

    random_samples_idx = random.sample(list(range(full_data_size)), data_size)

    filtered_dialogs = data['train'].select(random_samples_idx)


    output_json_path = root_path + f"{dataset_name}/random_dataset.json"  ## the json form is for funetunning
    filtered_dialogs.to_json(output_json_path)

    print(f"Store the file to path {root_path + dataset_name}")

if __name__ == "__main__":
    fire.Fire(main)