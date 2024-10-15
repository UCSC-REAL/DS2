from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm

random.seed(42)

def cosDistance(sample_embedding, selected_embeddings, k_near=10):
    # 归一化嵌入
    sample_embedding = F.normalize(sample_embedding, dim=0)
    selected_embeddings = F.normalize(selected_embeddings, dim=1)

    # 计算样本与样本池中所有样本的余弦相似度
    similarity_vector = torch.matmul(selected_embeddings, sample_embedding)

    distance_vector = 1.0 - similarity_vector
    if selected_embeddings.size(0) > k_near:
        distance_vector, _ = torch.topk(distance_vector, k=k_near, dim=0)

    mean_distance = distance_vector.mean()

    return mean_distance

# 加载嵌入数据

# 加载标签数据
dataset_name = 'flan_v2'
model_name = "meta/llama-3.1-8b-instruct"
root_path = f"data/{model_name}/{dataset_name}/"
labels = torch.load(root_path + "output_labels_revised.pt")

embedding_dataset = load_dataset("parquet", data_files=f'{dataset_name}_embeddings.parquet')['train']

high_quality_label_idx_5 = [idx for idx, label in enumerate(labels) if label == 5]
high_quality_label_idx_4 = [idx for idx, label in enumerate(labels) if label == 4]

embedding_dataset_5 = embedding_dataset.select(high_quality_label_idx_5)
embedding_dataset_4 = embedding_dataset.select(high_quality_label_idx_4)

tot_data_size = 6600
threshold = 0.45
k_near=10
# 初始化空的 tensor 来存储选择的样本嵌入
selected_embeddings = None
selected_samples = []
distances = []
# 使用 tqdm 包裹 for 循环来显示进度
for sample in tqdm(embedding_dataset_5, desc="Processing high-quality samples (label=5)"):
    sample_embedding = torch.tensor(sample['embeddings'])
    if selected_embeddings is None:
        selected_embeddings = sample_embedding.unsqueeze(0)
        selected_samples.append(sample)
        continue
    embed_distance = cosDistance(sample_embedding, selected_embeddings, k_near=k_near)
    distances.append(embed_distance)
    if embed_distance < threshold:
        selected_embeddings = torch.cat((selected_embeddings, sample_embedding.unsqueeze(0)), dim=0)
        selected_samples.append(sample)
    
    if len(selected_samples) == tot_data_size:
        break

print(f"current selected data size: {len(selected_samples)}; still need select sample from 4-rated sampels")
# print(f"embed distance: {distances}")
print(f"mean embed distance: {np.mean(distances)}")

print(f"min-max embed distance: ({min(distances)}, {max(distances)})")


if len(selected_samples) < tot_data_size:
    for sample in tqdm(embedding_dataset_4, desc="Processing high-quality samples (label=4)"):
        sample_embedding = torch.tensor(sample['embeddings'])
        embed_distance = cosDistance(sample_embedding, selected_embeddings, k_near=k_near)

        if embed_distance < threshold:
            selected_embeddings = torch.cat((selected_embeddings, sample_embedding.unsqueeze(0)), dim=0)
            selected_samples.append(sample)
        
        if len(selected_samples) == tot_data_size:
            break

# 将最终选择的样本列表转换为 Dataset
selected_dataset = Dataset.from_dict({col: [s[col] for s in selected_samples] for col in selected_samples[0].keys()})

print(f"selected data size: {len(selected_dataset)}")


selected_dataset.to_json(root_path + f'diversity-filtered_dataset.json', orient='records', lines=True)

print(f"Dataset saved to {root_path}diversity-filtered_dataset.json")
