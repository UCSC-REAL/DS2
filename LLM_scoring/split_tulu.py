

from datasets import load_dataset, DatasetDict
import os

# 加载数据集
dataset_name = "allenai/tulu-v2-sft-mixture"
data = load_dataset(dataset_name)

# 假设我们处理的是训练集
train_dataset = data['train']

# 定义函数来按键分割数据集
def split_by_key(dataset, key):
    unique_values = dataset.unique(key)
    split_datasets = {}
    for value in unique_values:
        split_datasets[value] = dataset.filter(lambda x: x[key] == value)
    return DatasetDict(split_datasets)

# 使用函数按 'dataset' 列分割数据集
split_datasets = split_by_key(train_dataset, 'dataset')

# 创建存储文件的目录
output_dir = "LLL_scoring/tulu_split_json"
os.makedirs(output_dir, exist_ok=True)

# 保存每个子集为单独的 Parquet 文件
for key, sub_dataset in split_datasets.items():
    file_path = os.path.join(output_dir, f"{key}.json")
    sub_dataset.to_json(file_path)
    print(f"Saved subset for {key} to {file_path}")

print("All subsets have been saved as Json files.")
