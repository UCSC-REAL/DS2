import os
from huggingface_hub import HfApi, upload_folder
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 Hugging Face Hub 的 API 实例
api = HfApi()

# 仓库名称（整个模型将上传到这个仓库）
repo_name = "jlpang888/70B-finetune-llama"

labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# labeling_model="gpt-4o-mini"
# labeling_model="mistralai/Mistral-7B-Instruct-v0.3"

# 定义模型文件路径（包含多个子文件夹，每个子文件夹是模型的一部分）
parent_model_path = f"finetune_70B_result/models/{labeling_model}/all_train/meta-llama/Meta-Llama-3-70B"

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(parent_model_path) if f.is_dir()]

# 定义并行处理的子文件夹上传任务
def process_subfolder(folder_path):
    folder_name = os.path.basename(folder_path)
    # 可以在这里对每个子文件夹做一些处理，或者准备它们上传
    print(f"Processing folder: {folder_name}")
    # 返回处理结果（这里不上传单个子文件夹，只是并行处理后，最终整合上传）
    return folder_name

# 使用线程池并行处理所有子文件夹
with ThreadPoolExecutor() as executor:
    # 提交任务给线程池
    futures = [executor.submit(process_subfolder, folder) for folder in subfolders]

    # 处理并行执行的结果
    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Processed {result}")
        except Exception as e:
            print(f"Error processing folder: {e}")

# 并行处理完子文件夹后，上传整个父文件夹到 Hugging Face Hub
print(f"Uploading the full model folder: {parent_model_path} to Hugging Face Hub")

# 上传整个父文件夹到 Hugging Face Hub 仓库
api.create_repo(repo_id=repo_name, exist_ok=True)
upload_folder(
    folder_path=parent_model_path,  # 上传整个父文件夹
    repo_id=repo_name  # Hugging Face 仓库名称
)

print("完整模型文件夹已成功上传到 Hugging Face Hub")
