import os
from huggingface_hub import HfApi, upload_folder
from concurrent.futures import ThreadPoolExecutor, as_completed

api = HfApi()

repo_name = "jlpang888/70B-finetune-llama"

labeling_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
# labeling_model="gpt-4o-mini"
# labeling_model="mistralai/Mistral-7B-Instruct-v0.3"

parent_model_path = f"finetune_70B_result/models/{labeling_model}/all_train/meta-llama/Meta-Llama-3-70B"

subfolders = [f.path for f in os.scandir(parent_model_path) if f.is_dir()]

def process_subfolder(folder_path):
    folder_name = os.path.basename(folder_path)
    print(f"Processing folder: {folder_name}")
    return folder_name

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_subfolder, folder) for folder in subfolders]

    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Processed {result}")
        except Exception as e:
            print(f"Error processing folder: {e}")

print(f"Uploading the full model folder: {parent_model_path} to Hugging Face Hub")

api.create_repo(repo_id=repo_name, exist_ok=True)
upload_folder(
    folder_path=parent_model_path,  
    repo_id=repo_name  
)

print("Successfully upload models to HuggingFace!")
