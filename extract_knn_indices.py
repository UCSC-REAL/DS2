from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
import random
import numpy as np
from tqdm import tqdm
import os

# Set random seed for reproducibility
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to calculate cosine distance and return KNN indices (excluding the sample itself)
def cosDistance(sample_embedding, all_embeddings, k_near=10):
    sample_embedding = sample_embedding.to(device)
    all_embeddings = all_embeddings.to(device)

    # Compute cosine similarity and convert to cosine distance
    similarity_vector = torch.matmul(all_embeddings, sample_embedding)
    distance_vector = 1.0 - similarity_vector

    # Get top-k nearest samples, including self at index 0
    distance_vector, topk_indices = torch.topk(distance_vector, k=k_near+1, dim=0, largest=False)

    # Exclude the sample itself from KNN results
    if topk_indices[0] == torch.arange(all_embeddings.size(0))[0]:
        topk_indices = topk_indices[1:]
        distance_vector = distance_vector[1:]

    return distance_vector, topk_indices

# Mean pooling function to obtain sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Process dialog messages to convert into a text format
def process_dialog(dialog):
    conversation = ""
    for message in dialog['messages']:
        conversation += f"### {message['role']}: {message['content']}\n"
    return {"features": conversation}

# Generate embeddings for text
def embed_text(batch):
    encoded_inputs = tokenizer(batch['features'], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_outputs = model(**encoded_inputs)
    sentence_embeddings = mean_pooling(model_outputs, encoded_inputs['attention_mask'])
    embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    batch['embeddings'] = embeddings.cpu().numpy().tolist()
    return batch

##############################################################################################################################

dataset_name = "all_train"

# Load or generate embeddings if they don't already exist
if not os.path.exists(f"{dataset_name}_embeddings.parquet"):
    
    print("calculating the embedding...")
    
    data = load_dataset('json', data_files=f"full_dataset.json")
    data['train'] = data['train'].map(process_dialog, batched=False)

    embedding_model_name = "BAAI/bge-large-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name).to(device)
    model = torch.nn.DataParallel(model)

    data['train'] = data['train'].map(embed_text, batched=True, batch_size=2048)
    data['train'].to_parquet(f'{dataset_name}_embeddings.parquet')
    print(f"Embeddings saved to {dataset_name}_embeddings.parquet")
else:
    print("load existing embedding file")

#########################################################################################################################

# Load generated embeddings dataset
embedding_dataset = load_dataset('parquet', data_files=f'{dataset_name}_embeddings.parquet')['train']

# Get all embeddings data
all_embeddings = torch.tensor(embedding_dataset['embeddings']).to(device)


knn_indices = []
distances = []

# overall_len_sorted_indices = torch.load("overall_len_indices_sorted.pt")

# selected_short_example_indices = overall_len_sorted_indices ##check the len
# embedding_subset = embedding_dataset.select(selected_short_example_indices)

for sample_idx, sample in enumerate(tqdm(embedding_dataset, desc=f"Processing 2-NN samples")):
    sample_embedding = torch.tensor(sample['embeddings']).to(device)

    # Calculate cosine distance for the sample and retrieve KNN indices
    
    distances_per_sample, topk_indices = cosDistance(sample_embedding, all_embeddings, k_near=2)
    distances.append(round(distances_per_sample.mean().item(), 5))
    knn_indices.append(topk_indices.cpu().numpy().tolist())


# torch.save(knn_indices, "all_train_knn_indices.pt")
torch.save(distances, "all_train_knn_embedding_distances.pt")