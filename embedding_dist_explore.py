from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
import random
import numpy as np
from tqdm import tqdm
import os
import gc
import pandas as pd

seed =3
random.seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def cosDistance(sample_embedding, selected_embeddings, k_near=0):
    similarity_vector = torch.matmul(selected_embeddings, sample_embedding)
    distance_vector = 1.0 - similarity_vector
    if selected_embeddings.size(0) > k_near:
        distance_vector, _ = torch.topk(distance_vector, k=k_near, dim=0)
    mean_distance = distance_vector.mean()
    return mean_distance

def mean_pooling(model_output, attention_mask, layer_idx=-1):
    # # layer_idx = len(model_output.hidden_states) //2
    # layer_idx = -5
    # print(f"current hidden state layer idx: {layer_idx}")
    token_embeddings = model_output.hidden_states[layer_idx]
    # import pdb;pdb.set_trace()
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def process_dialog(dialog):
    conversation = ""
    for message in dialog['messages']:
        conversation += f" #{message['role']}: {message['content']}\n" ##Please check the correctness of the following conversation:\n
    return {"features": conversation}

def embed_text(batch, layer_idx):
    encoded_inputs = tokenizer(batch['features'], padding=True, truncation=True, return_tensors="pt").to(device)
    if 'token_type_ids' in encoded_inputs:
        del encoded_inputs['token_type_ids']
        
    with torch.no_grad():
        model_outputs = model(**encoded_inputs)
    sentence_embeddings = mean_pooling(model_outputs, encoded_inputs['attention_mask'].to(device), layer_idx)
    embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    batch['embeddings'] = embeddings.cpu().numpy().tolist()
    return batch

##############################################################################################################################
samples = [
    {
        "messages": [
            {"role": "user", "content": "please tell me the answer of below math question '1+1=?'"},
            {"role": "assistant", "content": "answer:2"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "please tell me the answer of below math question '1+1=?'"},
            {"role": "assistant", "content": "answer:3"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "please tell me the answer of below math question '10+10=?'"},
            {"role": "assistant", "content": "answer:20"}
        ]
    },
    {
        "messages": [
            {'role': 'user', 'content': 'Calculate 15% of 500.\n\n'},
            {'role': 'assistant', 'content': '75'}
        ]
    },
    {
        "messages": [
            {'role': 'user', 'content': 'Calculate 50% of 300.'},
            {'role': 'assistant', 'content': '50% of 300 is 150.'}
        ]
    },
        {
        "messages": [
            {"role":"user","content":"Task: Generate an open-ended story based on the following prompts.\nInput: Prompts: False accusation, Tower, Mallet\nOutput:"},
            {"role":"assistant","content":"Once upon a time, there was a small village nestled in a meadow at the foot of a tall tower. The tower was used by the local villagers for refuge during times of trouble, and with its high walls, it was secure and impenetrable. \n\nOne day, a stranger appeared in the village without warning and accused one of the villagers of a serious crime. Perplexed and outraged, the villagers voted to conduct a trial. The stranger declared that the accused must climb to the top of the tower and if they could make it to the top with a mallet, then they would be found innocent. If they failed, then the stranger would claim that they had fled and were therefore guilty. The accused was terrified, as none had ever made it to the top before. But to their surprise, they succeeded with the aid of the mallet, and the stranger was forced to leave in shame. \nFrom then on, the villagers held a festival once a year to mark the occasion and celebrate the strength of the accused."}]
    }
]

embedding_models = [ "meta-llama/Llama-3.1-8B-Instruct",
                    "facebook/galactica-6.7b", 
                    "BAAI/bge-large-en-v1.5", 
                    "sentence-transformers/all-mpnet-base-v2", 
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "Qwen/Qwen2.5-Math-7B-Instruct",
                    "Qwen/Qwen2.5-Math-1.5B-Instruct",
                    "Qwen/Qwen2.5-14B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "tbs17/MathBERT",
                    "deepseek-ai/deepseek-math-7b-base",
                    ]



check_last_hidden_state = True ###calculate the last hidden state


distances = {}
for embedding_model_name in embedding_models:
    print(f"### embedding model: {embedding_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(
        embedding_model_name, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )


    total_layers = model.config.num_hidden_layers + 1
    print(f"Total layers including embedding layer: {total_layers}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # example_pairs = [(0,1), (1,2), (3,4)]

    dist_layers = []
    if not check_last_hidden_state:
        for layer_idx in range(total_layers):
            embeddings = []
            for sample in samples:
                dialog = process_dialog(sample)
                embeddings.append(embed_text(dialog, layer_idx))
            
            dist = cosDistance(torch.Tensor(embeddings[0]['embeddings'][0]), torch.Tensor(embeddings[1]['embeddings'][0]))
            dist_layers.append(round(dist.item(), 5))

    else:
        layer_idx = -1
        embeddings = []
        for sample in samples:
            dialog = process_dialog(sample)
            embeddings.append(embed_text(dialog, layer_idx))
        
        sample_idx1, sample_idx2 = 0, 1 ##selected sample
        
        emb1 = torch.Tensor(embeddings[sample_idx1]['embeddings'][0])
        emb2 = torch.Tensor(embeddings[sample_idx2]['embeddings'][0])
        dist = cosDistance(emb1, emb2)
        dist_layers.append(round(dist.item(), 5))    
        
    distances[os.path.basename(embedding_model_name)] = dist_layers
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

max_layers = max(len(layers) for layers in distances.values())

distances_df = pd.DataFrame.from_dict(distances, orient='index')
distances_df = distances_df.reindex(columns=range(max_layers)) 
distances_df.columns = [f"Layer {i}" for i in range(max_layers)]
distances_df.index.name = "Model"

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.colheader_justify', 'center')  

print(distances_df)
distances_df.to_csv("distances_last_hidden_state.csv", index=True)
