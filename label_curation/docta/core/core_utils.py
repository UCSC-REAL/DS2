
import torch.nn.functional as F
import torch


def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix


# def cosDistance_chunked(features, chunk_size=1024):
#     features = F.normalize(features, dim=1)
#     features = features.to('cuda').half()
#     num_samples = features.size(0)
#     distance_matrix = torch.zeros(num_samples, num_samples, device=features.device, dtype=features.dtype)
    
#     for i in range(0, num_samples, chunk_size):
#         end_i = min(i + chunk_size, num_samples)
#         for j in range(0, num_samples, chunk_size):
#             end_j = min(j + chunk_size, num_samples)
#             similarity_matrix_chunk = torch.matmul(features[i:end_i], features[j:end_j].T)
#             distance_matrix[i:end_i, j:end_j] = 1.0 - similarity_matrix_chunk
    
#     return distance_matrix

def cosDistance_chunked(features, all_features, chunk_size=1024):
    features = F.normalize(features, dim=1)
    features = features.to('cuda').half()
    num_samples = features.size(0)
    all_num_samples = all_features.size(0)
    distance_matrix = torch.zeros(num_samples, all_num_samples, device=features.device, dtype=features.dtype)
    
    similarity_matrix = torch.matmul(features, all_features.T)
    # 计算余弦距离
    distance_matrix = 1.0 - similarity_matrix
    
    return distance_matrix


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


