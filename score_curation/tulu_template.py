# dataset settings
seed = 0
modality = 'text' # image, text, tabular
num_classes = 6  ## Number of score categories

rating_model = 'meta/llama-3.1-8b-instruct'
dataset_name = 'tulu_300k'


feature_type = 'embedding'
details = False


# embedding_model = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = 'BAAI/bge-large-en-v1.5'
embedding_cfg = dict(
    shuffle = False,
    batch_size = 256,  
    save_num = 800,
    num_workers = 2,
    # new one
    use_pca = False,
    use_mi = False,
    n_neighbors = 10
)


hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 50000,
    already_2nn = False,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 51,
    sample_size = 50000,
    k = 10,
    name = 'simifeat',
    method = 'rank'
)


