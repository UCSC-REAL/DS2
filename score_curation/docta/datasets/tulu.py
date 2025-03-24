from .customize import CustomizedDataset
import gzip, json, re
import numpy as np
import os
import torch
from datasets import load_dataset
import pandas as pd

class TULU_RLHF(CustomizedDataset):
    def __init__(self, cfg, args, train=True):

        ## dataset 
        score_path = cfg.score_path
        self.feature_raw = load_dataset('jlpang888/tulu_300k')['train']

        ###########################################################
        # load & save datasets
        os.makedirs(cfg.save_path, exist_ok=True)
        score = torch.load(score_path)
        print(f'preprocessed dataset {cfg.preprocessed_dataset_path}...')

        feature = []

        for dialog in self.feature_raw:
            conversation = ""
            for messege in dialog['messages']:
                conversation += f"###{messege['role']}: {messege['content']}\n"
            feature.append(conversation)

        score = np.array(score)
        torch.save({'feature': feature, 'label': score}, cfg.preprocessed_dataset_path)
        print(f'Saved preprocessed dataset to {cfg.preprocessed_dataset_path}')
        
        assert len(feature) == len(score)
        print(f'Whole dataset size: {len(feature)}')
        
        index = range(len(feature))
        super(TULU_RLHF, self).__init__(feature, score, index=index, preprocess=None)
                
                


    def split_string_by_keywords(self, input_str, keywords):
        regex = re.compile('({})'.format('|'.join(map(re.escape, keywords))))
        substrings = regex.split(input_str.strip())
        substrings = [s.strip() for s in substrings if len(s.strip()) > 0]
        result = {}
        for keyword in keywords:
            result[keyword] = [substrings[i+1] for i in range(len(substrings) - 1) if substrings[i].startswith(keyword) and (substrings[i+1] not in keywords)]
        return result # divide responses according to human/assistant
        # dict{Human: xxx, Assistant: xxx}

    


    def filter_data(self, key = 'Assistant:'):
        rec, chosen_filtered, rejected_filtered = [], [], []
        for i in range(len(self.chosen)):
            chosen = self.chosen[i][key]
            rejected = self.rejected[i][key]
            if len(chosen) == 0: # 
                chosen = [self.chosen[i]['Human:'][2*j + 1] for j in range(len(self.chosen[i]['Human:'])//2)] 
            
            if len(rejected) == 0: # 
                rejected = [self.rejected[i]['Human:'][2*j + 1] for j in range(len(self.rejected[i]['Human:'])//2)] 

            cnt = 0
            range_i = min(len(chosen), len(rejected))
            for j in range(range_i):
                if chosen[j] != rejected[j]:
                    cnt += 1
                    chosen_filtered.append(chosen[j:])
                    rejected_filtered.append(rejected[j:])
            if cnt == 0:
                chosen_filtered.append(chosen[j:])
                rejected_filtered.append(rejected[j:])
            rec.append(cnt) # rec must be no larger than 1

        assert max(rec) == 1
        self.result = dict(
            chosen = chosen_filtered,
            rejected = rejected_filtered
        )

