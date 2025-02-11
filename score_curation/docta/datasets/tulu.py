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
        #  = cfg.data_root + f'data/train_data/{args.dataset_name}_data.jsonl'
        self.jsonfilename =  cfg.data_root + f'model_finetune/selected_data/{args.labeling_model}/{args.dataset_name}/full_dataset.json'
        # label_path  =  cfg.data_root + f'selected_data/{args.labeling_model}/{args.dataset_name}/clean_output_labels.pt' for clean evaluation

        label_path = cfg.label_path


        print(f"#### train data path: {self.jsonfilename} ")



        try:
            self.load_data()
        except:
            raise ImportError(f'Data must be downloaded from Huggingface and saved to {cfg.data_root}. self.jsonfilename is {self.jsonfilename}')
        
        ###########################################################
        # load & save datasets
        os.makedirs(cfg.save_path, exist_ok=True)
        

        print(f"#### labeling path: {label_path} ")

        label = torch.load(label_path)
        print(f'preprocessed dataset {cfg.preprocessed_dataset_path}...')

        feature = []

        for dialog in self.feature_raw:
            conversation = ""
            for messege in dialog['messages']:
                conversation += f"###{messege['role']}: {messege['content']}\n"
            feature.append(conversation)

        label = np.array(label)
        # import pdb;pdb.set_trace()
        torch.save({'feature': feature, 'label': label}, cfg.preprocessed_dataset_path)
        print(f'Saved preprocessed dataset to {cfg.preprocessed_dataset_path}')
        
        assert len(feature) == len(label)
        print(f'The whole dataset size: {len(feature)}')
        
        index = range(len(feature))
        super(TULU_RLHF, self).__init__(feature, label, index=index, preprocess=None)
                
                


    def split_string_by_keywords(self, input_str, keywords):
        regex = re.compile('({})'.format('|'.join(map(re.escape, keywords))))
        substrings = regex.split(input_str.strip())
        substrings = [s.strip() for s in substrings if len(s.strip()) > 0]
        result = {}
        for keyword in keywords:
            result[keyword] = [substrings[i+1] for i in range(len(substrings) - 1) if substrings[i].startswith(keyword) and (substrings[i+1] not in keywords)]
        return result # divide responses according to human/assistant
        # dict{Human: xxx, Assistant: xxx}

    
    def load_data(self):
        feature_all = []
        with open(self.jsonfilename, mode="rt", encoding='utf-8') as f:
            data = f.read().strip().splitlines()
            for i in range(len(data)):
                json_tmp = json.loads(data[i])
                #feature_tmp: {'prompt': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDo Psychiatrists in different cultures need to study differently to accommodate the cultural differences in individuals?\n\n\n### Response:\n', 'response': 'Some areas would require a greater awareness of local cultural norms than others, but I would expect that basic psychiatry would require the same kinds of knowledge and skills across different cultures.  In order to practice psychiatry successfully, you would need to understand the cultural context in which\n'}
                # feature = self.split_string_by_keywords(json_tmp, keywords = ['prompt:', 'response:']) ## keywords need to be revised further
                feature_all.append(json_tmp)
        self.feature_raw = feature_all


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

