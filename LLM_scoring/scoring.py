import torch
import fire
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import accelerate
from functools import partial
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import numpy as np
from accelerate.utils import is_xpu_available
from accelerate import Accelerator
import regex as re
from datasets import load_dataset
import sys
import gc
from collections import Counter

### store the model 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
B_INST, E_INST = "[INST]", "[/INST]"


class CustomDataset(Dataset):
    def __init__(self, dataset_name, dialogs, template=None):
        self.dataset_name = dataset_name
        self.dialogs = dialogs
        self.template = template
    def __getitem__(self, idx):

        return {'data': self.dialogs[idx], 'index': idx}
    
    def __len__(self):
        return len(self.dialogs)
    
    def map(self, function):
        self.dialogs = [function(item, self.template) for item in self.dialogs]
        return self
    

def preprocessing(dataset_name, prompt_template, system_prompt, user_prompt):
    inputs = []


    if dataset_name == 'tulu_300k':
        dialogs = load_dataset('jlpang888/tulu_300k')['train']
        for dialog in dialogs:
            conversation = ""
            for message in dialog['messages']:  #[{{'role': 'user', 'content': 'blabla'}, {'role': 'assistant', 'content': 'blabla'}]
                conversation += f"### {message['role']}: {message['content']}\n"
            inputs.append(prompt_template.format(system_prompt, user_prompt, conversation))
    else:
        print("Unknown dataset! Please check the dialog information!")
        raise NotImplementedError
    
    return inputs


def load_model(model_name):
    '''load model & tokenizer'''
    if 'llama' in model_name.lower():
        model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        batch_size =30
        # chat template: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
        prompt_template = '''
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>{} \n## Data sample (conversation):\n{}<|eot_id|> 
                    <|start_header_id|>assistant<|end_header_id|>
                    '''
        
    elif 'mistral' in model_name.lower():
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        batch_size = 50
        # chat template: https://www.promptingguide.ai/models/mistral-7b
        # chat template: <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
        prompt_template = '''
                    <s>[INST]system{}[/INST]
                    [INST]user{} \n## Data sample (conversation):\n{}[/INST]
                    [INST]assistant
                    '''    
    

    elif 'gemma' in model_name.lower():
        model_full_name = 'google/gemma-2-9b-it' 
        batch_size=20
        
        prompt_template = '''
            <bos><start_of_turn>system{}<end_of_turn>
            <bos><start_of_turn>user{} \n## Data sample (conversation):\n{}<end_of_turn>
            <start_of_turn>model
            '''    

    elif "phi" in model_name.lower():
        model_full_name = "microsoft/Phi-3.5-mini-instruct"
        batch_size = 20

        prompt_template = '''
             <|system|>{}<end>\n
            <|user|>{} \n## Data sample (conversation):\n{}<end>\n
            <|assistant|>
            '''    
            
    else:
        raise NotImplementedError
    
    return  model_full_name, batch_size, prompt_template

def gen_prompt():
    
    '''prompt template'''
    
    system_prompt = '''As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
        Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
        Ensure that the ratings are not overly concentrated around a specific score. If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences.
        '''
    
    user_prompt ='''Please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
        {
            "Rarity": <number, 1-10>,
            "Complexity": <number, 1-10>,
            "Informativeness": <number, 1-10>,
            "Overall rating": <number, 1-10>
        }
        '''   
        
    return system_prompt, user_prompt 


def score_compress(original_scores):
    original_scores = [score[-1] for score in original_scores] #take the overall rating scores
    print(f"Original score distribution{Counter(original_scores)}")

    #rematching to [0,1,2,3,4,5]
    scores_revised = []
    for score in original_scores:
        if score <= 4:
            scores_revised.append(4)
        elif score >=9:
            scores_revised.append(9)
        else:
            scores_revised.append(score)

    scores_revised = [score - 4 for score in scores_revised] 
    print(f"Revised scores distribution {Counter(scores_revised)}\n")

    return scores_revised


def main(
    model_name: str = "llama",
    dataset_name: str = 'flan_v2',
    subset_name: str = None,
    max_new_tokens = 128, 
    seed: int=42,
    root_path: str='logs',
    do_sample: bool=True, 
    temperature: float=1.2, 
    top_k: int=50, 
    top_p: float=0.9,
    repetition_penalty: float=1.0,
    length_penalty: int=1, 
    output_dir="./logs/",
    use_cache=False,
    **kwargs
    ):
    
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


    system_prompt, user_prompt = gen_prompt()

    model_full_name, batch_size, prompt_template = load_model(model_name)


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        torch_dtype=torch.bfloat16,
        quantization_config = bnb_config,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Preprocessing dataset...")
    inputs = preprocessing(dataset_name, prompt_template, system_prompt, user_prompt)


    dataset = CustomDataset(dataset_name, inputs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
    data_loader, model, tokenizer = accelerator.prepare(data_loader, model, tokenizer)

    output_scores = []
    results = [] 
    rating_all = []

    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    model.eval()
    for batch in tqdm(data_loader, desc="Generating inference info for answers"):

        batch_data = batch['data']
        batch_indices = batch['index'] #record the index for each sample to maintain the sequence

        encodings = tokenizer(batch_data, padding=True, max_length=2048, truncation=True, return_tensors="pt")
        encodings = {k: v.to(accelerator.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(
                input_ids=encodings['input_ids'].to(accelerator.device),
                attention_mask=encodings['attention_mask'].to(accelerator.device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=False,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            rating_batch = [None] * len(batch_data)
            for idx, output_text in enumerate(output_text_batch):

                retry_count = 10
                try:
                    while retry_count>0:
                        matches = json_pattern.findall(output_text)
                        if matches:
                            try:
                                # extract the json object
                                json_obj = json.loads(matches[-1])
                                rating_batch[idx] = [int(json_obj['Rarity']), int(json_obj['Complexity']), int(json_obj['Informativeness']), int(json_obj['Overall rating'])]
                                break  
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for batch data {batch_indices[idx]}")
                        else:
                            print(f"No JSON match for batch data {batch_indices[idx]}, recalculating...")

                            with torch.no_grad():
                                single_output = accelerator.unwrap_model(model).generate(
                                    input_ids=encodings['input_ids'][idx].unsqueeze(0).to(accelerator.device),
                                    attention_mask=encodings['attention_mask'][idx].unsqueeze(0).to(accelerator.device),
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    top_p=top_p,
                                    temperature=temperature,
                                    use_cache=use_cache,
                                    top_k=top_k,
                                    repetition_penalty=repetition_penalty,
                                    length_penalty=length_penalty,
                                    **kwargs
                                )
                            output_text = tokenizer.decode(single_output[0], skip_special_tokens=True)
                        retry_count -= 1
                        
                except Exception as exc:
                    print(f'{batch_indices[idx]} generated an exception: {exc}')

                results.append((batch_indices[idx], rating_batch[idx] if rating_batch[idx] is not None else [0,0,0,0]))


            rating_all.extend(rating_batch)
            print(f"Unrated samples size of each batch: {rating_batch.count(None)}")

        del encodings, output_text_batch, batch
        torch.cuda.empty_cache()


    print(f"Number of unrated samples: {rating_all.count(None)}")
    
    from collections import Counter
    rating_all_revise = [rating[-1] for rating in rating_all if rating is not None]
    print(f"Score distribution: {Counter(rating_all_revise)}")


    # Barrier to ensure all processes have finished saving
    accelerator.wait_for_everyone()
    
    # Convert results to tensors and move them to CUDA device
    indices_tensor = torch.tensor([x[0] for x in results], dtype=torch.long).to(accelerator.device)
    rating_tensor = torch.tensor([x[1] for x in results], dtype=torch.int).to(accelerator.device)

    # Gather results from all processes
    all_indices = accelerator.gather(indices_tensor)
    all_ratings = accelerator.gather(rating_tensor)


    if accelerator.is_main_process:
        # Convert gathered tensors back to list of tuples
        gathered_results = {}

        indices_list = all_indices.cpu().tolist()
        ratings_list = all_ratings.cpu().tolist()

        gathered_results = dict(zip(indices_list, zip(ratings_list)))

        # Sort results by original index
        sorted_results = sorted(gathered_results.items(), key=lambda x: x[0])
        output_scores = [x[1][0] for x in sorted_results]

        # Save the merged results
        accelerator.end_training()

        output_dir = output_dir + f"{model_full_name}/{dataset_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        ## rematching raw score to [0,1,2,3,4,5]
        output_scores_revised = score_compress(output_scores)

        torch.save(sorted_results, f'{output_dir}/results.pt')
        torch.save(output_scores, f'{output_dir}/output_scores.pt')
        torch.save(output_scores_revised, f'{output_dir}/output_scores_revised.pt')

        print('Finished generation and saving!')



if __name__ == '__main__':
    fire.Fire(main)
    
    
    