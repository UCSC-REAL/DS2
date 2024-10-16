import torch
import fire
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import accelerate
from functools import partial
from torch.utils.data import DataLoader,Dataset
# from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from accelerate.utils import is_xpu_available
# from typing import Iterable, List, Optional, Tuple
from accelerate import Accelerator
import regex as re
from datasets import load_dataset
import sys
import gc
print("Torch version:", torch.__version__)
import torch.nn.functional as F



### store the model 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
B_INST, E_INST = "[INST]", "[/INST]"


class CustomDataset(Dataset):
    def __init__(self, dataset_name, dialogs, template):
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
    



def main(
    model_name: str = "llama",
    dataset_name: str = 'flan_v2',
    subset_name: str = None,
    prompt_template = 4, ### the prompt template
    max_new_tokens = 128, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    token_gap: int=0,
    root_path: str='logs',
    gpu_id: int=None,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.2, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.2, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    output_dir="/mnt/azureml/crunch/outputs/",
    # output_dir=".",
    **kwargs
):

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


    '''prompt template'''
    # full_prompt_template = ('''As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
    # Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
    # Ensure that the ratings are not overly concentrated around a specific score. If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences.
    # Now, please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
    # {
    #     "Rarity": <number, 1-10>,
    #     "Complexity": <number, 1-10>,
    #     "Informativeness": <number, 1-10>,
    #     "Overall rating": <number, 1-10>
    # }
    # Remember: the output must strictly follow this format, without any deviations.
    # ''')  
    
    system_prompt = '''As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
        Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
        Ensure that the ratings are not overly concentrated around a specific score. If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences.
        '''
    
    user_prompt ='''Now, please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
        {
            "Rarity": <number, 1-10>,
            "Complexity": <number, 1-10>,
            "Informativeness": <number, 1-10>,
            "Overall rating": <number, 1-10>
        }
        Remember: the output must strictly follow this format, without any deviations.
        '''   


    '''load model & tokenizer'''
    if 'llama' in model_name.lower():
        model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        batch_size =30
        # chat template: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
        prompt_template = '''
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>
                    <|start_header_id|>user<|end_header_id|>{user_prompt} \n## Data sample (conversation):\n{conversation}<|eot_id|> 
                    <|start_header_id|>assistant<|end_header_id|>
                    '''
        
    elif 'mistral' in model_name.lower():
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        batch_size = 50
        # chat template: https://www.promptingguide.ai/models/mistral-7b
        # chat template: <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
        prompt_template = '''
                    <s>[INST]system{system_prompt}[/INST]
                    [INST]user{user_prompt} \n## Data sample (conversation):\n{conversation}[/INST]
                    [INST]assistant
                    '''    
    

    elif 'gemma' in model_name.lower():
        model_full_name = 'google/gemma-2-9b-it' 
        batch_size=20
        
        prompt_template = '''
            <bos><start_of_turn>system{system_prompt}<end_of_turn>
            <bos><start_of_turn>user{user_prompt} \n## Data sample (conversation):\n{conversation}<end_of_turn>
            <start_of_turn>model
            '''    

    elif "phi" in model_name.lower():
        model_full_name = "microsoft/Phi-3.5-mini-instruct"
        batch_size = 20

        prompt_template = '''
             <|system|>{system_prompt}<end>\n
            <|user|>{user_prompt} \n## Data sample (conversation):\n{conversation}<end>\n
            <|assistant|>
            '''    
            
    else:
        raise NotImplementedError
    
    print(f'####### Loading LLM model: {model_full_name}')
    print(f'####### Datset: {dataset_name}')
    print(f'####### Batch size: {batch_size}')


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    accelerator = Accelerator()

    device_map=f'cuda:{gpu_id}' if gpu_id is not None else 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        torch_dtype=torch.bfloat16,
        quantization_config = bnb_config,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  
        # device_map="balanced",#"auto", "balanced", "balanced_low_0", "sequential"
        # device_map="auto", # when you use the accelerator, you don't need to set device_map
        # device_map={'':torch.cuda.current_device()},
        # device_map = {"": accelerator.device},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    '''preprocess dataset'''
    print("Preprocessing dataset...")
    inputs= []
    tulu_subsets_list =['flan_v2', 'oasst1', 'wizardlm', 'dolly', 'stanford_alpaca']

    if dataset_name in tulu_subsets_list:
        data = load_dataset('json', data_files=f'./train_data/{dataset_name}_data.jsonl')
        dialogs = data['train']

        for dialog in dialogs:
            conversation = ""
            for message in dialog['messages']:  #[{{'role': 'user', 'content': 'blabla'}, {'role': 'assistant', 'content': 'blabla'}]
                conversation += f"### {message['role']}: {message['content']}\n"
            inputs.append(prompt_template.format(system_prompt, user_prompt, conversation))
    
    #load data from Huggingface
    elif 'alpaca' in dataset_name:
        dataset_name = 'tatsu-lab/alpaca'
        print(f"Loading dataset: {dataset_name}")
        dialogs = load_dataset(dataset_name)['train']
        
        for dialog in dialogs['text']:
            inputs.append(prompt_template.format(system_prompt, user_prompt, conversation))

    elif 'dolly' in dataset_name:        
        dataset_name = 'databricks/databricks-dolly-15k'
        print(f"Loading dataset: {dataset_name}")
        dialogs = load_dataset(dataset_name)['train']
        
        for dialog in dialogs:
            conversation = f"### Instruction: {dialog['instruction']} ### Response: {dialog['response']}"
            inputs.append(prompt_template.format(system_prompt, user_prompt, conversation))

    elif 'wizardLM' in dataset_name:
        dataset_name = 'WizardLMTeam/WizardLM_evol_instruct_V2_196k'
        print(f"Loading dataset: {dataset_name}")
        dialogs = load_dataset(dataset_name)['train']
        
        for dialog in dialogs['conversations']:
            f"### Human: {dialog[0]['value']} ### Assistant: {dialog[1]['value']}"
            conversation =  f"### Human: {dialog[0]['value']} ### Assistant: {dialog[1]['value']}"
            inputs.append(prompt_template.format(system_prompt, user_prompt, conversation))

    else:
        print("Dataset can not be found!")


    
    # dialogs = load_data(dataset_name, subset_name, data_size)
    dataset = CustomDataset(dataset_name, inputs, template=prompt_template)
    # dataset = dataset.map(create_prompt_formats)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, shuffle=True, seed=42 


    ###accelerator 
    data_loader, model, tokenizer = accelerator.prepare(data_loader, model, tokenizer)

    output_labels = []
    results = [] #store the results for data parallel
    rating_all = []

    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    model.eval() 
    for batch in tqdm(data_loader, desc=f"Generating inferences for dataset: {dataset_name}"):

        batch_data = batch['data']
        batch_indices = batch['index'] #record the index for each sample 

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
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            # calculate perplexity
            logits = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask']).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = encodings['input_ids'][..., 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
            loss = loss.view(shift_labels.size())
            perplexity_batch = torch.exp(loss.sum(dim=-1) / shift_labels.ne(tokenizer.pad_token_id).sum(dim=-1))


            output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            rating_batch = [None] * len(batch_data)
            
            for idx, output_text in enumerate(output_text_batch):
                # print("="*50 + "\n")
                # print(output_text)   
                # print("="*50 + "\n")
                
                retry_count = 20
                try:
                    while retry_count>0:
                        matches = json_pattern.findall(output_text)
                        if matches:
                            try:
                                # extract the json object
                                json_obj = json.loads(matches[-1])
                                # rating_batch[idx] = json.dumps(json_obj)
                                rating_batch[idx] = [int(json_obj['Rarity']), int(json_obj['Complexity']), int(json_obj['Informativeness']), int(json_obj['Overall rating'])]
                                break  
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for batch data {batch_indices[idx]}")
                        else:
                            print(f"No JSON match for batch data {batch_indices[idx]}, recalculating...")
                            # re-generate
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

                final_rating = rating_batch[idx] if rating_batch[idx] is not None else [0,0,0,0]

                results.append((batch_indices[idx], final_rating, perplexity_batch[idx]))


            rating_all.extend(rating_batch)
            print(f"Unlabeled samples size of each batch: {rating_batch.count(None)}")


        del encodings, output_text_batch, batch
        torch.cuda.empty_cache()


    print(f"All unlabeled samples: {rating_all.count(None)}")
    from collections import Counter
    rating_all_revise = [rating[-1] for rating in rating_all if rating is not None]
    print(f"score distribution: {Counter(rating_all_revise)}")


    '''load parameters'''
    print('Storing parameters...')
    if subset_name is not None: 
        path = os.path.join(root_path, model_name, f"{dataset_name}-{subset_name}")
    else:
        path = os.path.join(root_path, model_name, dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
        
    # Barrier to ensure all processes have finished saving
    accelerator.wait_for_everyone()
    
    # Convert results to tensors and move them to CUDA device
    indices_tensor = torch.tensor([x[0] for x in results], dtype=torch.long).to(accelerator.device)

    min_val = torch.iinfo(torch.int64).min

    max_val = torch.iinfo(torch.int64).max

    rating_tensor = torch.tensor(
                                [[min(max(val, min_val), max_val) for val in x[1]] for x in results], dtype=torch.long  
                                    ).to(accelerator.device)

    rating_tensor = torch.tensor([x[1] for x in results], dtype=torch.long).to(accelerator.device)
    perplexity_tensor = torch.tensor([x[2] for x in results], dtype=torch.float).to(accelerator.device)


    # Gather results from all processes
    all_indices = accelerator.gather(indices_tensor)
    all_ratings = accelerator.gather(rating_tensor)
    all_perplexitys = accelerator.gather(perplexity_tensor)

    # Only main process should sort and save results
    if accelerator.is_main_process:
        # Convert gathered tensors back to list of tuples
        gathered_results = {}

        indices_list = all_indices.cpu().tolist()
        ratings_list = all_ratings.cpu().tolist()
        perplexity_list = all_perplexitys.cpu().tolist()


        gathered_results = dict(zip(indices_list, zip(ratings_list, perplexity_list)))

        # Sort results by original index
        sorted_results = sorted(gathered_results.items(), key=lambda x: x[0])
        output_labels = [x[1][0] for x in sorted_results]
        output_perplexities = [x[1][1] for x in sorted_results]

        # Save the merged results
        accelerator.end_training()

        output_dir = output_dir + f"{model_full_name}/{dataset_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
 
        final_labels_path = f'{output_dir}/output_labels.pt'
        final_perplexity_path = f'{output_dir}/output_perplexities.pt'

        final_results_path = f'{output_dir}/results.pt'
        print(f"output_labels: {output_labels}")
        label_all = [label[-1] for label in output_labels]

        print(f"final label score distribution: {Counter(label_all)}")
        print(f"final perplexity length: {len(output_perplexities)}")
        print(f"final perplexity: {output_perplexities[:100]}")



        print("starting storing the outputs!!!")
        torch.save(sorted_results, final_results_path)
        torch.save(output_labels, final_labels_path)
        torch.save(output_perplexities, final_perplexity_path)

        print('Finished generation and saving!')

        files_in_output_dir = os.listdir(output_dir)
        print("Files in output directory:", files_in_output_dir)

        print("Main process is exiting...")



if __name__ == '__main__':
    fire.Fire(main)
    
    
    