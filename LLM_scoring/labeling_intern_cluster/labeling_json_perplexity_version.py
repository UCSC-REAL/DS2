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
        # 直接存储原始对话数据
        self.dataset_name = dataset_name
        self.dialogs = dialogs
        self.template = template
    def __getitem__(self, idx):

        # return self.dialogs[idx]

        return {'data': self.dialogs[idx], 'index': idx}
    
    def __len__(self):
        return len(self.dialogs)
    
    def map(self, function):
        self.dialogs = [function(item, self.template) for item in self.dialogs]
        return self
    


def read_dialogs_from_json(file: str,  data_size):
    dialogs = []
    # The number of dialogs
    # halueval
    # end_dialog_idx = 1010
    # start_dialog_idx = 10
    
    start_dialog_idx = 0
    # end_dialog_idx = 10000

    end_dialog_idx = data_size
    
    dialog_idx = 0  # Start counting from 0 to correctly match the first dialog as index 1
    with open(file, 'r') as f:
        for line in f:
            # Skip comment lines
            if not line.strip() or line.strip().startswith("//"):
                continue  # Skip the line if it's a comment or blank


            if start_dialog_idx <= dialog_idx <  end_dialog_idx:
                # This point can use pdb for debugging or directly process the data
                dialog_data = json.loads(line)
                user_query = dialog_data["user_query"]
                chatgpt_response = dialog_data['chatgpt_response']
                hallucination_label = dialog_data['hallucination']  # 'yes' or 'no'
                # hallucination_spans = dialog_data.get('hallucination_spans', [])  # Use get to handle missing fields

                # Construct the dialog dictionary
                dialog = [{
                    # "role": "user",
                    "content": user_query,
                    "chatgpt_response": chatgpt_response,
                    "hallucination_label": hallucination_label,
                    # "hallucination_spans": hallucination_spans
                }]
                
                dialogs.append(dialog)
            elif dialog_idx > end_dialog_idx:
                # Stop reading the file if the end of the target range is reached
                break
            
            
            dialog_idx += 1  # Increment dialog index only for non-comment lines

    return dialogs



def main(
    model_name: str = "llama",
    dataset_name: str = 'flan_v2',
    subset_name: str = None,
    prompt_template = 4, ### the prompt template
    data_size = 3000,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 128, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    token_gap: int=0,
    root_path: str='logs',
    gpu_id: int=None,
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.2, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.2, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    enable_llamaguard_content_safety: bool = False,
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

    '''load model & tokenizer'''
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # flash attention 2
    # torch.bfloat16
    # 8bit 4bit bitsandbytes
    # accelerate
    
    if 'llama' in model_name:
        model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        batch_size =30 #30

        # chat template: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
        model_start_tag = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>"
        model_end_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        pre_prompt = ('''
            As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
            Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
            Now, please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
            {
                "Rarity": <number, 1-10>,
                "Complexity": <number, 1-10>,
                "Informativeness": <number, 1-10>,
                "Overall rating": <number, 1-10>
            }
            Remember: the output must strictly follow this format, without any deviations.
            ''')  


    elif 'mistral' in model_name:
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        # model_full_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        batch_size = 30
        # chat template: https://www.promptingguide.ai/models/mistral-7b
        # chat template: <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
        model_start_tag = "<s>[INST]"
        model_end_tag = "[/INST]"
        pre_prompt = ('''
            As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
            Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
            Ensure that the ratings are not overly concentrated around a specific score. If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences.
            Now, please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
            {
                "Rarity": <number, 1-10>,
                "Complexity": <number, 1-10>,
                "Informativeness": <number, 1-10>,
                "Overall rating": <number, 1-10>
            }
            Remember: the output must strictly follow this format, without any deviations.
            ''')  


    elif 'gemma' in model_name:
        model_full_name = 'google/gemma-2b-it' # batch_size 20
        # model_full_name = 'google/gemma-7b-it' #batch_size 10
        # model_full_name = 'google/gemma-2-9b-it' #batch_size 10
        batch_size=10

        
    elif model_name == 'opt':
        model_full_name = 'facebook/opt-6.7b'



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
    # import pdb;pdb.set_trace()

    device_map=f'cuda:{gpu_id}' if gpu_id is not None else 'auto'
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        torch_dtype=torch.bfloat16,
        quantization_config = bnb_config,
        # attn_implementation="flash_attention_2",  # 假设模型支持这个参数
        # device_map="balanced",#"auto", "balanced", "balanced_low_0", "sequential"
        # device_map="auto", # when you use the accelerator, you don't need to set device_map
        # device_map={'':torch.cuda.current_device()},
        # device_map = {"": accelerator.device},
    )
    model.bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    '''prompting'''
    print("choose the prompt for model!!") ## pre_prompt


    '''preprocess dataset'''
    print("Preprocessing dataset...")
    inputs= []
    ##########################################################################################

    dataset_list =['flan_v2', 'oasst1', 'wizardlm', 'dolly', 'stanford_alpaca']

    if dataset_name in dataset_list:
        # data = load_dataset('parquet', data_files=f'./tulu_split_parquet/{dataset_name}.parquet')
        data = load_dataset('json', data_files=f'./train_data/{dataset_name}_data.jsonl')
        dialogs = data['train']

        for dialog in dialogs:
            conversation = ""
            for message in dialog['messages']:  #[{{'role': 'user', 'content': 'blabla'}, {'role': 'assistant', 'content': 'blabla'}]
                conversation += f"### {message['role']}: {message['content']}\n"
            inputs.append(model_start_tag + pre_prompt + "## Data sample (conversation):\n" + conversation + model_end_tag)
            
    elif 'alpaca' in dataset_name:
        print(f"###################load dataset: tatsu-lab/alpaca")
        data = load_dataset("tatsu-lab/alpaca")
        dialogs = data['train']
        for dialog in dialogs['text']:
            inputs.append(pre_prompt + "## Data sample (conversation):\n" + dialog + model_end_tag)

    elif 'dolly' in dataset_name:
        print(f"###################load dataset: databricks/databricks-dolly-15k")
        data = load_dataset("databricks/databricks-dolly-15k")
        dialogs = data['train']
        for dialog in dialogs:
            conversation = f"### Instruction: {dialog['instruction']} ### Response: {dialog['response']}"
            inputs.append(pre_prompt + f"## Data sample (conversation):\n" + conversation + model_end_tag)

    elif 'wizardLM' in dataset_name:
        print(f"###################load dataset: WizardLMTeam/WizardLM_evol_instruct_V2_196k")
        data = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k")
        dialogs = data['train']
        for dialog in dialogs['conversations']:
            f"### Human: {dialog[0]['value']} ### Assitant: {dialog[1]['value']}"
            conversation =  f"### Human: {dialog[0]['value']} ### Assitant: {dialog[1]['value']}"
            inputs.append(pre_prompt + f"## Data sample (conversation):\n" + conversation + model_end_tag)

    else:
        print("no such dataset used.")


    
    # dialogs = load_data(dataset_name, subset_name, data_size)
    dataset = CustomDataset(dataset_name, inputs, template=prompt_template)
    # dataset = dataset.map(create_prompt_formats)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, shuffle=True, seed=42 


    ###accelerator 
    data_loader, model, tokenizer = accelerator.prepare(data_loader, model, tokenizer)

    output_text_all = []
    output_labels = []
    results = [] #store the results for data parallel
    perplexity_all = []

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

            # 计算 perplexity
            logits = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask']).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = encodings['input_ids'][..., 1:].contiguous()

            # 使用交叉熵损失计算 perplexity
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
            loss = loss.view(shift_labels.size())
            perplexity_batch = torch.exp(loss.sum(dim=-1) / shift_labels.ne(tokenizer.pad_token_id).sum(dim=-1))

            # perplexity_batch = perplexity_batch.cpu().tolist()
            # perplexity_all.extend(perplexity_batch)


            output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            rating_batch = [None] * len(batch_data)
            
            for idx, output_text in enumerate(output_text_batch):
                # print("########################################################################################\n")
                # print(output_text)   
                # print("\n########################################################################################")

                retry_count = 20
                try:
                    while retry_count>0:
                        matches = json_pattern.findall(output_text)
                        if matches:
                            try:
                                # 解析并保存完整的 JSON 对象
                                json_obj = json.loads(matches[-1])
                                # rating_batch[idx] = json.dumps(json_obj)
                                rating_batch[idx] = [int(json_obj['Rarity']), int(json_obj['Complexity']), int(json_obj['Informativeness']), int(json_obj['Overall rating'])]
                                break  # 成功提取到 JSON，退出循环
                            except json.JSONDecodeError:
                                print(f"JSON Decode Error for batch data {batch_indices[idx]}")
                        else:
                            print(f"No JSON match for batch data {batch_indices[idx]}, recalculating...")
                            # 重新运行单个 example
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
            print(f"$$$$$$$$$$$$$ rating batch (size): {len(rating_batch)}): {rating_batch}")
            print(f"$$$$$$$$$$$$$ rating batch's unlabel samples: {rating_batch.count(None)}")

        del encodings, output_text_batch, batch
        torch.cuda.empty_cache()


    print(f"$$$$$$$$$$$$$ all unlabel samples: {rating_all.count(None)}")
    from collections import Counter
    rating_all_revise = [rating[-1] for rating in rating_all if rating is not None]
    print(f"score distribution: {Counter(rating_all_revise)}")
    # import pdb;pdb.set_trace()




    '''load parameters'''
    print('Storing parameters...')
    if subset_name is not None: 
        path = os.path.join(root_path, model_name, f"{dataset_name}-{subset_name}")
    else:
        path = os.path.join(root_path, model_name, dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
        
    # torch.save(output_text_all, path + f'/output_text_all.pt')
    # torch.save(output_labels, path + f'/output_labels.pt')


    #####################################################################################

    # Barrier to ensure all processes have finished saving
    accelerator.wait_for_everyone()
    
    # Convert results to tensors and move them to CUDA device
    indices_tensor = torch.tensor([x[0] for x in results], dtype=torch.long).to(accelerator.device)
    # text_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(list(x[1].encode('utf-8')), dtype=torch.long) for x in results], batch_first=True, padding_value=0).to(accelerator.device)
    
    # rating_tensor = torch.tensor([[temp if temp > 10 or temp < 0 else 0 for temp in x[1]] 
    #                                 for x in results
    #                             ], dtype=torch.long).to(accelerator.device)  
    
    min_val = torch.iinfo(torch.int64).min

    max_val = torch.iinfo(torch.int64).max

    rating_tensor = torch.tensor(
                                [[min(max(val, min_val), max_val) for val in x[1]] for x in results], dtype=torch.long  # 使用 int64
                                    ).to(accelerator.device)

    rating_tensor = torch.tensor([x[1] for x in results], dtype=torch.long).to(accelerator.device)
    perplexity_tensor = torch.tensor([x[2] for x in results], dtype=torch.float).to(accelerator.device)


    # Gather results from all processes
    all_indices = accelerator.gather(indices_tensor)
    # all_texts = accelerator.gather(text_tensor)
    all_ratings = accelerator.gather(rating_tensor)
    all_perplexitys = accelerator.gather(perplexity_tensor)

    # Only main process should sort and save results
    if accelerator.is_main_process:
        # Convert gathered tensors back to list of tuples
        gathered_results = {}

        indices_list = all_indices.cpu().tolist()
        ratings_list = all_ratings.cpu().tolist()
        perplexity_list = all_perplexitys.cpu().tolist()

        # for idx, text, rating in zip(indices_list, all_texts, ratings_list):
        #     # gathered_results[idx] = (bytes(text[text != 0].cpu().numpy()).decode('utf-8'), rating)
        #     gathered_results[idx] = (bytes(text[text != 0].cpu().numpy()).decode('utf-8', errors='replace'), rating)

        # from concurrent.futures import ThreadPoolExecutor
        # def decode_text(text):
        #     return bytes(text[text != 0]).decode('utf-8', errors='replace')

        # with ThreadPoolExecutor() as executor:
        #     texts_decoded = list(executor.map(decode_text, [t.cpu().numpy() for t in all_texts]))
        # gathered_results = dict(zip(indices_list, zip(texts_decoded, ratings_list)))

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
        # final_text_path = f'{path}/output_text_all.pt'

        # assert len(output_labels) == len(dialogs)
 
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
        # torch.save(output_text_all, final_text_path)
        torch.save(output_labels, final_labels_path)
        torch.save(output_perplexities, final_perplexity_path)

        print('Finished generation and saving!')

        files_in_output_dir = os.listdir(output_dir)
        print("Files in output directory:", files_in_output_dir)

        print("Main process is exiting...")





    

if __name__ == '__main__':
    fire.Fire(main)
    
    
    