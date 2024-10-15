import torch
import fire
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
from functools import partial
from torch.utils.data import DataLoader,Dataset
# from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from accelerate.utils import is_xpu_available
# from typing import Iterable, List, Optional, Tuple
from torch import Tensor
import re

from datasets import load_dataset

### store the model 
cache_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/jinlong-exp/code/.cache/huggingface/hub/'

B_INST, E_INST = "[INST]", "[/INST]"


class CustomDataset(Dataset):
    def __init__(self, dataset_name, dialogs, template):
        # 直接存储原始对话数据
        self.dataset_name = dataset_name
        self.dialogs = dialogs
        self.template = template
    def __getitem__(self, idx):
        # 直接返回对话中的内容和回答
        # import pdb;pdb.set_trace()
        # dialog = self.dialogs[idx]
        # prompts = dialog['content']
        # answers = dialog['chatgpt_response']
        # labels = dialog['hallucination_label'] 
        # return prompts, answers, labels
        return self.dialogs[idx]
        # return {'content': content, 'chatgpt_response': chatgpt_response, 'hallucination_label':hallucination_label}
        # dialog = self.dialogs[idx]
        # return {'content': dialog['content'], 'chatgpt_response': dialog['chatgpt_response']}
    
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


def create_prompt_formats(dialog, template):
    """
    Format various fields of the dialog ('content', 'chatgpt_response')
    Then concatenate them using two newline characters: pre_prompt & post_prompt
    """

    if template == 1: # chale
        #prompt template 1 
        
        pre_prompt = 'Question:'
        post_prompt = '\n\nPlease directly give the answer, followed by only one sentence to briefly and shortly describe the relevant information (less than 15 words).'
        
    elif template == 2:#truthfulqa, halueval
        #prompt template 2
        pre_prompt = 'Please directly provide the answer for the following question.\n\nQuestion: '
        post_prompt = '\nAnswer:'
        
    elif template == 3:
        # prompt template 3
        pre_prompt = 'Question:'
        post_prompt = '\n\nAnswer:'

    elif template == 4:
        #blank template
        pre_prompt = ''
        post_prompt = ''
    
    target_answer_length =3
    # typical question-answering type
    if template <=4:
        formatted_prompt = f"{pre_prompt}{dialog['content']}{post_prompt}{' '.join(dialog['chatgpt_response'].split()[:target_answer_length])}"
        # formatted_prompt = f"{B_INST}{pre_prompt}{dialog['content']}{post_prompt}{E_INST}"


    #special case: dialogue type
    if template == 5:
        background = "###Knowledge: "
        pre_prompt = '###Dialogue History: '
        post_prompt = '###Please answer the last question from human according to the Knowledge and dialogue history.\n\nAnswer:'
        
        formatted_prompt = f"{B_INST}{background + dialog['knowledge']}\n{pre_prompt + dialog['content']}\n{post_prompt}{E_INST}"
    
    
    if template == 6: #openmath dataset
        # prompt template 3
        pre_prompt = 'Please solve this question using Python code.  Question:'
        post_prompt = '\n\nAnswer:'
        
        formatted_prompt = f"{B_INST}{pre_prompt}{dialog['content']}{post_prompt}{E_INST}"

        
    
    dialog["content"] = formatted_prompt

    return dialog


def prompting(model_name):
    if model_name == 'gemma':
        prompt = (
            "As a conversation quality evaluator, your task is to assess the quality of the conversation below. "
            "Rate the conversation on a scale from 0 to 5 considering the factors of coherence, relevance, and informativeness. "
            "A rating of 0 means the conversation is of very low quality, and a rating of 5 means it is of very high quality.\n\n"
            "Rubric for Evaluation:\n\n"
            "1. Coherence:\n"
            "   - 0-1: The answer is incoherent or difficult to understand.\n"
            "   - 2-3: The answer is somewhat coherent but may contain unclear segments.\n"
            "   - 4-5: The answer is fully coherent, logical, and easy to understand.\n\n"
            "2. Relevance:\n"
            "   - 0-1: The answer is only partially relevant to the user's query.\n"
            "   - 2-3: The answer is mostly relevant but includes some unrelated information.\n"
            "   - 4-5: The answer is completely relevant and directly addresses the user's query.\n\n"
            "3. Informativeness:\n"
            "   - 0-1: The answer provides minimal useful information.\n"
            "   - 2-3: The answer provides some useful information but lacks detail.\n"
            "   - 4-5: The answer is highly informative and provides detailed and valuable information.\n\n"
            "Here are examples for reference:\n\n"
            "Example 1 (Basic Quality - Score: 1):\n"
            "Conversation:\n"
            "User: Can you explain the Pythagorean theorem?\n"
            "Assistant: Pythagorean theorem says that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is a^2 + b^2 = c^2, where c is the hypotenuse and a and b are the other two sides.\n"
            "## Rating: 1\n\n"
            "Example 2 (High Quality - Score: 4):\n"
            "Conversation:\n"
            "User: Can you explain the Pythagorean theorem?\n"
            "Assistant: The Pythagorean theorem is a fundamental principle in geometry, named after the ancient Greek mathematician Pythagoras. This theorem states that in a right-angled triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. Mathematically, this relationship is expressed as a^2 + b^2 = c^2, where c represents the hypotenuse, and a and b are the lengths of the triangle's other two sides. This theorem is crucial in various fields, including mathematics, physics, engineering, and computer science, because it provides a method to calculate the distance between two points in a plane. For example, if one side of the triangle is 3 units long and the other side is 4 units long, the hypotenuse will be 5 units long, as 3^2 + 4^2 = 5^2.\n"
            "## Rating: 4\n\n"
            "Now, please evaluate the following conversation:"
            )
    elif model_name == 'llama':
        prompt = (
        "As a conversation quality evaluator, your task is to assess the quality of the conversation below. "
        "Rate the conversation on a scale from 0 to 5 considering the factors of coherence, relevance, and informativeness. "
        "A rating of 0 means the conversation is with a low quality, and a rating of 4 means it is pretty good.\n\n"
        "Here is an example:\n"
        "Conversation:\n"
        "User: What is the capital of France?\n"
        "Assistant: The capital of France is Paris.\n"
        "## Rating: 1\n\n"
        "Now, please evaluate the following conversation and return the numerical (integer) rating score without explanations.\n"
        )
    elif model_name == 'mistral':
        prompt = (
            "As a conversation quality evaluator, your task is to assess the quality of the conversation below. "
            "Rate the conversation on a scale from 0 to 5 considering the factors of coherence, relevance, and informativeness. "
            "A rating of 0 means the conversation is of very low quality, and a rating of 5 means it is of very high quality.\n\n"
            "Now, please evaluate the following conversation and return the numerical (integer) rating score with a brief explanation (two or three sentences).\n"
            )
        
    elif model_name == 'opt':
        prompt = ("xx")
    elif model_name == 'gpt':
        prompt = ("xx")
    else:
        prompt = (
        "As a conversation quality evaluator, your task is to assess the quality of the conversation below. "
        "Rate the conversation on a scale from 0 to 10 considering the factors of coherence, relevance, and informativeness. "
        "A rating of 0 means the conversation is with a low quality, and a rating of 2 means it is pretty good.\n\n"
        "Here is an example:\n"
        "Conversation:\n"
        "User: What is the capital of France?\n"
        "Assistant: The capital of France is Paris.\n"
        "## Rating: 9\n\n"
        "Now, please evaluate the following conversation:\n"
        )
    return prompt


def load_data(dataset, subset_name, data_size):
    print(f'starting load data from dataset: {dataset}!')
    data_path = '/home/jlpang/llama/data/'
    if dataset == 'chale':
        dialogs = read_dialogs_from_json(f'{data_path}Chale.json', data_size) # Chale dataset
    elif dataset == 'truthfulqa':
        dialogs = read_dialogs_from_json(f'{data_path}Truthful_QA.json', data_size) # TruthfulQA dataset
    elif dataset == 'halueval':
        if subset_name == 'dialogue': #obtain knowledge info from dialogue
            dialogs = read_dialogs_from_json_halueval_dialogue(f'{data_path}HaluEval_{subset_name}.json', data_size)
        else:
            dialogs = read_dialogs_from_json(f'{data_path}HaluEval_{subset_name}.json', data_size)
    elif dataset == 'mmlu':
        dialogs  = read_dialogs_from_json(f'{data_path}mmlu.json', data_size)
    elif dataset == 'arc':
        dialogs = read_dialogs_from_json(f'{data_path}arc.json', data_size)
    elif dataset =='factCHD':
        dialogs = read_dialogs_from_json(f'{data_path}factCHD.json', data_size)
    elif dataset =='hallu_bio':
        dialogs = read_dialogs_from_json(f'{data_path}hallu_bio.json', data_size)
    elif dataset == 'openmath':
        dialogs = read_dialogs_from_json(f'{data_path}openmath.json', data_size)

        
    #sentence-level
    elif dataset =='felm':
        dialogs = read_dialogs_from_json_felm(f'{data_path}felm.json', data_size)
    elif dataset =='wikibio':
        dialogs = read_dialogs_from_json_felm(f'{data_path}wiki_bio.json', data_size)


    elif dataset == 'test':
        #naive example
        dialogs = [
        [{"role": "user", "content": "who had the most wins in the nfl?"}],
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        ]
    else:
        raise NotImplementedError

    return dialogs



# def generate(model, tokenizer, data_loader, top_p, top_k, temperature, target_token_idx,  top_g, replace_first_token):
#     """
#     This Python function generates inference information for answers using a given model and tokenizer,
#     calculating log probabilities and other statistics.
    
#     Return: The function `generate` returns the following outputs:
#     1. `out_logprobs`: List of log probabilities corresponding to the actual answer tokens.
#     2. `out_question_logprobs`: List of log probabilities corresponding to the question tokens.
#     3. `hallu_labels`: List of labels for the generated data.
#     4. `token_full_logits`: List of full logits for each token in the answer
#     """
#     padding_side = tokenizer.padding_side
    
#     output_text_all = []

#     # Get model's output (logits) for the batch
#     for batch in tqdm(data_loader, desc="Generating inference info for answers"):
        
#         prompts, answers, labels = batch
#         encodings = tokenizer(prompts, padding=True, max_length=2048, truncation=True, return_tensors="pt").to(model.device)
        
#         outputs = model.generate(**encodings, 
#                                 max_length=2048, 
#                                 do_sample=True, top_p=0.9, 
#                                 top_k=50, 
#                                 temperature = 0.6, 
#                                 eos_token_id=tokenizer.eos_token_id, 
#                                 pad_token_id=tokenizer.pad_token_id, 
#                                 target_token_idx=target_token_idx,  
#                                 top_g=top_g, 
#                                 replace_first_token=replace_first_token
#                                 )
        
#         # import pdb;pdb.set_trace()

#         output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
#         output_text_all.extend(output_text_batch)
#         for i, output_text in enumerate(output_text_batch):
#             print('#' * 50 + '\n')
#             print(output_text)   
#             print('\n' + '#' * 50)
     
#     return output_text_all



def main(
    model_name: str = "llama",
    dataset_name: str = 'GAIR/lima',
    subset_name: str = None,
    prompt_template = 4, ### the prompt template
    data_size = 3000,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =1024, #The maximum numbers of tokens to generate
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
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    enable_llamaguard_content_safety: bool = False,
    target_token_idx: int = 0, 
    top_g: int=5,
    replace_first_token: bool= False,
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
    
    if model_name == 'llama':
        # model_full_name = "meta-llama/Llama-2-7b-chat-hf" #batch_size 25
        model_full_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        # model_full_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        batch_size = 25

    elif model_name == 'mistral':
        model_full_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        # model_full_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        batch_size = 50
        
    elif model_name == 'gemma':
        model_full_name = 'google/gemma-2b-it' # batch_size 20
        # model_full_name = 'google/gemma-7b-it' #batch_size 10
        # model_full_name = 'google/gemma-2-9b-it' #batch_size 10
        batch_size=20

    else:
        raise NotImplementedError
    
    print(f'Loading LLM model: {model_full_name}')
    device_map=f'cuda:{gpu_id}' if gpu_id is not None else 'auto'

    # import pdb;pdb.set_trace()
    model = AutoModelForCausalLM.from_pretrained(
        model_full_name,
        # torch_dtype=torch.bfloat16,
        # load_in_4bit=True,
        # attn_implementation="flash_attention_2", #accelerate
        # device_map="auto",
        # device_map="balanced",#"auto", "balanced", "balanced_low_0", "sequential"
        device_map=device_map,
        cache_dir=cache_dir,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_full_name, padding_side='left',cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})



    '''prompting'''
    print("choose the prompt for model!!")
    pre_prompt = prompting(model_name)



    '''preprocess dataset'''
    print("Preprocessing dataset...")
    inputs= []
    ##########################################################################################
    # dataset_name = "allenai/tulu-v2-sft-mixture"
    # data = load_dataset(dataset_name, cache_dir=cache_dir)
    # dialogs = data['train'].select(range(1000))

    dataset_name = 'flan_v2'
    data = load_dataset('parquet', data_files=f'data_refine/tulu_split_parquet/{dataset_name}.parquet')
    dialogs = data['train']

    # for tulu dataset
    for dialog in dialogs:
        conversation = ""
        for message in dialog['messages']:  #[{{'role': 'user', 'content': 'blabla'}, {'role': 'assistant', 'content': 'blabla'}]
            conversation += f"### {message['role']}: {message['content']}\n"
        # import pdb;pdb.set_trace()
        inputs.append(pre_prompt + conversation + "\n### Rating:")

    ##########################################################################################
    # dataset_name = "GAIR/lima"
    # dataset = load_dataset(dataset_name,cache_dir=cache_dir)
    # dialogs = dataset['train']
    
    # # for LIMA dataset
    # for dialog in dialogs:
    #     # import pdb;pdb.set_trace()
    #     for message in dialog['conversations']: 
    #         conversation = f"### User: {dialog['conversations'][0]} \n\n### Assistant: {dialog['conversations'][-1]}\n ### Rating:"
        
    #     inputs.append(pre_prompt + conversation)
    ##########################################################################################

    # dataset_name = "mosaicml/dolly_hhrlhf"
    # dataset = load_dataset(dataset_name,cache_dir=cache_dir)
    # dialogs = dataset['train'][:10000] 

    # for prompt, response in zip(dialogs['prompt'], dialogs['response']):
    #     conversation = f"### User: {prompt} \n\n### Assistant: {response}\n### Rating:"
    #     inputs.append(pre_prompt + conversation)
    # # # import pdb;pdb.set_trace()


    ##########################################################################################

    # dialogs = load_data(dataset_name, subset_name, data_size)
    dataset = CustomDataset(dataset_name, inputs, template=prompt_template)
    # dataset = dataset.map(create_prompt_formats)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, shuffle=True, seed=42 


    output_text_all = []
    output_labels = []
    for batch in tqdm(data_loader, desc="Generating inference info for answers"):
        # import pdb;pdb.set_trace()

        encodings = tokenizer(batch, padding=True, max_length=2048, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # for input_ids, attention_mask in tqdm(zip(encodings["input_ids"], encodings['attention_mask'])):
                # import pdb;pdb.set_trace()
                # input_ids = input_ids.unsqueeze(0)  # 添加批次维度
                # attention_mask = attention_mask.unsqueeze(0)  # 添加批次维度
            outputs = model.generate(
                **encodings,
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

            # output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text_batch = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            # output_answer_text_batch = [tokenizer.decode(x[attention_mask.shape[1]:], skip_special_tokens=True) for x in outputs]


            for i, output_text in enumerate(output_text_batch):
                # print("########################################################################################\n")
                # print(output_text)   
                # print("\n########################################################################################")
    
                #extract rating score
                match = re.search(r"### Rating: (\d+)", output_text)
                rating = int(match.group(1)) if match else -1

                output_labels.append(rating)

            output_text_all.extend(output_text_batch)


    # import pdb; pdb.set_trace()

    '''load parameters'''
    print('Storing parameters...')
    if subset_name is not None: 
        path = os.path.join(root_path, model_name, f"{dataset_name}-{subset_name}")
    else:
        path = os.path.join(root_path, model_name, dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
        
    torch.save(output_text_all, path + f'/output_text_all.pt')
    torch.save(output_labels, path + f'/output_labels.pt')



    # torch.save(output_text_all, path + f'labels.pt')
    print('Finished generation!')
    
    
    
    

if __name__ == '__main__':
    fire.Fire(main)
    
    
    