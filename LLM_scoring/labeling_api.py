from openai import OpenAI
import regex as re
import torch
import fire
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np
import json
'''API key'''

# openai.api_key = 'your-api-key'
# client = openai.OpenAI(api_key=api_key)

# export OPENAI_API_KEY=your_actual_api_key

# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
# #   api_key = "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"
#   api_key = "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"
# )
##################################################################
import os
from openai import AzureOpenAI

os.environ["AZURE_OPENAI_KEY"] = "a35b8b00d740422590852f08ed15f8b0"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt-4o-mini-exp.openai.azure.com/"

# echo 'export AZURE_OPENAI_KEY=a35b8b00d740422590852f08ed15f8b0' >> ~/.bashrc
# echo 'export AZURE_OPENAI_ENDPOINT=https://gpt-4o-mini-exp.openai.azure.com/' >> ~/.bashrc

# export AZURE_OPENAI_KEY=a35b8b00d740422590852f08ed15f8b0
# export AZURE_OPENAI_ENDPOINT=https://gpt-4o-mini-exp.openai.azure.com/

# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),  
#     api_version="2024-02-01",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

client = AzureOpenAI(
    api_key='a35b8b00d740422590852f08ed15f8b0',
    api_version="2024-02-01",
    azure_endpoint="https://gpt-4o-mini-exp.openai.azure.com/"
)



def main(
    deployment_model: str = 'gpt-4o-mini-2024-08-28',
    dataset_name: str = 'flan_v2',
    dataset_type: str='train',
    ):



    # pre_prompt = (
    #     "We are evaluating data samples to determine their suitability for finetuning a language model (LLM). "
    #     "As a data sample quality evaluator, your task is to assess the following data sample based on the criteria listed below. "
    #     "Rate the sample on a scale from 1 to 10 for each criterion, and then give an overall rating on a scale from 1 to 5.\n\n"
    #     "Criteria:\n"
    #     "1. Rarity: How uncommon or unique is this data sample within the dataset?\n"
    #     "2. Completeness: How complete and coherent is the conversation or content in this data sample?\n"
    #     "3. Informativeness: How rich is the data sample in terms of useful information?\n\n"
    #     "A rating of 1 means the sample is not suitable, and a rating of 5 means it is very suitable for finetuning.\n\n"
    #     "Here is an example:\n\n"
    #     "Sample content:\n"
    #     "The quick brown fox jumps over the lazy dog.\n\n"
    #     # "Ratings:\n"
    #     # "Rarity: 2\n"
    #     # "Completeness: 10\n"
    #     # "Informativeness: 3\n"
    #     "### Rating: 1\n\n"
    #     "Now, please evaluate the following data sample and directly return the overall numerical (integer) rating score without explanations.\n\n"
    #     )

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
        ''') 

    print(f"Labeling Model: {deployment_model}")
    print(f"Dataset Name: {dataset_name} ")
    '''preprocess dataset'''

    print("Preprocessing dataset...")
    inputs= []
    ##########################################################################################
    # dataset_name = "allenai/tulu-v2-sft-mixture"
    # data = load_dataset(dataset_name, cache_dir=cache_dir)
    # dialogs = data['train'].select(range(10000))
    # dataset_name = 'flan_v2'
    data = load_dataset('json', data_files=f'data/train_data/{dataset_name}_data.jsonl', split=None, cache_dir=None)
    dialogs = data['train']
    # for tulu dataset
    for dialog in dialogs:
        conversation = ""
        for message in dialog['messages']:  #[{{'role': 'user', 'content': 'blabla'}, {'role': 'assistant', 'content': 'blabla'}]
            conversation += f"### {message['role']}: {message['content']}\n"

        inputs.append(pre_prompt + conversation + "\n### Rating:")


    path = f"./logs-api/{deployment_model}/{dataset_name}/"
    if not os.path.exists(path):
        os.makedirs(path)

    ##############################################################################################################################

    def fetch_content(input, idx):
        completion = client.chat.completions.create(
            model=deployment_model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input}],
            max_tokens=128)

        if completion.choices[0].message.content is not None:
            GPT_content = completion.choices[0].message.content
        return idx, GPT_content



    total_output_labels = []
                
    import concurrent.futures
    from tqdm import tqdm
    
    print("Start API call labeling...")
    print(f"Total dataset size: {len(inputs)}")
    
    batch_size = 1024 #1024 # the batch_size
    split_size = len(inputs)//batch_size + 1

    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')


    for batch_idx in tqdm(range(split_size), desc=f'API Call Labeling model -- {deployment_model}'):
        batch_end =  min((batch_idx+1) * batch_size, len(inputs)) # the end index of batch
        batch_inputs = inputs[batch_idx* batch_size:batch_end] ##data range
        
        contents = [None] * len(batch_inputs)
        output_labels = [None] * len(batch_inputs) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_input = {executor.submit(fetch_content, input, idx): idx for idx, input in enumerate(batch_inputs)}
            for future in concurrent.futures.as_completed(future_to_input):
                idx = future_to_input[future]
                try:
                    idx, GPT_content = future.result()
                    contents[idx] = GPT_content
                    
                    matches = json_pattern.findall(GPT_content)

                    retry_count = 0  # 初始化重试计数器

                    max_retries=3
                    while not matches and retry_count < max_retries:
                        retry_count += 1
                        print(f"Retrying for input index {idx}, attempt {retry_count}")
                        idx, GPT_content = fetch_content(batch_inputs[idx], idx)
                        matches = json_pattern.findall(GPT_content)

                    if matches:
                        try:
                            # 解析并保存完整的 JSON 对象
                            json_obj = json.loads(matches[-1])
                            output_labels[idx] = [
                                int(json_obj['Rarity']),
                                int(json_obj['Complexity']),
                                int(json_obj['Informativeness']),
                                int(json_obj['Overall rating'])
                            ]

                        except json.JSONDecodeError:
                            print(f"JSON Decode Error for batch data {batch_indices[idx]}")
                    else:
                        print("fail to match the json format!")
                        output_labels[idx] = [0,0,0,0]
                    
                except Exception as exc:
                    print(f'{inputs[idx]} generated an exception: {exc}')

        # for content in contents:
        #     print(f"==" *50)
        #     print(f"{content}")
        
        print(f'### {batch_idx}-th batch\'s output_labels:: length {len(output_labels)} ;; labels : {output_labels}')
        if len(output_labels) != batch_size:
            print(f"{batch_idx}-th batch's label output is not matching the original size !!!")

        # print(f'### {batch_idx}-th batch\'s unlabeled size: {output_labels.count(0)}')

        torch.save(output_labels, path + f"output_labels_{batch_idx}.pt")
        total_output_labels.extend(output_labels)

    from collections import Counter

    # assert len(inputs) == len(total_output_labels)
    print(f'total data size: {len(inputs)};; labeling data size: {len(total_output_labels)}')
    print(f'Total unlabeled data proportion: {np.array(total_output_labels)[:,-1].tolist().count(0)/len(inputs)*100}%')
    print(f'overal label score distribution: {Counter(np.array(total_output_labels)[:,-1].tolist())}')
    print("Finish API CALL labeling!!!")

    print(f"save labels to path: {path + f"total_output_labels.pt"}")

    torch.save(total_output_labels, path + f"total_output_labels.pt")

    ##############################################################################################################################
    '''single sample labeling'''


    # inputs = inputs[:10]
    # contents = []
    # output_labels = []
    # for idx, input in tqdm(enumerate(inputs)):
    #     messages= [{"role":"user","content": input}]

    #     completion = client.chat.completions.create(
    #     model=deployment_model, ##"meta/llama-3.1-8b-instruct",
    #     messages=messages,
    #     temperature=0.2,
    #     top_p=0.7,
    #     max_tokens=8192,
    #     stream=True
    #     )


    #     GPT_content ='### Rating:'
    #     for chunk in completion:
    #         # CHUNK form: ChatCompletionChunk(id='chatcmpl-9mpk2ZphIgvKsrBWcNkLzvLbMjE9a', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None)], created=1721425666, model='gpt-4-0613', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None)
    #         if chunk.choices[0].delta.content is not None:
    #             print(chunk.choices[0].delta.content, end="")
    #             GPT_content += chunk.choices[0].delta.content


    #     print('\n' + '='*100 +'\n')
    #     print(f"Input idx {idx}: {GPT_content}")
    #     print('\n' + '='*100 +'\n')
    #     match = re.search(r"### Rating:(\d+)", GPT_content)
    #     label = match.group(1) if match else -1

        
    #     contents.append(GPT_content)
    #     output_labels.append(label)


    # import pdb;pdb.set_trace()


        
    # torch.save(contents, path + "output_contents.pt")
    # torch.save(output_labels, path + "output_labels.pt")


    # print("Finishing GPT labeling!!!")


if __name__ == '__main__':
    fire.Fire(main)
    

# nohup python3 data_refine/labeling_api.py --dataset_name flan_v2 > GPT_labeling.log &
