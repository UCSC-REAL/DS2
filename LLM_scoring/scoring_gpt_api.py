import regex as re
import torch
import fire
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np
import json
from collections import Counter
import os
from openai import OpenAI, AzureOpenAI



client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def main(
    deployment_model: str = 'gpt-4o-mini-2024-08-28',
    dataset_name: str = 'flan_v2',
    dataset_type: str='train',
    ):


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
    
    
    print("Preprocessing dataset...")

    data = load_dataset('json', data_files=f'data/train_data/{dataset_name}_data.jsonl', split=None, cache_dir=None)
    dialogs = data['train']
    
    inputs= []
    for dialog in dialogs:
        conversation = ""
        for message in dialog['messages']:  #format: [{{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
            conversation += f"### {message['role']}: {message['content']}\n"

        inputs.append(pre_prompt + conversation + "\n### Rating:")


    path = f"./logs-api/{deployment_model}/{dataset_name}/"
    if not os.path.exists(path):
        os.makedirs(path)


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
    
    batch_size = 1024 # batch_size
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

                    retry_count = 0  # retry count

                    max_retries=3
                    while not matches and retry_count < max_retries:
                        retry_count += 1
                        print(f"Retrying for input index {idx}, attempt {retry_count}")
                        idx, GPT_content = fetch_content(batch_inputs[idx], idx)
                        matches = json_pattern.findall(GPT_content)

                    if matches:
                        try:
                            # extract the json object
                            json_obj = json.loads(matches[-1])
                            output_labels[idx] = [
                                int(json_obj['Rarity']),
                                int(json_obj['Complexity']),
                                int(json_obj['Informativeness']),
                                int(json_obj['Overall rating'])
                            ]

                        except json.JSONDecodeError:
                            print(f"JSON Decode Error for inputs with {idx}")
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


    # assert len(inputs) == len(total_output_labels)
    print(f'total data size: {len(inputs)};; labeling data size: {len(total_output_labels)}')
    print(f'Total unlabeled data proportion: {np.array(total_output_labels)[:,-1].tolist().count(0)/len(inputs)*100}%')
    print(f'overal label score distribution: {Counter(np.array(total_output_labels)[:,-1].tolist())}')
    print("Finish API CALL labeling!!!")

    print(f"save labels to path: {path + f"total_output_labels.pt"}")

    torch.save(total_output_labels, path + f"total_output_labels.pt")



if __name__ == '__main__':
    fire.Fire(main)
    
