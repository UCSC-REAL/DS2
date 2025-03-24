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



def score_compress(original_scores):
    original_scores = [score[-1] for score in original_scores] #take the overall rating scores
    print(f"Original score distribution{Counter(original_scores)}")

    #rematching to [0,1,2,3,4,5]
    scores_revised = []
    for score in original_scores:
        if score <= 4:
            scores_revised.append(4)
        elif score >= 9:
            scores_revised.append(9)
        else:
            scores_revised.append(score)

    scores_revised = [score - 4 for score in scores_revised] 
    print(f"Revised scores distribution {Counter(scores_revised)}\n")

    return scores_revised


def main(
    deployment_model: str = 'gpt-4o-mini-2024-08-28',
    dataset_name: str = 'flan_v2',
    output_dir="./logs/",
    ):


    pre_prompt = ('''
        As a data quality estimator, your task is to assess the quality of data sample based on the criteria: Rarity, Complexity, Informativeness.
        Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a scale from 1 to 10, where a higher score indicates higher level of quality.
        Ensure that the ratings are not overly concentrated around a specific score. If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences.
        Please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:
        {
            "Rarity": <number, 1-10>,
            "Complexity": <number, 1-10>,
            "Informativeness": <number, 1-10>,
            "Overall rating": <number, 1-10>
        }
        ''') 

    print(f"Rating Model: {deployment_model}")
    print(f"Dataset Name: {dataset_name} ")
    print("Preprocessing dataset...")
    
    if dataset_name == 'tulu_300k':
        dialogs = load_dataset('jlpang888/tulu_300k')['train']
    else:
        raise NotImplementedError
    
    inputs= []
    for dialog in dialogs:
        conversation = ""
        for message in dialog['messages']:  #format: [{{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
            conversation += f"### {message['role']}: {message['content']}\n"

        inputs.append(pre_prompt + conversation + "\n### Rating:")


    output_path = output_dir + f"{deployment_model}/{dataset_name}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    def fetch_content(input, idx):
        completion = client.chat.completions.create(
            model=deployment_model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input}],
            max_tokens=128)

        if completion.choices[0].message.content is not None:
            GPT_content = completion.choices[0].message.content
        return idx, GPT_content



    total_output_scores = []
                
    import concurrent.futures
    from tqdm import tqdm
    
    print(f"Total dataset size: {len(inputs)}")
    
    batch_size = 1024
    split_size = len(inputs)//batch_size + 1

    json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')


    for batch_idx in tqdm(range(split_size), desc=f'API Call rating model -- {deployment_model}'):
        batch_end =  min((batch_idx+1) * batch_size, len(inputs)) # the end index of batch
        batch_inputs = inputs[batch_idx* batch_size:batch_end] ##data range
        
        contents = [None] * len(batch_inputs)
        output_scores = [None] * len(batch_inputs) 
        
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
                            output_scores[idx] = [
                                int(json_obj['Rarity']),
                                int(json_obj['Complexity']),
                                int(json_obj['Informativeness']),
                                int(json_obj['Overall rating'])
                            ]

                        except json.JSONDecodeError:
                            print(f"JSON Decode Error for inputs with {idx}")
                    else:
                        print("fail to match the json format!")
                        output_scores[idx] = [0,0,0,0]
                    
                except Exception as exc:
                    print(f'{inputs[idx]} generated an exception: {exc}')

        
        if len(output_scores) != batch_size:
            print(f"{batch_idx}-th batch's score output is not matching the original size !!!")


        torch.save(output_scores, output_path + f"output_scores_{batch_idx}.pt")
        total_output_scores.extend(output_scores)


    print(f'Total data size: {len(inputs)};; Scoring data size: {len(total_output_scores)}')
    print(f'Total unrated data proportion: {np.array(total_output_scores)[:,-1].tolist().count(0)/len(inputs)*100}%')
    print(f'Overall score score distribution: {Counter(np.array(total_output_scores)[:,-1].tolist())}')
    print("Finish API call scoring!!!")

    print(f"save scores to path: {output_path}total_output_scores.pt")

    ## rematching raw score to [0,1,2,3,4,5]
    output_scores_revised = score_compress(total_output_scores)
    
    torch.save(total_output_scores, output_path + f"output_scores.pt")
    torch.save(output_scores_revised, output_path + f"output_scores_revised.pt")


if __name__ == '__main__':
    fire.Fire(main)
    
