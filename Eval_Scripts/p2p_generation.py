import json
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', '-data_path', type=str, required=True)
parser.add_argument('--model_path', '-model_path', type=str, required=True)
parser.add_argument('--output_path', '-output_path', type=str, required=True)

args = parser.parse_args()

def parse_conversation(conversation):
    # Regular expression pattern to match 'A:' and 'B:' with optional preceding newlines
    pattern = r'\n*\s*(A:|B:)'

    # Splitting the conversation using the regular expression pattern
    parts = re.split(pattern, conversation)

    # Lists to hold conversations of A and B
    a_dialogue, b_dialogue = [], []

    # Iterate through each part to segregate dialogues of A and B
    for i in range(1, len(parts), 2):
        speaker = parts[i]
        dialogue = parts[i+1].strip()
        if speaker == 'A:':
            a_dialogue.append(dialogue)
        elif speaker == 'B:':
            b_dialogue.append(dialogue)

    return a_dialogue, b_dialogue


# Example usage with multiple 'A:' and 'B:'

chat_data = []
original_dataset = []
first_follow_up_dataset = []
second_follow_up_dataset = []
# Replace 'your_file.jsonl' with the path to your JSON Lines file
file_path = args.data_path

# Using 'with' ensures the file is properly closed after reading
prompt2ques_dict = {}
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        one_chat_data = {'new_problem':''}
        json_object = json.loads(line.strip())
        question = json_object['question']
        answer = json_object['answer']
        #print(json_object['followup'])
        #print(len(b_dialogue))
        one_chat_data['new_problem'] = [
            {"role": "user", "content": 'Your task is to create a new math problem based on a given seed problem. The generated problems should either explore the same topic in greater depth or apply the same mathematical principles in a different context. Each problem should be accompanied by a detailed solution that demonstrates the correct application of the mathematical principles involved.'},
            {"role": "assistant", "content": 'Understood, please give me the seed problem.'},
            {"role": "user", "content": 'Seed problem: ' + question + ' Solution: ' + answer + '\n'},
        ]
        print(one_chat_data['new_problem'])
        one_chat_data['question'] = json_object['question']
        chat_data.append(one_chat_data)

first_output_file = open(args.output_path,'w')
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
llm = LLM(args.model_path, tensor_parallel_size=4, dtype=torch.float16)
first_prompt_list = [] #i['new_problem'] for i in chat_data
for i in chat_data:
    first_prompt = tokenizer.apply_chat_template(i['new_problem'], tokenize=False)
    first_prompt_list.append(first_prompt)
    prompt2ques_dict[first_prompt] = i['question']
sampling_params = SamplingParams(temperature=0.0, max_tokens = 1024)

outputs = llm.generate(first_prompt_list, sampling_params)

first_result = []
qa_dict = {}
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    first_result.append({'input': prompt, 'output':generated_text})
    qa_dict[prompt] = generated_text
    first_output_file.write(json.dumps({'input': prompt, 'output':generated_text, 'question': prompt2ques_dict[prompt]}))
    first_output_file.write('\n')
    first_output_file.flush()
