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
parser.add_argument('--second_output_path', '-second_output_path', type=str, required=True)
parser.add_argument('--third_output_path', '-third_output_path', type=str, required=True)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

def extract_question_alternative(text):
    # Regular expression pattern to find the desired string
    pattern = r'\[INST\](.*?)\[/INST\]'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # If a match is found, return the matched group, otherwise return None
    return match.group(1) if match else None

def extract_question_alternative_gemma(text):
    # Regular expression pattern to find the desired string
    pattern = r'<\|im_start\|>user(.*?)<\|im_end\|>'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # If a match is found, return the matched group, otherwise return None
    return match.group(1) if match else None

def extract_question_alternative_gemma_it(text):
    # Regular expression pattern to find the desired string
    pattern = r'<start_of_turn>user(.*?)<end_of_turn>'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # If a match is found, return the matched group, otherwise return None
    return match.group(1) if match else None

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
instruction = 'Solve this problem: '
chat_data = []
original_dataset = []
first_follow_up_dataset = []
second_follow_up_dataset = []
file_path = args.data_path
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        one_chat_data = {'original':'', 'first_follow_up':'', 'second_follow_up':''}
        # Parse the JSON object in this line
        json_object = json.loads(line.strip())
        a_dialogue, b_dialogue = parse_conversation(json_object['followup'])
        one_chat_data['original'] = instruction + json_object['question']
        #print(json_object['followup'])
        #print(len(b_dialogue))
        if len(a_dialogue) == 2:
            one_chat_data['first_follow_up'] = [
                {"role": "user", "content": instruction + json_object['question']},
                {"role": "assistant", "content": 'PLACE_HOLDER'},
                {"role": "user", "content": a_dialogue[0]},
            ]
            one_chat_data['second_follow_up'] = [
                {"role": "user", "content": instruction + json_object['question']},
                {"role": "assistant", "content": 'PLACE_HOLDER'},
                {"role": "user", "content": a_dialogue[0]},
                {"role": "assistant", "content": 'PLACE_HOLDER'},
                {"role": "user", "content": a_dialogue[1]},
            ]
            chat_data.append(one_chat_data)
        else:
            continue

first_prompt_list = []
for i in chat_data:
    first_prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": i['original']},
            ], tokenize=False, add_generation_prompt=True)

    #print(first_prompt)
    #print(extract_question_alternative_gemma_it(first_prompt))
    first_prompt_list.append(first_prompt)
sampling_params = SamplingParams(temperature=0.0, max_tokens = 400)
llm = LLM(args.model_path, tensor_parallel_size=4, dtype=torch.float16, trust_remote_code=True)
outputs = llm.generate(first_prompt_list, sampling_params)
first_output_file = open(args.output_path,'w')
first_result = []
qa_dict = {}
for output in outputs:
    prompt = output.prompt
    if 'gemma' in args.model_path:
        prompt = extract_question_alternative_gemma_it(prompt).strip()
    elif 'mistral' in args.model_path.lower():
        prompt = extract_question_alternative(prompt).strip()
    generated_text = output.outputs[0].text
    first_result.append({'input': prompt, 'output':generated_text})
    qa_dict[prompt] = generated_text
    first_output_file.write(json.dumps({'input': prompt, 'output':generated_text}))
    first_output_file.write('\n')
    first_output_file.flush()



second_temp_list = [i['first_follow_up'] for i in chat_data]
for idx in range(len(second_temp_list)):
    second_temp_list[idx][1]['content'] = qa_dict[second_temp_list[idx][0]['content']]
second_prompt_list = []
second_to_one_dict = {}
for i in second_temp_list:
    second_prompt = tokenizer.apply_chat_template(i, tokenize=False, add_generation_prompt=True)
    second_prompt_list.append(second_prompt)
    second_to_one_dict[second_prompt] = i[0]['content']

outputs = llm.generate(second_prompt_list, sampling_params)
second_output_file = open(args.second_output_path,'w')
second_result = []
second_qa_dict = {}
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    second_qa_dict[second_to_one_dict[prompt]] = generated_text
    second_result.append({'input': prompt, 'output':generated_text})
    second_output_file.write(json.dumps({'input': prompt, 'output':generated_text}))
    second_output_file.write('\n')
    second_output_file.flush()

third_temp_list = [i['second_follow_up'] for i in chat_data]
third_prompt_list = []
for idx in range(len(third_temp_list)):
    third_temp_list[idx][1]['content'] = qa_dict[third_temp_list[idx][0]['content']]
    third_temp_list[idx][3]['content'] = second_qa_dict[third_temp_list[idx][0]['content']]
for i in third_temp_list:
    third_prompt = tokenizer.apply_chat_template(i, tokenize=False, add_generation_prompt=True)
    third_prompt_list.append(third_prompt)

outputs = llm.generate(third_prompt_list, sampling_params)
third_output_file = open(args.third_output_path,'w')
third_result = []
third_qa_dict = {}
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    third_result.append({'input': prompt, 'output':generated_text})
    third_output_file.write(json.dumps({'input': prompt, 'output':generated_text}))
    third_output_file.write('\n')
    third_output_file.flush()
