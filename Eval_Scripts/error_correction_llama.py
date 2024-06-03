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

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
llm = LLM(args.model_path, tensor_parallel_size=4, dtype=torch.float16, trust_remote_code=True)
first_output_file = open(args.output_path,'w')

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
prompt2ques_dict = {}
prompt2ans_dict = {}
file_path = args.data_path
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        one_chat_data = {'error_correction':''}
        # Parse the JSON object in this line
        json_object = json.loads(line.strip())
        a_dialogue, b_dialogue = parse_conversation(json_object['error_correction'])

        if len(a_dialogue) == 2:
            one_chat_data['error_correction'] = [
                {"role": "user", "content": a_dialogue[0]},
                {"role": "assistant", "content": b_dialogue[0]},
                {"role": "user", "content": a_dialogue[1]},
            ]
            chat_data.append(one_chat_data)
            one_chat_data['question'] = json_object['question']
            one_chat_data['answer'] = json_object['answer']
            #print(one_chat_data)
        else:
            continue

first_prompt_list = [] #i['error_correction'] for i in chat_data
for i in chat_data:
    first_prompt = tokenizer.apply_chat_template(i['error_correction'], tokenize=False, add_generation_prompt = True)
    first_prompt_list.append(first_prompt)
    prompt2ques_dict[first_prompt] = i['question']
    prompt2ans_dict[first_prompt] = i['answer']
sampling_params = SamplingParams(temperature=0.0, max_tokens = 300)

outputs = llm.generate(first_prompt_list, sampling_params)

first_result = []
qa_dict = {}
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    first_result.append({'input': prompt, 'output':generated_text})
    qa_dict[prompt] = generated_text
    first_output_file.write(json.dumps({'input': prompt, 'output':generated_text, 'question': prompt2ques_dict[prompt], 'answer': prompt2ans_dict[prompt]}))
    first_output_file.write('\n')
    first_output_file.flush()
