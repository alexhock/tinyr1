import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
import random
import re 
from datasets import load_dataset


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "openai/gsm8k"
    dataset_splits: str = "main"  # socratic
    tokenizer_name_or_path: str = None


script_args = ScriptArguments()


###############
# Load datasets
###############
# Load dataset from Hugging Face Hub
dataset = load_dataset(script_args.dataset_id_or_path, name="main")
# select a random subset of 50k samples
dataset = dataset['train'].shuffle(seed=42).select(range(50))

#####################
# Prepare and format dataset
#####################


# gemerate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(question, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
        "role": "user",
        "content": f"Using the maths problem question text: '{question}', find the equation that equals the answer to the problem question text, in this case: {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 16 </answer>. Think step by step inside <think> tags."
        },
        {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
        }]
    return {"prompt": r1_prefix, "target": target, "question": question}

# Define a function to process the 'question' and 'answer' columns
def extract_text_after_delimiter(example):
    for column in ['answer']:
        if '####' in example[column]:
            example[column] = example[column].split('####', 1)[1].strip()
        else:
            example[column] = ''
    return example


# Apply the function to the dataset
processed_dataset = dataset.map(extract_text_after_delimiter)

q, target = processed_dataset[0]


# convert our dataset to the r1 prompt
dataset = processed_dataset.map(lambda x: generate_r1_prompt(x["question"], x["answer"]))

# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]


