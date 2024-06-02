import torch
import math
import os
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from time import time
import argparse

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset

def main(args):
    if args.model_name == "chinese":
        tokenizer = AutoTokenizer.from_pretrained('./gpt2-chinese-cluecorpussmall')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
    
    model_name = args.model_name
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('./gpt2_model')
    elif model_name == "wiki2":
        model = GPT2LMHeadModel.from_pretrained('./epoch40')
    elif model_name == "wiki103":
        model = GPT2LMHeadModel.from_pretrained('./runs')
    elif model_name == "chinese":
        model = GPT2LMHeadModel.from_pretrained('./gpt2-chinese-cluecorpussmall')

    param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
    total_params = sum(param_sizes)
    print(f"Total number of parameters: {total_params / 1024 ** 2}M")

    tokenizer.pad_token = tokenizer.eos_token

    model.to('cuda')

    # 定义各个类别的 input_text
    input_texts = {
        'story': "In a distant kingdom, a young wizard discovered an ancient book of spells. One day, he decided to try one of the spells from the book, and as a result...",
        'conversation': "Reporter: Can you tell us how you began your career as an artist? Artist: It all started with a chance encounter when I was a child...",
        'poetry': "The morning sun casts its glow upon the lake, a gentle breeze stirs, carrying the scent of blossoms. The longing in my heart, like the ripples on the water...",
        'news': "Tech giant Apple has just unveiled its latest innovation:...",
        'knowledge': "The theory of evolution, first proposed by Charles Darwin, suggests that...",
        # 'chinese': "在一个遥远的国度，有一位年轻的法师发现了一本古老的咒语书。有一天，他决定尝试书中的一个咒语，结果",
        'chinese': "在一个遥远的国度，有一位年轻的法师发现了一本古老的咒语书。有一天，他突然突然发现孙雨林是神，结果",
    }
    
    # 获取选择的 input_text
    selected_text = input_texts[args.prompt]

    input_ids = tokenizer.encode(selected_text, return_tensors='pt').to('cuda')

    max_length = 50
    stop_token = '.'  # tokenizer.eos_token  # 停止生成的标记，通常是'<|endoftext|>'或'.'等

    # 使用模型生成预测
    output_sequences = []

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs[0]

        predicted_id = torch.argmax(predictions[:, -1], dim=-1).item()
        predicted_token = tokenizer.decode(predicted_id)

        input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]]).to('cuda')], dim=-1)
        output_sequences.append(predicted_token)
        if predicted_token == stop_token:
            break

    print(f"Input: {selected_text}")
    print(f"Generated output: {''.join(output_sequences)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model")
    
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path. ['gpt2', 'wiki2', 'wiki103']")
    parser.add_argument("--prompt", type=str, default="story", help="The category of input text you want to select. ['story', 'conversation', 'poetry']")
    
    args = parser.parse_args()
    
    main(args)