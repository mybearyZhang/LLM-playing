from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset

import torch
import math
import os
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from time import time

start_time = time()

tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
model = GPT2LMHeadModel.from_pretrained('./gpt2_model')

param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
total_params = sum(param_sizes)
print(f"Total number of parameters: {total_params / 1024 ** 2}M")

tokenizer.pad_token = tokenizer.eos_token

# 导入模型checkpoint
model = GPT2LMHeadModel.from_pretrained('./checkpoints/epoch40')
model.to('cuda')


# Story
input_text = "In a distant kingdom, a young wizard discovered an ancient book of spells. One day, he decided to try one of the spells from the book, and as a result..."

# Conversation
input_text = "Reporter: Can you tell us how you began your career as an artist? Artist: It all started with a chance encounter when I was a child..."

# Poetry
input_text = "The morning sun casts its glow upon the lake, a gentle breeze stirs, carrying the scent of blossoms. The longing in my heart, like the ripples on the water..."

input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

max_length = 50
stop_token = '.'  # tokenizer.eos_token  # 停止生成的标记，通常是'<|endoftext|>'或'.'等

# 使用模型生成预测
output_sequences = []

for i in range(max_length):
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]

    predicted_id = torch.argmax(predictions[:, -1], dim=-1).item()
    predicted_token = tokenizer.decode(predicted_id)

    input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]]).to('cuda')], dim=-1)
    output_sequences.append(predicted_token)
    if predicted_token == stop_token:
        break


print(f"Input: {input_text}")
print(f"Generated output: {''.join(output_sequences)}")
