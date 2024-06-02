import torch
from torch import nn
from transformers import GPT2Config, GPT2Tokenizer
from learningGPT2 import learningGPT2

config = GPT2Config()
model = learningGPT2.from_pretrained('../gpt2_model/', config=config)
tokenizer = GPT2Tokenizer.from_pretrained('../gpt2_model/')

SAMPLE_TEXT = ' '.join(['Hello world! '] * 256)
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)
labels = input_ids.clone()

outputs = model(input_ids, labels=labels)
print(outputs[0].item())
