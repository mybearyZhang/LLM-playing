import torch
from thop import profile
from transformers import GPT2Tokenizer
from LongformerGPT2 import LongformerGPT2LMHeadModel, LongformerGPT2Config, LongformerSelfAttention, SelfAttention
from sliding_chunks import pad_to_window_size

config = LongformerGPT2Config.from_pretrained('./longformer_gpt2_model/')
# choose the attention mode 'n2', or 'sliding_chunks'
#               'n2': for regular n2 attantion
#   'sliding_chunks': a PyTorch implementation of our sliding window attention
config.attention_mode = 'sliding_chunks'
# config.attention_mode = 'n2'
assert config.attention_mode == 'sliding_chunks' or config.attention_mode == 'n2'
print("Attention Mode:", config.attention_mode)

if config.attention_mode == 'sliding_chunks':
    model = LongformerGPT2LMHeadModel.from_pretrained('./longformer_gpt2_model/', config=config)
else:  # config.attention_mode == n2
    model = LongformerGPT2LMHeadModel.from_pretrained('./gpt2_model/', config=config)
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model/')
tokenizer.model_max_length = 4096

# GPT2 pretrained seq max length = 1024, thus too long input is invalid
SAMPLE_TEXT = ' '.join(['Hello world! '] * 64)  # long input document

input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to local attention
attention_mask[:, [1, 4, 21,]] = 2  # Set global attention based on the task. For example,
                                      # classification: the <s> token
                                      # QA: question tokens

# padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
input_ids, attention_mask = pad_to_window_size(input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)
labels = input_ids.clone()

# check range of <input_ids>
if input_ids.max() >= tokenizer.vocab_size:
    print("ERROR: input_ids contains values out of vocabulary range")

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
print(outputs[0].item())


import time
SEQ_LEN = [16, 32, 64, 128, 256, 512, 1024]

print("Longformer")
for seq_len in SEQ_LEN:
    SAMPLE_TEXT = ' '.join(['Hello world! '] * seq_len)
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)
    attention = LongformerSelfAttention(config, 0)
    inputs = model.transformer.wte(input_ids)
    inputs = torch.cat([inputs, inputs], dim=0)
    start = time.time()
    attention(inputs)
    end = time.time()
    print(f'SeqLen = {seq_len * 4}, Time: {end - start} s')

print("GPT-2")
for seq_len in SEQ_LEN:
    SAMPLE_TEXT = ' '.join(['Hello world! '] * seq_len)
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)
    attention = SelfAttention(n_head=12, d_k=768, d_v=768, d_x=768, d_o=768)
    inputs = model.transformer.wte(input_ids)
    inputs = torch.cat([inputs, inputs], dim=0)
    start = time.time()
    attention(inputs)
    end = time.time()
    print(f'SeqLen = {seq_len * 4}, Time: {end - start} s')
