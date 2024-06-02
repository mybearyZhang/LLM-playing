import torch
from thop import profile
from transformers import GPT2Tokenizer
from LongformerGPT2 import LongformerGPT2LMHeadModel, LongformerGPT2Config
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
SAMPLE_TEXT = ' '.join(['Hello world! '] * 8)  # long input document

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

flops, params = profile(model, inputs=(input_ids,))
print(f'FLOPs: {flops / 1024 ** 3} B')
print(f'Parameters (from profile): {params / 1024 ** 2} M')
