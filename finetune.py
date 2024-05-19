import torch
import math
import os
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from time import time
import wandb
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset


# 初始化wandb，填入你的API密钥和项目名称
wandb.init(project='GPT2 finetune', config={"x-axis": "epoch"})


start_time = time()

tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
model = GPT2LMHeadModel.from_pretrained('./gpt2_model')

param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
total_params = sum(param_sizes)
print(f"Total number of parameters: {total_params / 1024 ** 2}M")


# train_file = "wikitext-103-raw/wiki.train.raw"
# eval_file = "wikitext-103-raw/wiki.valid.raw"
train_file = "wikitext-2/train.csv"
eval_file = "wikitext-2/test.csv"
max_seq_length = 512
out_model_path = "mygpt"
train_epoches = 40
batch_size = 8
num_worker = 4
lr = 1e-3
weight_decay = 1e-8
warmup = 2000

tokenizer.pad_token = tokenizer.eos_token

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=max_seq_length,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=max_seq_length,
)

print(f"Train dataset: {len(train_dataset)}")
print(f"Train dataset: {len(eval_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=data_collator)
valid_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=data_collator)

# model = GPT2LMHeadModel.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=weight_decay)
total_steps = len(train_loader) * train_epoches
warmup_steps = int(total_steps * warmup)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# start training
model.to('cuda')

for epoch in range(train_epoches):
    loss = 0
    model.train()
    model.zero_grad()

        # 在训练循环中
    for batch in tqdm(train_loader, desc=f'training: epoch {epoch + 1} of {train_epoches}'):
        # batch里是input_ids, attention_mask, label
        ids = batch['input_ids'].to('cuda')
        mask = batch['attention_mask'].to('cuda')
        label = batch['labels'].to('cuda')
        
        output = model(input_ids=ids, attention_mask=mask, labels=label)
        batch_loss = output[0]
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        loss += batch_loss.item()

    # 在训练循环结束后，记录训练损失到wandb
    train_perplexity = np.exp(loss / len(train_loader))

    # 验证循环
    eval_loss = []
    model.eval()
    # 在验证循环中
    for batch in tqdm(valid_loader, desc=f'evaluating: epoch {epoch + 1} of {train_epoches}'):
        ids = batch['input_ids'].to('cuda')
        mask = batch['attention_mask'].to('cuda')
        label = batch['labels'].to('cuda')
        
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=mask, labels=label)
            batch_loss = output[0]
        
        eval_loss.append(batch_loss.cpu().item())

    # 在验证循环结束后，记录验证损失到wandb
    wandb.log({'Training Loss': loss / len(train_loader), 'Training Perplexity': train_perplexity, 'Validation Loss': np.mean(eval_loss), 'Perplexity': np.exp(np.mean(eval_loss)), 'Epoch': epoch + 1})

    eval_loss = np.mean(eval_loss)
    perplexity = np.exp(eval_loss)
    print(f"Validation loss: {eval_loss} in epoch {epoch + 1}.")
    print(f"Perplexity: {perplexity} in epoch {epoch + 1}.")
    
    if (epoch + 1) % 5 == 0:
        ckp_path = os.path.join('checkpoints', f"epoch{epoch + 1}")
        if not os.path.exists(ckp_path):
            os.makedirs(ckp_path)
        model.save_pretrained(ckp_path)
        print(f"Model saved to {ckp_path}.")
