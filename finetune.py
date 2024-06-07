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
import argparse
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from datasets import load_dataset
from utils.dataset import COIGDataset
from utils.learningGPT2 import learningGPT2


def main(args):
    if args.wandb:
        wandb.init(project='GPT2 finetune', config={"x-axis": "epoch"})

    if args.model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
        model = GPT2LMHeadModel.from_pretrained('./gpt2_model')
    elif args.model_name == "chinese":
        tokenizer = AutoTokenizer.from_pretrained('./gpt2-chinese-cluecorpussmall')
        model = GPT2LMHeadModel.from_pretrained('./gpt2-chinese-cluecorpussmall')
    elif args.model_name == "learning":
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
        model = learningGPT2.from_pretrained('./gpt2_model', config=GPT2Config())
    else:
        raise NotImplementedError

    param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
    total_params = sum(param_sizes)
    print(f"Total number of parameters: {total_params / 1024 ** 2}M")
    
    lr = args.lr
    weight_decay = args.weight_decay
    warmup = args.warmup
    train_epoches = args.train_epoches
    batch_size = args.batch_size
    num_worker = args.num_worker
    max_seq_length = args.max_seq_length
    out_model_path = args.out_model_path
    
    # 确认并设置填充令牌
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.dataset.startswith("wiki"):
        if args.dataset == "wiki2":
            train_file = "wikitext-2/train.csv"
            eval_file = "wikitext-2/test.csv"
        elif args.dataset == "wiki103":
            train_file = "wikitext-103/train.csv"
            eval_file = "wikitext-103/test.csv"
        
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=train_file,
            block_size=max_seq_length,
        )
        eval_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=eval_file,
            block_size=max_seq_length,
        )
    elif args.dataset == "chinese":
        dataset = COIGDataset(data='./COIG-CQIA', tokenizer=tokenizer, max_length=max_seq_length, name=args.dataset_name, answer_from=args.answer_from)
        
        # split the dataset into train and eval
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
    else:
        raise NotImplementedError

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Test dataset: {len(eval_dataset)}")

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
        if args.wandb:
            wandb.log({'Training Loss': loss / len(train_loader), 'Training Perplexity': train_perplexity, 'Validation Loss': np.mean(eval_loss), 'Perplexity': np.exp(np.mean(eval_loss)), 'Epoch': epoch + 1})

        eval_loss = np.mean(eval_loss)
        perplexity = np.exp(eval_loss)
        print(f"Validation loss: {eval_loss} in epoch {epoch + 1}.")
        print(f"Perplexity: {perplexity} in epoch {epoch + 1}.")
        
        if args.save_steps > 0 and (epoch + 1) % args.save_steps == 0:
            ckp_path = os.path.join(out_model_path, 'checkpoints', f"epoch{epoch + 1}")
            if not os.path.exists(ckp_path):
                os.makedirs(ckp_path)
            model.save_pretrained(ckp_path)
            print(f"Model saved to {ckp_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune GPT-2 model")
    
    parser.add_argument("--model_name", type=str, help="Pretrained model name or path.")
    parser.add_argument("--dataset", type=str, help="Dataset to finetune on. ['wiki2', 'wiki103']")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument("--warmup", type=float, default=2000, help="Warmup steps")
    parser.add_argument("--train_epoches", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_worker", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=0, help="Save model every n steps, 0 to disable")
    parser.add_argument("--out_model_path", type=str, default="mygpt", help="Output model path")
    parser.add_argument("--wandb", action='store_false', help="Use wandb or not")
    parser.add_argument("--dataset_name", type=str, help="Subset for Chinese Dataset")
    parser.add_argument("--answer_from", type=str, help="Answer from LLM or human")
    
    args = parser.parse_args()
    
    main(args)