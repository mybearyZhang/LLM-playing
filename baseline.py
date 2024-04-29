import argparse
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_gpt2(args):
    # 加载预训练的GPT-2 tokenizer和模型
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model_config = GPT2Config.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name, config=model_config)

    # 加载数据集
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.train_data,
        block_size=args.block_size
    )
    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.eval_data,
        block_size=args.block_size
    )

    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
    )

    # 创建Trainer对象
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # 开始微调
    trainer.train()

    # 保存微调后的模型
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned model")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite the output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for data")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save model checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints saved")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluate the model every X steps")
    parser.add_argument("--logging_steps", type=int, default=2000, help="Log training information every X steps")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")

    args = parser.parse_args()

    train_gpt2(args)
