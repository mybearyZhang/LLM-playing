from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import GPT2Tokenizer, GPT2Model, AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset

import torch
import math


tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_model')
model = GPT2LMHeadModel.from_pretrained('./gpt2_model')

param_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
total_params = sum(param_sizes)
print(f"Total number of parameters: {total_params / 1024 ** 2}M")


train_file = "wikitext-103-raw/wiki.train.raw"
eval_file = "wikitext-103-raw/wiki.valid.raw"
max_seq_length = 512
out_model_path = "mygpt"
train_epoches = 10
batch_size = 10

tokenizer.pad_token = tokenizer.eos_token

dataset = LineByLineTextDataset(
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

training_args = TrainingArguments(
    output_dir=out_model_path,
    overwrite_output_dir=True,
    num_train_epochs=train_epoches,
    per_device_train_batch_size=batch_size,
    save_steps=2000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# trainer.train()
# trainer.save_model(out_model_path)

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# input_text = "It's a good day"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
#
# max_length = 50
# stop_token = '.'  # tokenizer.eos_token  # 停止生成的标记，通常是'<|endoftext|>'或'.'等
#
# # 使用模型生成预测
# output_sequences = []
#
# for i in range(max_length):
#     with torch.no_grad():
#         outputs = model(input_ids)
#         predictions = outputs[0]
#
#     predicted_id = torch.argmax(predictions[:, -1], dim=-1).item()
#     predicted_token = tokenizer.decode(predicted_id)
#
#     input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]])], dim=-1)
#     output_sequences.append(predicted_token)
#     if predicted_token == stop_token:
#         break
#
#
# print(f"Input: {input_text}")
# print(f"Generated output: {' '.join(output_sequences)}")
