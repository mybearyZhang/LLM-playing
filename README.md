# LLM Playing

This project allows users to finetune Large Language Models (LLMs) on custom datasets and generate text using the finetuned model. The project is built using the Hugging Face Transformers library and the GPT-2 model.

## Table of Contents

- [Installation](#installation)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Exploration for Attention](#exploration-for-attention)

## Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/mybearyZhang/LLM-playing.git

# Navigate to the project directory
cd LLM-playing

# Install dependencies
pip install -r requirements.txt
```

## Finetuning
To run the fine-tuning script of the model, use the following command format:

```bash
python finetune.py [arguments]
```

Here are the available command-line arguments you can use:

+ `--model_name`: Pretrained model name or path (default: "gpt2").
+ `--dataset`: Dataset to fine-tune on (choices: "wiki2", "wiki103", default: "wiki2").
+ `--lr`: Learning rate (default: 1e-3).
+ `--weight_decay`: Weight decay (default: 1e-8).
+ `--warmup`: Warmup steps (default: 2000).
+ `--train_epoches`: Number of training epochs (default: 40).
+ `--batch_size`: Batch size (default: 8).
+ `--num_worker`: Number of workers for DataLoader (default: 4).
+ `--max_seq_length`: Maximum sequence length (default: 512).
+ `--out_model_path`: Output model path (default: "mygpt").
+ `--wandb`: Forbid Weights & Biases for logging.
+ `--dataset_name`: Subset name of the Chinese dataset.
+ `--answer_from`: Answer from the dataset (choices: "llm", "human", default for all).

Here is an example commands to fine-tune the GPT-2 model:
    
```bash
python finetune.py --model_name "gpt2" --dataset "wiki2" --lr 1e-2 --weight_decay 1e-8 --warmup 2000 --train_epoches 30 --batch_size 8 --num_worker 4 --max_seq_length 512 --out_model_path "mygpt"
```

## Evaluation
To run the evaluation script of the model, use the following command format:

```bash
python eval.py [arguments]
```

Here are the available command-line arguments you can use:

+ `--model_name`: Pretrained model name or path (default: "gpt2").
+ `--prompt`: Prompt for text generation, chosen from `['news', 'knowledge', 'story', 'conversation', 'poetry']`, chinese from `['chinese-story', 'chinese-ruozhiba', 'chinese-traditional', 'chinese-xhs', 'chinese-wiki']`.

Here is an example command to evaluate the GPT-2 model:

```bash
python eval.py --model_name "gpt2" --prompt "story"
```

Furthermore, we provide a script to evaluate the outputs of the model judged by a LLM. Before testing, fill the prompts and the outputs into the `prompt.json`. To run the script, use the following command format:

```bash
python llm_test.py
```

## Exploration for Attention
To run the exploration script of the model, use the following command format:

```bash
python main.py
```
The script will print out calculation time with input sequences of different lengths. You can also create a mixed pretrained model file.

```
python create_mix_model.py
```

