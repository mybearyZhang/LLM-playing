import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class COIGDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, name=None):
        if name is not None:
            self.data = load_dataset(data, name)['train']
        else:
            self.data = load_dataset("./COIG-CQIA")['train']
            raise NotImplementedError
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']

        # 将instruction和input拼接，并进行编码
        encoding = self.tokenizer(
            instruction + " " + input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = self.tokenizer(
            output_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 确保将数据从tensor转换为一维张量
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = labels['input_ids'].squeeze(0)  # 使用输出文本的input_ids作为标签

        return item
