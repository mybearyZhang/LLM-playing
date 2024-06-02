import math
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embedding_dim))

    def forward(self, position_ids):
        return self.positional_encoding[position_ids].squeeze(0)


class learningGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super(learningGPT2, self).__init__(config)
        self.transformer.wpe = LearnablePositionalEncoding(config.n_positions, config.n_embd)
