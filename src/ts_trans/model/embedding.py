"""Future input representation and positional/time embedding components."""

# Intended shape: (B, N, 1) -> (B, N, d_model)
# TODO: implement the scalar input representation.
import torch
from torch import nn
import math

class ScalarEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_model, d_model)
    def forward(self, x):
        # x: (B, N, 1)
        x = self.linear1(x)  # (B, N, d_model)
        x = self.activation(x)
        x = self.linear2(x)  # (B, N, d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len]