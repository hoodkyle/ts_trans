"""Future input representation and positional/time embedding components."""

# Intended shape: (B, N, 1) -> (B, N, d_model)
# TODO: implement the scalar input representation.
import torch
from torch import nn

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
    
