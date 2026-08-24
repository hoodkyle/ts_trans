"""Future explicit self-attention implementation."""

# Intended flow: X -> Q, K, V -> scores -> attention weights -> output
# TODO: implement attention with inspectable intermediate tensors.

import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, N, d_model)
        Q = self.W_q(x)  # (B, N, d_model)
        K = self.W_k(x)  # (B, N, d_model)
        V = self.W_v(x)  # (B, N, d_model)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        N = scores.size(-1)
        mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))     
        attention_weights = torch.softmax(scores, dim=-1)  # (B, N, N)
        output = torch.matmul(attention_weights, V)  # (B, N, d_model)

        return output, attention_weights