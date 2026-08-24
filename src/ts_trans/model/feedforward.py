""" position-wise feed-forward network implementation."""


from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        x = self.linear1(x)  # (B, N, d_ff)
        x = self.activation(x)
        x = self.linear2(x)  # (B, N, d_model)
        return x