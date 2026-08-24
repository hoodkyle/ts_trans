"""forecast head for producing a one-step scalar prediction."""

from torch import nn

from .embedding import ScalarEmbedding, PositionalEncoding
from .block import TransformerBlock



class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_blocks: int):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff)
            for _ in range(n_blocks)
        ])

        self.embedding = ScalarEmbedding(d_model)
        self.position = PositionalEncoding(d_model)

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        for block in self.blocks:
            x, _ = block(x)
        last = x[:, -1, :]
        forecast = self.head(last)
        return forecast
