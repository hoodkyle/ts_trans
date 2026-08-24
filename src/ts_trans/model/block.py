"""Future transformer block combining attention and feed-forward stages."""

# Intended flow: attention -> residual -> feed-forward -> residual
# TODO: implement one transparent transformer block.
from .attention import SelfAttention
from .feedforward import FeedForward
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Attention block with residual connection
        attn_output, attn_weights = self.attention(self.norm1(x))
        x = x + attn_output

        # Feed-forward block with residual connection
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        return x, attn_weights