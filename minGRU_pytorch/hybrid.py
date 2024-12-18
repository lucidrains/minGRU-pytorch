import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from minGRU_pytorch.minGRU import minGRU

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

# hybrid minGRU and attention, following the same design as Hymba
# Hymba split the features into two, carried out associative scan RNN + attention on separate branches, followed by norm, scale for both then averaged, projected out

class minGRUAttnHybrid(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        learned_mix = True
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_mix = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, heads, bias = False)) if learned_mix else None

        self.rnn = minGRU(dim, expansion_factor = dim_inner / dim, proj_out = False)

        self.rnn_out_norm = nn.RMSNorm(dim_head)
        self.attn_out_norm = nn.RMSNorm(dim_head)

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        # minGRU branch

        rnn_out = self.rnn(x)

        rnn_out = rearrange(rnn_out, 'b n (h d) -> b h n d', h = self.heads)

        # attention branch

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        # in paper, they simply averaged the two branches

        mix = 0.5

        if exists(self.to_mix):
            # maybe learned per-token / head mixing
            mix = self.to_mix(x).sigmoid()
            mix = rearrange(mix, 'b n h -> b h n 1')

        # the scheme for hybridizing is normalizing + scaling each branch then averaging


        out = mix * (self.rnn_out_norm(rnn_out) + (1. - mix) * self.attn_out_norm(attn_out))

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
