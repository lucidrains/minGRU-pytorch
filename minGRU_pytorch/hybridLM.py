import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from minGRU_pytorch.hybrid import minGRUAttnHybrid

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class hybridLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                nn.RMSNorm(dim),
                minGRUAttnHybrid(dim, dim_head = dim_head, heads = heads),
                nn.RMSNorm(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        return_loss = False,
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for norm, hybrid, ff_norm, ff in self.layers:

            # hybrid attn + min gru

            x = hybrid(norm(x)) + x

            # feedforward

            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss
