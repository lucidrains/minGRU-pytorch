import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from minGRU_pytorch.minGRU import minGRU

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

# main class

class minGRULM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        min_gru_expansion = 1.5,
        conv_kernel_size = 3
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size),
                RMSNorm(dim),
                minGRU(dim, expansion_factor = min_gru_expansion),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        return_loss = False
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for conv, norm, mingru, ff_norm, ff in self.layers:

            x = conv(x) + x

            x = mingru(norm(x)) + x

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
