# https://arxiv.org/abs/2410.01201v1

import torch
from torch.nn import Linear, Module

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan(gate, query):
    eps = 1e-7 if gate.dtype == torch.float16 else 1e-20
    log_gate = gate.clamp(min = eps).log()
    log_query = query.clamp(min = eps).log()

    a_star = log_gate.cumsum(dim = 1)
    log_h0_plus_b_star = (log_query - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# min GRU

class minGRU(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_query_and_gate = Linear(dim, dim * 2, bias = False)

    def forward(self, x):
        query, gate = self.to_query_and_gate(x).chunk(2, dim = -1)
        out = heinsen_associative_scan(gate.sigmoid(), query)
        return out
