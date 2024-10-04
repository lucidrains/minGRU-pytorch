# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_hidden_and_gate = Linear(dim, dim * 2, bias = False)

    def forward(self, x, prev_hidden = None):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        # handle sequential

        if seq_len == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            return torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)

        # parallel

        log_coeffs = -F.softplus(gate)

        log_z = -F.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h

        if exists(prev_hidden):
            log_values = torch.cat((log_g(prev_hidden), log_values), dim = 1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

        out = heinsen_associative_scan_log(log_coeffs, log_values)
        return out[:, -seq_len:]
