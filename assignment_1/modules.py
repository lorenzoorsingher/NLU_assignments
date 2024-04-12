import torch
from torch import nn


class VarDropout(nn.Module):
    def __init__(self, p=0.1):
        super(VarDropout, self).__init__()
        self.p = p

    def forward(self, x):

        if not self.training:
            return x

        rand_mask = torch.rand((x.shape[::2]), requires_grad=True, device="cuda")
        expanded_mask = (rand_mask > self.p).int().unsqueeze(1)
        full_mask = expanded_mask.repeat(1, x.shape[1], 1)

        return (x * full_mask) * (1.0 / (1.0 - self.p))
