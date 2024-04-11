import torch
from torch import nn


class VarDropout(nn.Module):
    def __init__(self, p=0.1):
        super(VarDropout, self).__init__()
        self.p = p

    def forward(self, input_sequence):
        rand_mask = torch.rand(
            (input_sequence.shape[::2]), requires_grad=True, device="cuda"
        )
        batch_mask = (rand_mask > self.p).int()
        expanded_mask = batch_mask.unsqueeze(1)
        full_mask = expanded_mask.repeat(1, input_sequence.shape[1], 1)

        return (input_sequence * full_mask) * (1.0 / (1.0 - self.p))
