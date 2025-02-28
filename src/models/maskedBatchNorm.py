import torch.nn as nn
import torch

class MaskedBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        gamma = nn.Parameter(torch.tensors(1))
        beta = nn.Parameter(torch.tensor(1))
    def forward(self,x,mask):


        return