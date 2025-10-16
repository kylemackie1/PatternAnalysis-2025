"""
Neural Network Modules for Alzheimer's Disease Classification
Using ConvNeXt based architecture for brain MRI classification
"""

import torch
import torch.nn as nn

class LayerNorm2D(nn.Module):
    """2D Layer Normalization for ConvNeXt blocks"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class DropPath(nn.Module):
    """
    Drop entire samples from path to prevent overfitting
    Samples are dropped entirely with drop_prob chance    
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        # Sets tensor to all 0s or all 1s with drop_prob or keep_prob chance respectively
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output