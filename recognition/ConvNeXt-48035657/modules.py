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
    
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise convolution and inverted bottleneck"""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise 7x7 convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm
        self.norm = LayerNorm2D(dim)
        # Pointwise 1x1 convolution to expand channels (inverted bottleneck)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        # GELU activation
        self.act = nn.GELU()
        # Pointwise 1x1 convolution to project back
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # Layer scale parameter for better training stability
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim, 1, 1), 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        # Depthwise convolution
        x = self.dwconv(x)
        # Normalization
        x = self.norm(x)
        # Pointwise expansion
        x = self.pwconv1(x)
        # Activation
        x = self.act(x)
        # Pointwise projection
        x = self.pwconv2(x)
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        # Residual connection with drop path
        x = input + self.drop_path(x)
        return x
    
class ConvNeXt(nn.Module):
    """
    ConvNeXt architecture built based on the report and code
    provided in "A ConvNet for the 2020s" (https://arxiv.org/abs/2201.03545)
    """
    def __init__(self, in_channels=1, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.0):
        super().__init__()
        
        # Stem: aggressive downsampling with 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2D(dims[0])
        )
        
        # Build 4 stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            # Downsampling layer between stages (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm2D(dims[i-1]),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Stack ConvNeXt blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Final normalization
        self.norm = LayerNorm2D(dims[-1])
        self.feature_dim = dims[-1]
    
    def forward(self, x):
        """Extract features from input"""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        # Global average pooling
        x = x.mean([-2, -1])  # (N, C, H, W) -> (N, C)
        return x
    
class AlzheimerClassifier(nn.Module):
    """
    ConvNeXt-based classifier for Alzheimer's disease detection
    Handles 3D MRI volumes by processing multiple 2D slices
    """
    def __init__(self, num_classes=2, dropout=0.5, num_slices=16):
        super().__init__()
        self.num_slices = num_slices
        
        # Build ConvNeXt backbone (feature extractor only)
        # Using ConvNeXt-Tiny architecture: depths=[3,3,9,3], dims=[96,192,384,768]
        self.backbone = ConvNeXt(
            in_channels=1,  # Grayscale MRI
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1
        )
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.feature_dim  # 768
        
        # Attention mechanism for slice aggregation
        self.slice_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
        
        # Final classification head for 2 classes (Normal vs AD)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, slices, height, width) or (batch, 1, depth, height, width)
        batch_size = x.shape[0]
        
        # Handle different input formats
        if len(x.shape) == 5:  # (batch, 1, depth, height, width)
            x = x.squeeze(1)  # (batch, depth, height, width)
        
        # Sample slices if we have more than needed
        if x.shape[1] > self.num_slices:
            indices = torch.linspace(0, x.shape[1] - 1, self.num_slices).long()
            x = x[:, indices, :, :]
        
        num_slices = x.shape[1]
        
        # Reshape to process all slices: (batch * slices, 1, height, width)
        x = x.unsqueeze(2)  # Add channel dimension
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        
        # Extract features from all slices
        features = self.backbone(x)  # (batch * slices, 768)
        
        # Reshape back: (batch, slices, feature_dim)
        features = features.reshape(batch_size, num_slices, -1)
        
        # Compute attention weights for each slice
        attention_weights = self.slice_attention(features)  # (batch, slices, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Aggregate features using attention
        aggregated = (features * attention_weights).sum(dim=1)  # (batch, feature_dim)
        
        # Final classification
        output = self.classifier(aggregated)
        
        return output, attention_weights.squeeze(-1)