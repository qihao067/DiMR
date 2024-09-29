"""This file contains the code for certain common layers.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trunc_normal_(self.weight, mean = 0, std = 0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trunc_normal_(self.weight, mean = 0, std = 0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trunc_normal_(self.weight, mean = 0, std = 0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trunc_normal_(self.weight, mean = 0, std = 0.02)

class ImageNorm(nn.Module):
    def forward(self, x):
        assert x.dim() == 4
        eps = 1e-05
        x = x / (x.var(dim = (1, 2, 3), keepdim = True) + eps).sqrt()
        return x

class Flatten(nn.Module):
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x
    
class ChannelLast(nn.Module):
    def forward(self, x):
        assert x.dim() == 4
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        return x
    
class ChannelFirst(nn.Module):
    def forward(self, x):
        assert x.dim() == 4
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        return x
    
class OddUpInterpolate(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.ratio == 1:
            return x
        assert x.dim() == 4
        B, C, H, W = x.shape
        x = F.interpolate(x, size = ((H - 1) * self.ratio + 1, (W - 1) * self.ratio + 1), mode = "bilinear", align_corners = True)
        return x
    
    def __repr__(self):
        return f"UpInterpolate(ratio={self.ratio})"
    
class OddDownInterpolate(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.ratio == 1:
            return x
        assert x.dim() == 4
        B, C, H, W = x.shape
        x = F.interpolate(x, size = ((H - 1) // self.ratio + 1, (W - 1) // self.ratio + 1), mode = "area")
        return x
    
    def __repr__(self):
        return f"DownInterpolate(ratio={self.ratio})"
    
class EvenDownInterpolate(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.ratio == 1:
            return x
        assert len(x.shape) == 4
        B, C, H, W = x.shape
        x = F.interpolate(x, size = (H // self.ratio, W // self.ratio), mode = "area")
        return x
    
    def __repr__(self):
        return f"DownInterpolate(ratio={self.ratio})"