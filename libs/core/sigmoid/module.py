"""This file contains the class definitions of common normalization modules.

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

from functools import reduce
import math
import torch
import torch.nn as nn
import numpy as np
from .kernel import exported_tdp
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_

class TimeDependentParameter(nn.Module):
    def __init__(self, shape, init_fn):
        super().__init__()
        self.shape = shape

        w = torch.empty(*shape)
        init_fn(w)

        self.param0 = nn.Parameter(w.clone().detach())
        self.param1 = nn.Parameter(w.clone().detach())

        self.nodecay_weight = nn.Parameter(torch.zeros(*shape))
        self.nodecay_bias = nn.Parameter(torch.zeros(*shape))
        self.curr_weight = None

    def forward(self):
        weight = self.curr_weight
        # self.curr_weight = None
        return weight
    
    def __repr__(self):
        return f"TimeDependentParameter(shape={self.shape})"
    
    @staticmethod
    def seed_time(model, log_snr):
        assert log_snr.dim() == 1
        if torch.all(log_snr == log_snr[0]):
            log_snr = log_snr[0][None]
        time_condition = log_snr / 4.0

        tdp_list = [module for module in model.modules() if isinstance(module, TimeDependentParameter)]
        for tdp in tdp_list:
            tdp.curr_weight = exported_tdp(tdp.param0, tdp.param1, tdp.nodecay_weight + 1, tdp.nodecay_bias, time_condition, custom = False)
    
class LayerNorm(nn.Module):
    def __init__(self, dim, num_groups = 1, eps = 1e-05):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.num_groups = num_groups
        self.weight = TimeDependentParameter((dim, ), nn.init.ones_)
        self.bias = TimeDependentParameter((dim, ), nn.init.zeros_)

    def _forward(self, x):
        weight, bias = self.weight(), self.bias()
        assert weight.shape[0] == bias.shape[0]

        assert x.shape[-1] == self.dim

        if weight.shape[0] == 1:
            x = F.layer_norm(x, (self.dim, ), weight = weight[0], bias = bias[0], eps = self.eps)
        else:
            assert x.shape[0] == weight.shape[0]
            x = F.layer_norm(x, (self.dim, ), eps = self.eps)
            x = torch.addcmul(bias[:, None, :], weight[:, None, :], x)

        return x
    
    def forward(self, x):
        original_shape = x.shape
        batch_size = x.shape[0]
        assert self.dim == x.shape[-1]
        
        x = x.reshape(batch_size, -1, self.dim)
        x = self._forward(x)
        x = x.reshape(*original_shape)

        return x
    
class Linear(nn.Module):
    def __init__(self, din, dout, bias = True, weight_init_fn = partial(trunc_normal_, std = 0.02)):
        super().__init__()
        self.din = din
        self.dout = dout
        self.weight = TimeDependentParameter((din, dout), weight_init_fn)
        if bias:
            self.bias = TimeDependentParameter((dout, ), nn.init.zeros_)
        else:
            self.bias = None

    def _forward(self, x):
        weight = self.weight()
        bias = self.bias() if self.bias is not None else None
        
        if bias is not None:
            x = torch.baddbmm(bias[:, None, :], x, weight)
        else:
            x = torch.bmm(x, weight)

        return x

    def forward(self, x):
        original_shape = x.shape
        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, -1, self.din)
        x = self._forward(x)
        x = x.reshape(*(list(original_shape[:-1]) + [self.dout]))

        return x

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class TDRMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(TDRMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        # self.scale = nn.Parameter(torch.ones(d))
        self.scale = TimeDependentParameter((d, ), nn.init.ones_)
        # self.register_parameter("scale", self.scale)

        if self.bias:
            # self.offset = nn.Parameter(torch.zeros(d))
            self.offset = TimeDependentParameter((d, ), nn.init.zeros_)
            # self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        _scale = self.scale()

        if self.bias:
            # return self.scale * x_normed + self.offset
            _offset = self.offset()
            if _scale.shape[0] == 1:
                return _scale[0] * x_normed + _offset[0]
            elif x_normed.dim() == 3:
                return torch.addcmul(_offset[:, None, :], _scale[:, None, :], x_normed)
            elif x_normed.dim() == 4:
                return torch.addcmul(_offset[:, None, None, :], _scale[:, None, None, :], x_normed)
            else:
                raise NotImplementedError

        # return self.scale * x_normed
        if _scale.shape[0] == 1:
            return _scale[0] * x_normed
        elif x_normed.dim() == 3:
            return _scale[:, None, :] * x_normed
        elif x_normed.dim() == 4:
            return _scale[:, None, None, :] * x_normed
        else:
            raise NotImplementedError


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)

class QKNorm(nn.Module):
    def __init__(self, n_heads, eps=1e-6, max_scale=100.0):
        super().__init__()
        self.eps = eps
        self.max_scale = math.log(max_scale)
        self.scale = nn.Parameter(torch.full((n_heads,), math.log(10.0)))
        self.proj_()

    def extra_repr(self):
        return f"n_heads={self.scale.shape[0]}, eps={self.eps}"

    @torch.no_grad()
    def proj_(self):
        """Modify the scale in-place so it doesn't get "stuck" with zero gradient if it's clamped
        to the max value."""
        self.scale.clamp_(max=self.max_scale)

    def forward(self, x):
        self.proj_()
        scale = torch.exp(0.5 * self.scale - 0.25 * math.log(x.shape[-1]))
        return rms_norm(x, scale[:, None, None], self.eps)