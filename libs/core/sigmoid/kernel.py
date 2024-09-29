"""This file contains the class definition of Time-dependent Parameters.

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
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from torch.utils.checkpoint import checkpoint
from torch.autograd import Function
from functools import partial
import warnings


def exported_tdp(param0, param1, weight, bias, times, custom = True):
    original_shape = param0.shape
    param0 = param0.reshape(-1)
    param1 = param1.reshape(-1)
    weight = weight.reshape(-1)
    bias = bias.reshape(-1)
    if custom and param0.shape[0] % 2 == 0:
        result = TDP.apply(param0, param1, weight, bias, times)
    else:
        warnings.warn(f'Using slower tdp_torch implementation for a tensor with shape {param0.shape}')
        result = tdp_torch(param0, param1, weight, bias, times)
    result = result.reshape(*([times.shape[0]] + [d for d in original_shape]))
    return result

class TDP(Function):
    @staticmethod
    def forward(ctx, param0, param1, weight, bias, times):
        assert param0.shape[0] % 2 == 0
        param0 = param0.contiguous()
        param1 = param1.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        times = times.contiguous()
        assert param0.shape[0] == param1.shape[0] and param0.shape[0] == weight.shape[0] and param0.shape[0] == bias.shape[0]
        assert param0.dim() == 1 and param1.dim() == 1 and weight.dim() == 1 and bias.dim() == 1 and times.dim() == 1
        ctx.save_for_backward(param0, param1, weight, bias, times)
        return tdp_cuda(param0, param1, weight, bias, times)

    @staticmethod
    def backward(ctx, g_result):
        g_result = g_result.contiguous()
        param0, param1, weight, bias, times = ctx.saved_tensors
        g_param0, g_param1, g_weight, g_bias = backward_tdp_cuda(param0, param1, weight, bias, times, g_result)
        return g_param0, g_param1, g_weight, g_bias, None

def backward_tdp_torch(param0, param1, weight, bias, times, g_result):
    param0 = param0[None]
    param1 = param1[None]
    weight = weight[None]
    bias = bias[None]
    
    a = times[:, None] * weight + bias
    s = torch.sigmoid(a)
    g_param0 = (s * g_result).sum(0)
    g_param1 = ((1 - s) * g_result).sum(0)
    g_s = (param0 - param1) * g_result
    g_a = g_s * s * (1 - s)
    g_weight = (g_a * times[:, None]).sum(0)
    g_bias = g_a.sum(0)
    
    return g_param0, g_param1, g_weight, g_bias

def backward_tdp_cuda(param0, param1, weight, bias, times, g_result):
    g_param0 = torch.empty_like(param0)
    g_param1 = torch.empty_like(param0)
    g_weight = torch.empty_like(param0)
    g_bias = torch.empty_like(param0)
    if param0.dtype == torch.half:
        tdp.backward_tdp_fp16(param0, param1, weight, bias, times, g_result, g_param0, g_param1, g_weight, g_bias)
    elif param0.dtype == torch.float:
        tdp.backward_tdp_fp32(param0, param1, weight, bias, times, g_result, g_param0, g_param1, g_weight, g_bias)
    else:
        raise NotImplementedError
    return g_param0, g_param1, g_weight, g_bias

def tdp_torch(param0, param1, weight, bias, times):
    a = torch.addcmul(bias[None], times[:, None], weight[None])
    s = torch.sigmoid(a)
    result = torch.addcmul(param1[None], s, param0[None] - param1[None])
    return result

def tdp_cuda(param0, param1, weight, bias, times):
    result = torch.empty(times.shape[0], param0.shape[0], dtype = param0.dtype, device = param0.device)
    if param0.dtype == torch.half:
        tdp.tdp_fp16(param0, param1, weight, bias, times, result)
    elif param0.dtype == torch.float:
        tdp.tdp_fp32(param0, param1, weight, bias, times, result)
    else:
        raise NotImplementedError
    return result

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def tdp_cuda_unit_test():
    print("***** tdp_cuda_unit_test *****")
    
    batch_size = random.randrange(1, 128)
    num_params = random.randrange(1, 1000000) * 2
    print("batch_size", batch_size, "num_params", num_params)
    
    param0 = torch.randn(num_params).cuda()
    param1 = torch.randn(num_params).cuda()
    weight = torch.randn(num_params).cuda()
    bias = torch.randn(num_params).cuda()
    times = torch.rand(batch_size).cuda()
    
    ref = tdp_torch(param0, param1, weight, bias, times)
    
    out = tdp_cuda(param0, param1, weight, bias, times)
    print(corrcoef(ref, out), (ref - out).abs().max())
    
    out = tdp_cuda(param0.half(), param1.half(), weight.half(), bias.half(), times.half()).float()
    print(corrcoef(ref, out), (ref - out).abs().max())
    
def backward_tdp_cuda_unit_test():
    print("***** backward_tdp_cuda_unit_test *****")
    
    batch_size = random.randrange(1, 128)
    num_params = random.randrange(1, 100000) * 2
    print("batch_size", batch_size, "num_params", num_params)
    
    param0 = torch.randn(num_params).cuda()
    param1 = torch.randn(num_params).cuda()
    weight = torch.randn(num_params).cuda()
    bias = torch.randn(num_params).cuda()
    times = torch.rand(batch_size).cuda()
    g_result = torch.randn(batch_size, num_params).cuda()
    
    refs = backward_tdp_torch(param0, param1, weight, bias, times, g_result)
    
    outs = backward_tdp_cuda(param0, param1, weight, bias, times, g_result)
    for r, o in zip(refs, outs):
        print(corrcoef(r, o), (r - o).abs().max())
    
    outs = backward_tdp_cuda(param0.half(), param1.half(), weight.half(), bias.half(), times.half(), g_result.half())
    for r, o in zip(refs, outs):
        print(corrcoef(r, o), (r - o).abs().max())
    
def autograd_unit_test():
    print("***** autograd_unit_test *****")
    batch_size = random.randrange(1, 128)
    num_params = random.randrange(1, 100000) * 2
    print("batch_size", batch_size, "num_params", num_params)
    
    def get_outputs(fn):
        torch.manual_seed(1)
        param0 = torch.randn(num_params, requires_grad = True).cuda()
        param1 = torch.randn(num_params, requires_grad = True).cuda()
        weight = torch.randn(num_params, requires_grad = True).cuda()
        bias = torch.randn(num_params, requires_grad = True).cuda()
        times = torch.rand(batch_size).cuda()
        
        out = fn(param0, param1, weight, bias, times)
        loss = ((out - 1.5) ** 2).mean()
        
        param0.retain_grad()
        param1.retain_grad()
        weight.retain_grad()
        bias.retain_grad()
        
        loss.backward()
        g_param0 = param0.grad
        g_param1 = param1.grad
        g_weight = weight.grad
        g_bias = bias.grad
        
        return out, g_param0, g_param1, g_weight, g_bias
    
    refs = get_outputs(tdp_torch)
    outs = get_outputs(TDP.apply)
    for r, o in zip(refs, outs):
        print(corrcoef(r, o), (r - o).abs().max())
    
def exported_tdp_unit_test():
    print("***** exported_tdp_unit_test *****")
    batch_size = random.randrange(1, 128)
    num_params = random.randrange(1, 100000) * 2
    print("batch_size", batch_size, "num_params", num_params)
    
    def get_outputs(fn):
        torch.manual_seed(1)
        param0 = torch.randn(num_params, requires_grad = True).cuda()
        param1 = torch.randn(num_params, requires_grad = True).cuda()
        weight = torch.randn(num_params, requires_grad = True).cuda()
        bias = torch.randn(num_params, requires_grad = True).cuda()
        times = torch.rand(batch_size).cuda()
        
        out = fn(param0, param1, weight, bias, times)
        loss = ((out - 1.5) ** 2).mean()
        
        param0.retain_grad()
        param1.retain_grad()
        weight.retain_grad()
        bias.retain_grad()
        
        loss.backward()
        g_param0 = param0.grad
        g_param1 = param1.grad
        g_weight = weight.grad
        g_bias = bias.grad
        
        return out, g_param0, g_param1, g_weight, g_bias
    
    refs = get_outputs(partial(exported_tdp, custom = False))
    outs = get_outputs(partial(exported_tdp, custom = True))
    for r, o in zip(refs, outs):
        print(corrcoef(r, o), (r - o).abs().max())
    
def tdp_cuda_profile():
    print("***** tdp_cuda_profile *****")
    def profiler(fn, args):
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            fn(*args)
        torch.cuda.synchronize()
        t1 = time.time()
        return t1 - t0
    
    batch_size = 16
    num_params = 1024 * 1024
    print("batch_size", batch_size, "num_params", num_params)
    
    param0 = torch.randn(num_params).cuda()
    param1 = torch.randn(num_params).cuda()
    weight = torch.randn(num_params).cuda()
    bias = torch.randn(num_params).cuda()
    times = torch.rand(batch_size).cuda()
    
    print("ref", profiler(tdp_torch, (param0, param1, weight, bias, times)))
    print("cuda", profiler(tdp_cuda, (param0, param1, weight, bias, times)))
    
    print("ref", profiler(tdp_torch, (param0.half(), param1.half(), weight.half(), bias.half(), times.half())))
    print("cuda", profiler(tdp_cuda, (param0.half(), param1.half(), weight.half(), bias.half(), times.half())))
    
def backward_tdp_cuda_profile():
    print("***** backward_tdp_cuda_profile *****")
    def profiler(fn, args):
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            fn(*args)
        torch.cuda.synchronize()
        t1 = time.time()
        return t1 - t0
    
    batch_size = 16
    num_params = 1024 * 1024
    print("batch_size", batch_size, "num_params", num_params)
    
    param0 = torch.randn(num_params).cuda()
    param1 = torch.randn(num_params).cuda()
    weight = torch.randn(num_params).cuda()
    bias = torch.randn(num_params).cuda()
    times = torch.rand(batch_size).cuda()
    g_result = torch.randn(batch_size, num_params).cuda()
    
    
    print("ref", profiler(backward_tdp_torch, (param0, param1, weight, bias, times, g_result)))
    print("cuda", profiler(backward_tdp_cuda, (param0, param1, weight, bias, times, g_result)))
    
    print("ref", profiler(backward_tdp_torch, (param0.half(), param1.half(), weight.half(), bias.half(), times.half(), g_result.half())))
    print("cuda", profiler(backward_tdp_cuda, (param0.half(), param1.half(), weight.half(), bias.half(), times.half(), g_result.half())))
    
def autogad_profile():
    print("***** autogad_profile *****")
    def profiler(fn, args):
        for _ in range(10):
            fn(*args).mean().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            fn(*args).mean().backward()
        torch.cuda.synchronize()
        t1 = time.time()
        return t1 - t0
    
    batch_size = 16
    num_params = 1024 * 1024
    print("batch_size", batch_size, "num_params", num_params)
    
    param0 = nn.Parameter(torch.randn(num_params)).cuda()
    param1 = nn.Parameter(torch.randn(num_params)).cuda()
    weight = nn.Parameter(torch.randn(num_params)).cuda()
    bias = nn.Parameter(torch.randn(num_params)).cuda()
    times = torch.rand(batch_size).cuda()
    
    print("ref", profiler(tdp_torch, (param0, param1, weight, bias, times)))
    print("cuda", profiler(TDP.apply, (param0, param1, weight, bias, times)))
    
    print("ref", profiler(tdp_torch, (param0.half(), param1.half(), weight.half(), bias.half(), times.half())))
    print("cuda", profiler(TDP.apply, (param0.half(), param1.half(), weight.half(), bias.half(), times.half())))
    
if __name__ == "__main__":
    tdp_cuda_unit_test()
    backward_tdp_cuda_unit_test()
    autograd_unit_test()
    exported_tdp_unit_test()
    tdp_cuda_profile()
    backward_tdp_cuda_profile()
    autogad_profile()