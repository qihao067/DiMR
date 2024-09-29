"""This file contains the model definition of DiMR.

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

from re import A
import torch
import torch.nn as nn
import math
import einops
import torch.utils.checkpoint
from functools import partial

import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .sigmoid.module import LayerNorm, RMSNorm, AdaRMSNorm, TDRMSNorm, QKNorm, TimeDependentParameter
from .common_layers import Linear, EvenDownInterpolate, ChannelFirst, ChannelLast, Embedding
from .axial_rope import AxialRoPE, make_axial_pos

def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)
    
class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim_in, dim_out, ratio = 2):
        super().__init__()
        self.ratio = ratio
        self.kernel = Linear(dim_in, dim_out * self.ratio * self.ratio)
    
    def forward(self, x):
        x = self.kernel(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H, W, self.ratio, self.ratio, C // self.ratio // self.ratio)
        x = x.transpose(2, 3)
        x = x.reshape(B, H * self.ratio, W * self.ratio, C // self.ratio // self.ratio)
        return x
    
class PositionEmbeddings(nn.Module):
    def __init__(self, max_height, max_width, dim):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.position_embeddings = Embedding(self.max_height * self.max_width, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        height_idxes = torch.arange(H, device = x.device)[:, None].repeat(1, W)
        width_idxes = torch.arange(W, device = x.device)[None, :].repeat(H, 1)
        idxes = height_idxes * self.max_width + width_idxes
        x = x + self.position_embeddings(idxes[None])
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.norm_type == 'LN':
            self.norm_type = 'LN'
            self.norm = LayerNorm(config.dim)
        elif config.norm_type == 'RMSN':
            self.norm_type = 'RMSN'
            self.norm = RMSNorm(config.dim)
        elif config.norm_type == 'TDRMSN':
            self.norm_type = 'TDRMSN'
            self.norm = TDRMSNorm(config.dim)
        elif config.norm_type == 'ADARMSN':
            self.norm_type = 'ADARMSN'
            self.norm = AdaRMSNorm(config.dim, config.dim)
        self.act = nn.GELU()
        self.w0 = Linear(config.dim, config.hidden_dim)
        self.w1 = Linear(config.dim, config.hidden_dim)
        self.w2 = Linear(config.hidden_dim, config.dim)

    def forward(self, x):
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            x = self.norm(x)
        elif self.norm_type == 'ADARMSN':
            condition = x[:,0]
            x = self.norm(x, condition)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.dim // config.num_attention_heads

        if hasattr(config, "condition_dim"):
            self.condition_key_value = Linear(config.condition_dim, 2 * config.dim, bias = False)

        if config.norm_type == 'LN':
            self.norm_type = 'LN'
            self.norm = LayerNorm(config.dim)
        elif config.norm_type == 'RMSN':
            self.norm_type = 'RMSN'
            self.norm = RMSNorm(config.dim)
        elif config.norm_type == 'TDRMSN':
            self.norm_type = 'TDRMSN'
            self.norm = TDRMSNorm(config.dim)
        elif config.norm_type == 'ADARMSN':
            self.norm_type = 'ADARMSN'
            self.norm = AdaRMSNorm(config.dim, config.dim)

        self.pe_type = config.pe_type
        if config.pe_type == 'Axial_RoPE':
            self.pos_emb = AxialRoPE(self.head_dim, self.num_heads)
            self.qk_norm = QKNorm(self.num_heads)

        self.query_key_value = Linear(config.dim, 3 * config.dim, bias = False)
        self.dense = Linear(config.dim, config.dim)

    def forward(self, x, condition_embeds, condition_masks, pos=None):
        B, N, C = x.shape
        
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            qkv = self.query_key_value(self.norm(x))
        elif self.norm_type == 'ADARMSN':
            condition = x[:,0]
            qkv = self.query_key_value(self.norm(x, condition))
        q, k, v = qkv.reshape(B, N, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).float().chunk(3, dim = 1)

        if self.pe_type == 'Axial_RoPE':
            q = self.pos_emb(self.qk_norm(q), pos)
            k = self.pos_emb(self.qk_norm(k), pos)

        if condition_embeds is not None:
            _, L, D = condition_embeds.shape
            kcvc = self.condition_key_value(condition_embeds)
            kc, vc = kcvc.reshape(B, L, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).float().chunk(2, dim = 1)
            k = torch.cat([k, kc], dim = 2)
            v = torch.cat([v, vc], dim = 2)
            mask = torch.cat([torch.ones(B, N, dtype = torch.bool, device = condition_masks.device), condition_masks], dim = -1)
            mask = mask[:, None, None, :]
        else:
            mask = None

        x = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        x = self.dense(x.permute(0, 2, 1, 3).reshape(B, N, C))

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = SelfAttention(config)
        self.block2 = MLPBlock(config)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, condition_embeds, condition_masks, pos):
        return torch.utils.checkpoint.checkpoint(self._forward, x, condition_embeds, condition_masks, pos)
        # return self._forward(x, condition_embeds, condition_masks, pos)
    
    def _forward(self, x, condition_embeds, condition_masks, pos):
        x = x + self.dropout(self.block1(x, condition_embeds, condition_masks, pos))
        x = x + self.dropout(self.block2(x))
        return x
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = nn.Sequential(
            ChannelFirst(), 
            nn.Conv2d(config.dim, config.dim, kernel_size = config.kernel_size, padding = config.kernel_size // 2, stride = 1, groups = config.dim), 
            ChannelLast()
        )
        self.block2 = MLPBlock(config)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, condition_embeds, condition_masks, pos):
        return torch.utils.checkpoint.checkpoint(self._forward, x, condition_embeds, condition_masks, pos)
        # return self._forward(x, condition_embeds, condition_masks, pos)
    
    def _forward(self, x, condition_embeds, condition_masks, pos):
        x = x + self.dropout(self.block1(x))
        x = x + self.dropout(self.block2(x))
        return x


class Stage(nn.Module):
    def __init__(self, channels, config, lowres_dim = None, lowres_height = None):
        super().__init__()
        if config.block_type == "TransformerBlock":
            self.encoder_cls = TransformerBlock
        elif config.block_type == "ConvNeXtBlock":
            self.encoder_cls = ConvNeXtBlock
        else:
            raise Exception()
        
        self.pe_type = config.pe_type
        
        if config.pe_type == 'Axial_RoPE' and config.block_type == 'TransformerBlock':
            self.input_layer = nn.Sequential(
                EvenDownInterpolate(config.image_input_ratio),
                nn.Conv2d(channels, config.dim, kernel_size = config.input_feature_ratio, stride = config.input_feature_ratio),
                ChannelLast()
            )
            self.pos_emb = AxialRoPE(config.dim, config.num_attention_heads)
        else:
            self.input_layer = nn.Sequential(
                EvenDownInterpolate(config.image_input_ratio),
                nn.Conv2d(channels, config.dim, kernel_size = config.input_feature_ratio, stride = config.input_feature_ratio),
                ChannelLast(),
                PositionEmbeddings(config.max_height, config.max_width, config.dim)
            )

        self.class_embeds = Embedding(config.num_classes, config.dim) if self.encoder_cls is TransformerBlock and hasattr(config, "num_classes") else None

        if lowres_dim is not None:
            ratio = config.max_height // lowres_height
            self.upsample = nn.Sequential(
                LayerNorm(lowres_dim),
                PixelShuffleUpsample(lowres_dim, config.dim, ratio = ratio),
                LayerNorm(config.dim),
            )

        self.blocks = nn.ModuleList([self.encoder_cls(config) for _ in range(config.num_blocks // 2 * 2 + 1)])
        self.skip_denses = nn.ModuleList([Linear(config.dim * 2, config.dim) for _ in range(config.num_blocks // 2)])

        self.output_layer = nn.Sequential(
            LayerNorm(config.dim),
            ChannelFirst(),
            nn.Conv2d(config.dim, channels, kernel_size = config.final_kernel_size, padding = config.final_kernel_size // 2),
        )

    def forward(self, images, lowres_skips = None, condition_classes = None, condition_embeds = None, condition_masks = None):
        if self.pe_type == 'Axial_RoPE' and self.encoder_cls is TransformerBlock:
            x = self.input_layer(images)
            _, H, W, _ = x.shape
            pos = make_axial_pos(H, W)
        else:
            x = self.input_layer(images)
            pos = None

        if lowres_skips is not None:
            x = x + self.upsample(lowres_skips)

        if self.encoder_cls is TransformerBlock:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)

        if self.class_embeds is not None:
            cls_token = self.class_embeds(condition_classes)[:, None, :]
            x = torch.cat([cls_token, x], dim = 1)

        external_skips = [x]

        num_blocks = len(self.blocks)
        in_blocks = self.blocks[:(num_blocks // 2)]
        mid_block = self.blocks[(num_blocks // 2)]
        out_blocks = self.blocks[(num_blocks // 2 + 1):]

        skips = []
        for block in in_blocks:
            x = block(x, condition_embeds, condition_masks, pos=pos)
            external_skips.append(x)
            skips.append(x)
        
        x = mid_block(x, condition_embeds, condition_masks, pos=pos)
        external_skips.append(x)

        for dense, block in check_zip(self.skip_denses, out_blocks):
            x = dense(torch.cat([x, skips.pop()], dim = -1))
            x = block(x, condition_embeds, condition_masks, pos=pos)
            external_skips.append(x)

        if self.class_embeds is not None:
            x = x[:, 1:, :]
            external_skips = [skip[:, 1:, :] for skip in external_skips]

        if self.encoder_cls is TransformerBlock:
            x = x.reshape(B, H, W, C)
            external_skips = [skip.reshape(B, H, W, C) for skip in external_skips]

        output = self.output_layer(x)
        return output, external_skips
    

class MRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.block_grad_to_lowres = config.block_grad_to_lowres

        for stage_config in config.stage_configs:
            if hasattr(config, "condition_dim"):
                stage_config.condition_dim = config.condition_dim
            if hasattr(config, "num_classes"):
                stage_config.num_classes = config.num_classes
            if hasattr(config, "pe_type"):
                stage_config.pe_type = config.pe_type
            else:
                stage_config.pe_type = 'APE'
            if hasattr(config, "norm_type"):
                stage_config.norm_type = config.norm_type
            else:
                stage_config.norm_type = 'LN'

        lowres_dims = [None] + [stage_config.dim * (stage_config.num_blocks // 2 * 2 + 2) for stage_config in config.stage_configs[:-1]]
        lowres_heights = [None] + [stage_config.max_height for stage_config in config.stage_configs[:-1]]
        self.stages = nn.ModuleList([
            Stage(self.channels, stage_config, lowres_dim = lowres_dim, lowres_height=lowres_height) 
            for stage_config, lowres_dim, lowres_height in check_zip(config.stage_configs, lowres_dims, lowres_heights)]
        )

    def _forward(self, images, log_snr, condition_classes = None, condition_text_embeds = None, condition_text_masks = None, condition_drop_prob = None):
        TimeDependentParameter.seed_time(self, log_snr)

        if condition_text_embeds is not None:
            condition_embeds = self.text_conditioning(condition_text_embeds)
            condition_masks = condition_text_masks
        else:
            condition_embeds = None
            condition_masks = None

        outputs = []
        lowres_skips = None
        for stage in self.stages:
            output, lowres_skips = stage(images, lowres_skips = lowres_skips, condition_classes = condition_classes, condition_embeds = condition_embeds, condition_masks = condition_masks)
            outputs.append(output)
            lowres_skips = torch.cat(lowres_skips, dim = -1)
            if self.block_grad_to_lowres:
                lowres_skips = lowres_skips.detach()

        return outputs

    def forward(self, x, t = None, log_snr = None, y = None):
        return self._forward(images = x, log_snr = log_snr, condition_classes = y)
        