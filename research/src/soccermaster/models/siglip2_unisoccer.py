# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List

from soccermaster.models.deformable_detr.position_encoding import build_position_encoding
from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig, SiglipTextModel, AutoTokenizer
from soccermaster.models.modeling_timesformer_siglip import SiglipVisionModel as TimesformerSiglipVisionModel
from timm.models.layers import DropPath
from einops import rearrange

class ResidualAttentionBlock(nn.Module):
    def __init__(self, res_idx, d_model, n_head, drop_path=0., attn_mask=None, dropout=0., attention_type='divided_space_time', model_name="google/siglip-base-patch16-224"):
        super().__init__()
        model = SiglipVisionModel.from_pretrained(model_name)
        vision_model = model.vision_model

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')

        # Temporal Attention Parameters
        if attention_type == 'divided_space_time':
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.temporal_fc = nn.Linear(d_model, d_model)
            self.register_parameter('temporal_alpha_attn', nn.Parameter(torch.tensor(0.)))

        self.encoder = vision_model.encoder.layers[res_idx]
        self.attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x)[0]
    
    def temporal_attention(self, x):
        return self.temporal_attn(x, x, x)[0]

    def forward(self, x, B, T):
        # divided_space_time 

        ## Temporal 
        xt = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        res_temporal = self.drop_path(self.temporal_attention(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + self.temporal_alpha_attn.tanh() * res_temporal # 180 196 768

        ## Spatial
        xs = xt # always 180 196 768
        res_spatial = self.encoder(xs, self.attn_mask)[0]
        
        return res_spatial
    

class Timesformer(nn.Module):
    def __init__(self, width, layers, heads, model_name, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(res_idx=idx, d_model=width, n_head=heads, drop_path=dpr[idx], dropout=dropout, model_name=model_name))
        self.checkpoint_num = checkpoint_num
            
    def forward(self, x, B, T):
        for idx, blk in enumerate(self.resblocks):
            # if idx < self.checkpoint_num:
            #     x = checkpoint.checkpoint(blk, x)
            # else:
            x = blk(x, B, T)
        return x
    
class UniSoccerBackbone(nn.Module):
    def __init__(self, ckpt_path: str, num_frames: int, stage_1_backbone_dir: str, hidden_dim: int = 768):
        super().__init__()

        model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        siglip_vision_model = model.vision_model
        self.vision_model_embedding = siglip_vision_model.embeddings
        config = SiglipVisionConfig.from_pretrained(ckpt_path)
        self.timesformer = Timesformer(width=hidden_dim, layers=config.num_hidden_layers, heads=config.num_attention_heads, model_name=ckpt_path, drop_path=0., checkpoint_num=0, dropout=0.)
        self.post_norm = siglip_vision_model.post_layernorm
        self.head = siglip_vision_model.head
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        B, T, _, _, _ = images.shape
        images = rearrange(images, 'b t c h w -> (b t) c h w')
        x = self.vision_model_embedding(images)
        x = rearrange(x, '(b t) n m -> b n t m', b=B, t=T)
        x = x + self.temporal_embedding
        x = rearrange(x, 'b n t m -> (b t) n m')
        x = self.timesformer(x, B, T)
        x2 = self.post_norm(x) # [B*T, N, D]
        x2 = self.head(x2)
        x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
        x2 = rearrange(x2, '(b t) m -> b t m', b=B, t=T)
        return x, None, x2

class SiglipBackbone(nn.Module):
    def __init__(self, backbone_type: str, 
                 num_frames: int,
                 ckpt_path: str,
                 stage_1_ckpt_dir: str,
                 text_encoder_ckpt_path: str,
                 use_lora: bool,
                 use_temporal_gate: bool,
                 freeze_vision_encoder: bool = False,
                 freeze_text_encoder: bool = True,
                 hidden_dim: int = 768):
        super().__init__()
        assert backbone_type in ['image', 'video']
        if backbone_type == 'image':
            self.vision_model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        elif backbone_type == 'video':
            stage_1_backbone_dir = os.path.join(stage_1_ckpt_dir, 'backbone')
            if not os.path.exists(stage_1_backbone_dir):
                stage_1_backbone_dir = stage_1_ckpt_dir
            self.vision_model = UniSoccerBackbone(ckpt_path, num_frames, stage_1_backbone_dir, hidden_dim)
            
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
                
        self.text_model = TextEncoder(text_encoder_ckpt_path)
        
        if freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        else:
            for param in self.vision_model.parameters():
                param.requires_grad = True
        
        if freeze_text_encoder:
            for param in self.text_model.parameters():
                param.requires_grad = False
        else:
            for param in self.text_model.parameters():
                param.requires_grad = True
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        if self.backbone_type == 'video':
            last_hidden_state, hidden_states, pooled_output = self.vision_model(images, temporal_attention_mask, text)
        else:
            vision_outputs = self.vision_model(images, output_hidden_states=True)
            last_hidden_state = vision_outputs.last_hidden_state # [N, L, D] or [N, T, L, D]
            hidden_states = vision_outputs.hidden_states  # 修正属性名
            pooled_output = vision_outputs.pooler_output # [N, D] or [N, T, D]
        
        # For unisoccer (without part_temporal), early and late features are the same
        local_features_early = last_hidden_state
        local_features_late = last_hidden_state
        
        if text is not None:
            # 过滤出非None的text并记录其索引
            valid_texts = []
            valid_indices = []
            for i, t in enumerate(text):
                if t is not None:
                    valid_texts.append(t)
                    valid_indices.append(i)
            
            # 创建和原始batch_size匹配的tensor，None位置用零向量填充
            batch_size = len(text)
            text_dim = self.hidden_dim
            text_pooled_output = torch.zeros(batch_size, text_dim, device=images.device, dtype=images.dtype)
            
            # 如果valid_texts为空，则返回全0，不要返回None
            if valid_texts:
                # 对有效的text进行编码
                text_pooled_output_valid = self.text_model(valid_texts)[0] # only get the pooled output

                # 填充有效text的特征到对应位置
                for valid_idx, original_idx in enumerate(valid_indices):
                    text_pooled_output[original_idx] = text_pooled_output_valid[valid_idx]
        else:
            text_pooled_output = None
        
        output = {
            'global_features': pooled_output, 
            'local_features': last_hidden_state,  # For backward compatibility
            'local_features_early': local_features_early, 
            'local_features_late': local_features_late,
            'hidden_states': hidden_states, 
            'text_features': text_pooled_output
        }
        return output
    
class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        
        self.model_name = model_name
        self.model = SiglipTextModel.from_pretrained(model_name, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cpu", use_fast=False)

    def forward(self, text):
        # important: make sure to set padding="max_length" as that's how the model was trained
        if 'siglip2' in self.model_name:
            inputs = self.tokenizer(text=text, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        else:
            inputs = self.tokenizer(text=text, padding="max_length", return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return pooled_output, last_hidden_state