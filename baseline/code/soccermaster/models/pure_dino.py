# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DINOv2 Backbone modules.
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List

from transformers import AutoModel, AutoImageProcessor, SiglipTextModel, AutoTokenizer

class PureDinoBackbone(nn.Module):
    def __init__(self, backbone_type: str, 
                 num_frames: int,
                 ckpt_path: str,
                 stage_1_ckpt_dir: str,
                 text_encoder_ckpt_path: str,
                 use_lora: bool,
                 use_temporal_gate: bool,
                 freeze_vision_encoder: bool,
                 freeze_text_encoder: bool = True,
                 hidden_dim: int = 1024):
        super().__init__()
        assert backbone_type in ['image', 'video']
        
        # 加载 DINOv3 模型
        dino_repo_dir = './dinov3'
        model_size = None
        if 'dinov3_vitl16' in ckpt_path:
            model_size = 'dinov3_vitl16'
        elif 'dinov3_vitb16' in ckpt_path:
            model_size = 'dinov3_vitb16'
        else:
            raise ValueError(f"Unsupported DINOv3 model: {ckpt_path}")
        self.vision_model = torch.hub.load(dino_repo_dir, model_size, source='local', weights=ckpt_path)
        
        if backbone_type == 'video':
            self.num_frames = num_frames
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
        
        # 文本编码器（保持与SigLIP相同）
        # self.text_model = TextEncoder(text_encoder_ckpt_path)
        self.text_model = None
        
        # 冻结视觉编码器
        if freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            # if self.vision_projection is not None:
            #     for param in self.vision_projection.parameters():
            #         param.requires_grad = True  # 投影层仍然可训练
        else:
            for param in self.vision_model.parameters():
                param.requires_grad = True
        
        # 冻结文本编码器
        # if freeze_text_encoder:
        #     for param in self.text_model.parameters():
        #         param.requires_grad = False
        # else:
        #     for param in self.text_model.parameters():
        #         param.requires_grad = True
        
        if use_lora:
            raise NotImplementedError("DINOv2 does not support LoRA in this implementation.")
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        if self.backbone_type == 'video':
            B, T, _, _, _ = images.shape
            images = images.reshape(images.shape[0] * images.shape[1], *images.shape[2:])
        
        # DINOv3 前向传播 (is_training=True 返回字典格式)
        vision_outputs = self.vision_model(images, is_training=True)
        
        # 文本特征设为 None (DINOv3 不处理文本)
        text_pooled_output = None
        
        # 提取 CLS token 作为全局特征
        pooled_output = vision_outputs["x_norm_clstoken"]  # [N, D]
        
        # 提取 patch tokens 作为局部特征
        patch_tokens = vision_outputs["x_norm_patchtokens"]  # [N, L, D]
        
        # 获取中间层的 hidden states
        # n_layers = self.vision_model.n_blocks
        # intermediate_outputs = self.vision_model.get_intermediate_layers(
        #     images, 
        #     n=list(range(n_layers)),  # 获取所有层
        #     reshape=False,
        #     return_class_token=True,
        #     norm=True
        # )
        
        # intermediate_outputs 是 tuple of (patch_tokens, cls_token)
        # 我们需要重新组合成完整的 hidden states (包含 cls token)
        # hidden_states = []
        # for patch_tok, cls_tok in intermediate_outputs:
        #     # patch_tok: [N, L, D], cls_tok: [N, D]
        #     hidden_states.append(patch_tok)
        hidden_states = None
        
        # 视频模式：reshape 回原始维度
        if self.backbone_type == 'video':
            pooled_output = pooled_output.reshape(B, T, -1)
            patch_tokens = patch_tokens.reshape(B, T, *patch_tokens.shape[1:])
            # hidden_states = [hs.reshape(B, T, *hs.shape[1:]) for hs in hidden_states]
            hidden_states = None
        
        output = {
            'global_features': pooled_output, 
            'local_features': patch_tokens,  # 只使用 patch tokens
            'hidden_states': hidden_states,  # 所有层的 patch tokens
            'text_features': text_pooled_output
        }
        return output
