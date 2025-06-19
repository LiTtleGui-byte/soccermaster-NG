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

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

from models.deformable_detr.position_encoding import build_position_encoding
from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig
from models.modeling_timesformer_siglip import SiglipVisionModel as TimesformerSiglipVisionModel


class SiglipBackbone(nn.Module):
    def __init__(self, backbone_type: str, 
                 num_frames: int,
                 ckpt_path: str,
                 train_backbone: bool,
                 use_lora: bool):
        super().__init__()
        assert backbone_type in ['image', 'video']
        if backbone_type == 'image':
            self.model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        elif backbone_type == 'video':
            self.num_frames = num_frames
            self.model = TimesformerSiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        self.backbone_type = backbone_type
        
        if train_backbone:
            for name, param in self.model.named_parameters():
                # param.requires_grad = not name.startswith('head')
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            raise NotImplementedError("Siglip does not support LoRA.")
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None):
        if self.backbone_type == 'video':
            if temporal_attention_mask is None:
                if len(images.shape) == 4: # image input
                    temporal_attention_mask = torch.zeros(images.shape[0], self.num_frames, dtype=torch.bool, device=images.device)
                    temporal_attention_mask[:, 0] = 1.0
            
            outputs = self.model(images, temporal_attention_mask=temporal_attention_mask, output_hidden_states=True)
        else:
            outputs = self.model(images, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state # [N, L, D] or [N, T, L, D]
        pooled_output = outputs.pooler_output # [N, D] or [N, T, D]
        output = {'global_features': pooled_output, 'local_features': last_hidden_state}
        return output


