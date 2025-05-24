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
import torchvision
from torch import nn
from typing import Dict, List
import math

from models.deformable_detr.position_encoding import build_position_encoding
from transformers import AutoProcessor, SiglipVisionModel


class SiglipBackbone(nn.Module):
    def __init__(self, name: str,
                 ckpt_path: str,
                 train_backbone: bool,
                 use_lora: bool):
        super().__init__()

        
        self.model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu").vision_model
        self.strides = [16]
        self.num_channels = [768]
        
        if train_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = not name.startswith('head')
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            raise NotImplementedError("Siglip does not support LoRA.")
        
    def forward(self, images: torch.Tensor):
        outputs = self.model(images, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state # [N, L, D]
        pooled_output = outputs.pooler_output # [N, D]
        output = {'global_features': pooled_output, 'local_features': last_hidden_state}
        return output

def build_backbone(args):
    # position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    # return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if 'siglip' in args.backbone:
        backbone = SiglipBackbone(args.backbone, args.ckpt_path, train_backbone, False)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    # model = Joiner(backbone, position_embedding)
    return backbone
