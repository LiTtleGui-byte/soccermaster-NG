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
from typing import Optional, List

from models.deformable_detr.position_encoding import build_position_encoding
from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig, SiglipTextModel, AutoTokenizer
from models.modeling_timesformer_siglip import SiglipVisionModel as TimesformerSiglipVisionModel


class SiglipBackbone(nn.Module):
    def __init__(self, backbone_type: str, 
                 num_frames: int,
                 ckpt_path: str,
                 text_encoder_ckpt_path: str,
                 train_backbone: bool,
                 use_lora: bool):
        super().__init__()
        assert backbone_type in ['image', 'video']
        if backbone_type == 'image':
            self.vision_model = SiglipVisionModel.from_pretrained(ckpt_path, device_map="cpu")
        elif backbone_type == 'video':
            self.num_frames = num_frames
            config = SiglipVisionConfig.from_pretrained(ckpt_path)
            config.num_frames = num_frames
            self.vision_model = TimesformerSiglipVisionModel.from_pretrained(ckpt_path, config=config, device_map="cpu")
        self.backbone_type = backbone_type
        
        if train_backbone:
            for name, param in self.vision_model.named_parameters():
                param.requires_grad = True
        else:
            for param in self.vision_model.parameters():
                param.requires_grad = False
                
        self.text_model = TextEncoder(text_encoder_ckpt_path)
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        if use_lora:
            raise NotImplementedError("Siglip does not support LoRA.")
        
    def forward(self, images: torch.Tensor, temporal_attention_mask: Optional[torch.Tensor] = None, text: Optional[List[str]] = None):
        if self.backbone_type == 'video':
            if temporal_attention_mask is None:
                if len(images.shape) == 4: # image input
                    # temporal_attention_mask = torch.zeros(images.shape[0], self.num_frames, dtype=torch.bool, device=images.device)
                    temporal_attention_mask = torch.zeros(1, self.num_frames, dtype=torch.bool, device=images.device) # 依靠广播机制
                    temporal_attention_mask[:, 0] = 1.0
            
            vision_outputs = self.vision_model(images, temporal_attention_mask=temporal_attention_mask, output_hidden_states=True)
        else:
            vision_outputs = self.vision_model(images, output_hidden_states=True)
        
        if text is not None:
            text_pooled_output = self.text_model(text)[0] # only get the pooled output
        else:
            text_pooled_output = None
        
        last_hidden_state = vision_outputs.last_hidden_state # [N, L, D] or [N, T, L, D]
        pooled_output = vision_outputs.pooler_output # [N, D] or [N, T, D]
        output = {'global_features': pooled_output, 'local_features': last_hidden_state, 'text_features': text_pooled_output}
        return output

class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        
        self.model = SiglipTextModel.from_pretrained(model_name, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cpu")

    def forward(self, text):
        # important: make sure to set padding="max_length" as that's how the model was trained
        inputs = self.tokenizer(text=text, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return pooled_output, last_hidden_state