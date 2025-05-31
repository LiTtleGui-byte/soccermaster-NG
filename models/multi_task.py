import collections
import collections.abc
from dataclasses import dataclass
import types
from typing import Dict, List, Optional, Tuple, Union
import random
import math
import copy
import torch
import torch.nn.functional
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from models.siglip2 import SiglipBackbone
from models.deformable_detr.deformable_detr import build_deformable_detr_head
from models.SoccerNetGSR_ReID import build_soccer_net_gsr_reid_head

def build_backbone(config: dict):
    # position_embedding = build_position_encoding(args)
    train_backbone = config['TRAIN_BACKBONE']
    # return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if 'siglip' in config['BACKBONE']:
        backbone = SiglipBackbone(config['BACKBONE'], config['CKPT_PATH'], train_backbone, False)
    else:
        raise ValueError(f"Unsupported backbone: {config['BACKBONE']}")
    return backbone

class MultiTaskingSigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.backbone = SiglipBackbone(config['CKPT_PATH'], config['TRAIN_BACKBONE'], False)
        
        # multi-task heads
        self.multi_task_head = nn.ModuleDict()
        tasks = config["TASKS"]
        for task in tasks:
            if task == "SoccerNetGSR_Detection":
                self.multi_task_head[task] = build_deformable_detr_head(config)
            elif task == "SoccerNetGSR_ReID":
                self.multi_task_head[task] = build_soccer_net_gsr_reid_head(config)
            else:
                raise ValueError(f"Task {task} is not supported.")

    def forward(self, images, task):
        backbone_outputs = self.backbone(images)
        
        # outputs = {task: self.multi_task_head[task](backbone_outputs) for task in self.multi_task_head.keys()}
        # 不同数据对应了不同的head，只给对应的那个
        outputs = {task: self.multi_task_head[task](backbone_outputs)}
        
        return outputs