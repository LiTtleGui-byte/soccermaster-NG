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
from models.lines_detection import build_lines_detection_head
from models.keypoints_detection import build_keypoints_detection_head
from models.soccernet_gsr_reid import build_soccer_net_gsr_reid_head
from models.video_caption import build_video_caption_head
from models.camera import build_camera_head

# def build_backbone(config: dict):
#     # position_embedding = build_position_encoding(args)
#     train_backbone = config['TRAIN_BACKBONE']
#     # return_interm_layers = args.masks or (args.num_feature_levels > 1)
#     if 'siglip' in config['BACKBONE']:
#         backbone = SiglipBackbone(config['BACKBONE'], config['NUM_FRAMES'], config['CKPT_PATH'], train_backbone, False)
#     else:
#         raise ValueError(f"Unsupported backbone: {config['BACKBONE']}")
#     return backbone

class MultiTaskingSigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.backbone = SiglipBackbone(config['BACKBONE_TYPE'], config['NUM_FRAMES'], config['CKPT_PATH'], config['TEXT_ENCODER_CKPT_PATH'], config['TRAIN_BACKBONE'], False)
        
        # multi-task heads
        self.multi_task_head = nn.ModuleDict()
        # tasks = config["TASKS"]
        self.datasets_to_heads = config["DATASETS_TO_HEADS"]
        all_heads = []
        for dataset, heads in self.datasets_to_heads.items():
            all_heads.extend(heads)
        all_heads = list(set(all_heads))
        all_heads.sort()
        for head in all_heads:
            if head == "SoccerNetGSR_Detection":
                self.multi_task_head[head] = build_deformable_detr_head(config)
            elif head == "LinesDetection":
                self.multi_task_head[head] = build_lines_detection_head(config)
            elif head == "KeypointsDetection":
                self.multi_task_head[head] = build_keypoints_detection_head(config)
            elif head == "SoccerNetGSR_ReID":
                self.multi_task_head[head] = build_soccer_net_gsr_reid_head(config)
            elif head == "VideoCaption":
                self.multi_task_head[head] = build_video_caption_head(config)
            elif head == "CameraRegression":
                self.multi_task_head[head] = build_camera_head(config)
            else:
                raise ValueError(f"Head {head} is not supported.")

    def forward(self, images, dataset_name, metas, text=None):
        backbone_outputs = self.backbone(images, text=text)
        
        outputs = {}
        for head in self.datasets_to_heads[dataset_name]:
            outputs[head] = self.multi_task_head[head](backbone_outputs, metas)
        
        return outputs