import os
import torch
from torch import nn

from models.siglip2 import SiglipBackbone
from models.deformable_detr.deformable_detr import build_deformable_detr_head
from models.lines_detection import build_lines_detection_head
from models.keypoints_detection import build_keypoints_detection_head
from models.soccernet_gsr_reid import build_soccer_net_gsr_reid_head
from models.video_caption import build_video_caption_head
from models.caption_classification import build_caption_classification_head
from models.camera import build_camera_head
from transformers import SiglipVisionConfig, SiglipVisionModel

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
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        
        self.backbone = SiglipBackbone(config['BACKBONE_TYPE'], config['NUM_FRAMES'], config['CKPT_PATH'], config['STAGE_1_CKPT_DIR'], config['TEXT_ENCODER_CKPT_PATH'], config['TRAIN_BACKBONE'], False)
        
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
            elif head == "CaptionClassification":
                self.multi_task_head[head] = build_caption_classification_head(config)
            elif head == "CameraRegression":
                self.multi_task_head[head] = build_camera_head(config)
            else:
                raise ValueError(f"Head {head} is not supported.")
            
        # if config['BACKBONE_TYPE'] == 'video':
        #     stage_1_ckpt_dir = config['STAGE_1_CKPT_DIR']
        #     for head in all_heads:
        #         head_path = os.path.join(stage_1_ckpt_dir, f'{head}.pt')
        #         if os.path.exists(head_path):
        #             logger.info(f"Loading {head} head from: {head_path}")
        #             head_state_dict = torch.load(head_path, map_location='cpu')
        #             self.multi_task_head[head].load_state_dict(head_state_dict)
        #         else:
        #             logger.warning(f"Warning: {head} head checkpoint not found at {head_path}")

    def forward(self, images, dataset_name, metas=None, text=None):
        backbone_outputs = self.backbone(images, text=text)
        
        outputs = {}
        for head in self.datasets_to_heads[dataset_name]:
            outputs[head] = self.multi_task_head[head](backbone_outputs, metas)
        
        return outputs
    
    def load_checkpoint(self, checkpoint_dir: str, logger=None):
        backbone_ckpt_path = os.path.join(checkpoint_dir, "backbone")
        siglip_vision_config = SiglipVisionConfig.from_pretrained(backbone_ckpt_path)
        siglip_vision_config.num_frames = self.config['NUM_FRAMES']
        del self.backbone.vision_model
        self.backbone.vision_model = SiglipVisionModel.from_pretrained(backbone_ckpt_path, config=siglip_vision_config, device_map="cpu")
        for head in self.multi_task_head:
            head_ckpt_path = os.path.join(checkpoint_dir, f"{head}.pt")
            if os.path.exists(head_ckpt_path):
                if logger is not None:
                    logger.info(f"Loading {head} head from: {head_ckpt_path}")
                else:
                    print(f"Loading {head} head from: {head_ckpt_path}")
                head_state_dict = torch.load(head_ckpt_path, map_location="cpu")
                self.multi_task_head[head].load_state_dict(head_state_dict)
            else:
                if logger is not None:
                    logger.warning(f"Warning: {head} head checkpoint not found at {head_ckpt_path}")
                else:
                    print(f"Warning: {head} head checkpoint not found at {head_ckpt_path}")