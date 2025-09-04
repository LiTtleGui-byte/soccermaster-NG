import os
import torch
from torch import nn

from models.siglip2 import SiglipBackbone
from models.siglip2_unisoccer import SiglipBackbone as UniSoccerSiglipBackbone
from models.deformable_detr.deformable_detr import build_deformable_detr_head
from models.lines_detection import build_lines_detection_head
from models.keypoints_detection import build_keypoints_detection_head
from models.soccernet_gsr_reid import build_soccer_net_gsr_reid_head
from models.video_caption import build_video_caption_head
from models.caption_classification import build_caption_classification_head
from models.caption_classification_align import build_caption_classification_head_align
from models.camera import build_camera_head
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.models.siglip.modeling_siglip import SiglipPreTrainedModel
from models.modeling_timesformer_siglip import SiglipVisionModel as TimesformerSiglipVisionModel
from safetensors import safe_open

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
        
        # 根据配置选择使用哪种SiglipBackbone
        siglip_backbone_type = config['SIGLIP_BACKBONE_TYPE'].lower()
        
        if siglip_backbone_type == 'standard':
            SiglipBackboneType = SiglipBackbone
        elif siglip_backbone_type == 'unisoccer':
            SiglipBackboneType = UniSoccerSiglipBackbone
        else:
            raise ValueError(f"Unsupported SIGLIP_BACKBONE_TYPE: {siglip_backbone_type}. Supported types: 'standard', 'unisoccer'")
        self.backbone = SiglipBackboneType(
            config['BACKBONE_TYPE'], 
            config['NUM_FRAMES'], 
            config['CKPT_PATH'], 
            config['STAGE_1_CKPT_DIR'], 
            config['TEXT_ENCODER_CKPT_PATH'], 
            config['TRAIN_BACKBONE'], 
            False, 
            config['BACKBONE_USE_TEMPORAL_GATE'], 
            config['FREEZE_TEXT_ENCODER'],
            config['BACKBONE_HIDDEN_DIM']
        )
        if logger is not None:
            logger.info(f"Using SiglipBackbone type: {siglip_backbone_type}")
        else:
            print(f"Using SiglipBackbone type: {siglip_backbone_type}")
        
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
            elif head == "CaptionClassificationAlign":
                self.multi_task_head[head] = build_caption_classification_head_align(config)
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
    
    def save_checkpoint(self, checkpoint_dir: str, logger=None):
        """
        Save model checkpoint including backbone, text encoder, and task heads
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            logger: Logger instance for logging messages
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 判断vision_model的类型来决定保存方式
        if isinstance(self.backbone.vision_model, SiglipPreTrainedModel):
            # 对于标准的SiglipVisionModel，使用save_pretrained方法
            backbone_dir = os.path.join(checkpoint_dir, 'backbone')
            self.backbone.vision_model.save_pretrained(backbone_dir)
            if logger is not None:
                logger.info(f"Saved SiglipPreTrainedModel backbone weights to: {backbone_dir}")
            else:
                print(f"Saved SiglipPreTrainedModel backbone weights to: {backbone_dir}")
        else:
            # 对于自定义的UniSoccerBackbone，使用torch.save保存state_dict
            backbone_path = os.path.join(checkpoint_dir, 'backbone.pt')
            torch.save(self.backbone.vision_model.state_dict(), backbone_path)
            if logger is not None:
                logger.info(f"Saved custom backbone weights to: {backbone_path}")
            else:
                print(f"Saved custom backbone weights to: {backbone_path}")
        
        # Save text encoder weights
        text_model_dir = os.path.join(checkpoint_dir, 'text_model')
        self.backbone.text_model.model.save_pretrained(text_model_dir)
        if logger is not None:
            logger.info(f"Saved text encoder weights to: {text_model_dir}")
        else:
            print(f"Saved text encoder weights to: {text_model_dir}")
        
        # Save task heads
        for head_name, head in self.multi_task_head.items():
            head_path = os.path.join(checkpoint_dir, f'{head_name}.pt')
            torch.save(head.state_dict(), head_path)
            if logger is not None:
                logger.info(f"Saved {head_name} head to: {head_path}")
            else:
                print(f"Saved {head_name} head to: {head_path}")
    
    def load_checkpoint(self, checkpoint_dir: str, logger=None, load_heads: bool = True):
        """
        Load model checkpoint including backbone, text encoder, and task heads
        
        Args:
            checkpoint_dir: Directory to load checkpoint from
            logger: Logger instance for logging messages
        """
        # # 判断vision_model的类型来决定加载方式
        # if isinstance(self.backbone.vision_model, SiglipPreTrainedModel):
        #     # 对于标准的SiglipVisionModel，从safetensors文件加载
        #     backbone_ckpt_path = os.path.join(checkpoint_dir, "backbone", "model.safetensors")
        #     if os.path.exists(backbone_ckpt_path):
        #         with safe_open(backbone_ckpt_path, framework="pt") as f:
        #             state_dict = {k: f.get_tensor(k) for k in f.keys()}
        #             self.backbone.vision_model.load_state_dict(state_dict, strict=False)
        #             if logger is not None:
        #                 logger.info(f"Loaded SiglipPreTrainedModel backbone weights from: {backbone_ckpt_path}")
        #             else:
        #                 print(f"Loaded SiglipPreTrainedModel backbone weights from: {backbone_ckpt_path}")
        #     else:
        #         if logger is not None:
        #             logger.warning(f"Warning: SiglipPreTrainedModel backbone checkpoint not found at {backbone_ckpt_path}")
        #         else:
        #             print(f"Warning: SiglipPreTrainedModel backbone checkpoint not found at {backbone_ckpt_path}")
        # else:
        #     # 对于自定义的UniSoccerBackbone，从.pt文件加载
        #     backbone_ckpt_path = os.path.join(checkpoint_dir, "backbone.pt")
        #     if os.path.exists(backbone_ckpt_path):
        #         backbone_state_dict = torch.load(backbone_ckpt_path, map_location="cpu")
        #         self.backbone.vision_model.load_state_dict(backbone_state_dict, strict=False)
        #         if logger is not None:
        #             logger.info(f"Loaded custom backbone weights from: {backbone_ckpt_path}")
        #         else:
        #             print(f"Loaded custom backbone weights from: {backbone_ckpt_path}")
        #     else:
        #         if logger is not None:
        #             logger.warning(f"Warning: custom backbone checkpoint not found at {backbone_ckpt_path}")
        #         else:
        #             print(f"Warning: custom backbone checkpoint not found at {backbone_ckpt_path}")
        backbone_ckpt_path_hf = os.path.join(checkpoint_dir, "backbone", "model.safetensors")
        backbone_ckpt_path_unisoccer = os.path.join(checkpoint_dir, "backbone.pt")
        if os.path.exists(backbone_ckpt_path_hf):
            with safe_open(backbone_ckpt_path_hf, framework="pt") as f:
                backbone_state_dict = {k: f.get_tensor(k) for k in f.keys()}
                self.backbone.vision_model.load_state_dict(backbone_state_dict, strict=False)
                if logger is not None:
                    logger.info(f"Loaded backbone weights from: {backbone_ckpt_path_hf}")
                else:
                    print(f"Loaded backbone weights from: {backbone_ckpt_path_hf}")
        elif os.path.exists(backbone_ckpt_path_unisoccer):
            backbone_state_dict = torch.load(backbone_ckpt_path_unisoccer, map_location="cpu")
            self.backbone.vision_model.load_state_dict(backbone_state_dict, strict=False)
            if logger is not None:
                logger.info(f"Loaded backbone weights from: {backbone_ckpt_path_unisoccer}")
            else:
                print(f"Loaded backbone weights from: {backbone_ckpt_path_unisoccer}")
        else:
            if logger is not None:
                logger.warning(f"Warning: backbone checkpoint not found at {backbone_ckpt_path_hf} or {backbone_ckpt_path_unisoccer}")
            else:
                print(f"Warning: backbone checkpoint not found at {backbone_ckpt_path_hf} or {backbone_ckpt_path_unisoccer}")

        
        # Load text encoder weights
        text_model_ckpt_path = os.path.join(checkpoint_dir, "text_model", "model.safetensors")
        if os.path.exists(text_model_ckpt_path):
            with safe_open(text_model_ckpt_path, framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
                self.backbone.text_model.model.load_state_dict(state_dict, strict=False)
                if logger is not None:
                    logger.info(f"Loaded text encoder weights from: {text_model_ckpt_path}")
                else:
                    print(f"Loaded text encoder weights from: {text_model_ckpt_path}")
        else:
            if logger is not None:
                logger.warning(f"Warning: text encoder checkpoint not found at {text_model_ckpt_path}")
            else:
                print(f"Warning: text encoder checkpoint not found at {text_model_ckpt_path}")
        
        # Load task heads
        if load_heads:
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
        else:
            if logger is not None:
                logger.info(f"Skipping loading task heads from: {checkpoint_dir}")
            else:
                print(f"Skipping loading task heads from: {checkpoint_dir}")