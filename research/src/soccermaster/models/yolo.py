import os
import torch
from torch import nn

from soccermaster.models.siglip2 import SiglipBackbone
from soccermaster.models.siglip2_unisoccer import SiglipBackbone as UniSoccerSiglipBackbone
from soccermaster.models.deformable_detr.deformable_detr import build_deformable_detr_head
from soccermaster.models.lines_detection import build_lines_detection_head
from soccermaster.models.keypoints_detection import build_keypoints_detection_head
from soccermaster.models.soccernet_gsr_reid import build_soccer_net_gsr_reid_head
from soccermaster.models.video_caption import build_video_caption_head
from soccermaster.models.caption_classification import build_caption_classification_head
from soccermaster.models.caption_classification_align import build_caption_classification_head_align
from soccermaster.models.camera import build_camera_head
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.models.siglip.modeling_siglip import SiglipPreTrainedModel
from soccermaster.models.modeling_timesformer_siglip import SiglipVisionModel as TimesformerSiglipVisionModel
from safetensors import safe_open

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

class YOLOModel(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.model = YOLO(config['YOLO_MODEL_PATH'])
        self.yolo_imgsz = config['YOLO_IMGSZ']
        self.yolo_use_rect = config['YOLO_USE_RECT']
        
        all_heads = []
        for dataset, heads in config["DATASETS_TO_HEADS"].items():
            all_heads.extend(heads)
        all_heads = list(set(all_heads))
        all_heads.sort()
        assert 'SoccerNetGSR_Detection' in all_heads, "SoccerNetGSR_Detection is not in all_heads"
        assert len(all_heads) == 1, "Only SoccerNetGSR_Detection is supported"

        self.backbone_type = config['BACKBONE_TYPE']
        assert self.backbone_type == 'image', "Only image backbone is supported"
            
    def forward(self, images, dataset_name, metas=None, text=None, accelerator=None):
        # if self.backbone_type == 'video':
        #     bs, num_frames, _, _, _ = images.shape
        #     images = images.reshape(bs * num_frames, *images.shape[2:])
        # else:
        #     bs, _, _, _ = images.shape
        results_by_image = self.model(images, imgsz=self.yolo_imgsz, rect=self.yolo_use_rect)
        device = accelerator.device
        num_queries = 300
        pred_logits_batch = []
        pred_boxes_batch = []
        for results in results_by_image:
            results = results.boxes
            num_boxes = len(results)
            pred_logits = torch.zeros(num_queries, 2, device=device)
            pred_logits[:num_boxes, 0] = results.conf.to(device)
            pred_boxes = torch.zeros(num_queries, 4, device=device)
            pred_boxes[:num_boxes] = results.xywhn.to(device)
            pred_logits_batch.append(pred_logits)
            pred_boxes_batch.append(pred_boxes)
        pred_logits_batch = torch.stack(pred_logits_batch)
        pred_boxes_batch = torch.stack(pred_boxes_batch)

        outputs = {}
        detection_out = {'pred_logits': pred_logits_batch, 'pred_boxes': pred_boxes_batch}
        outputs['SoccerNetGSR_Detection'] = detection_out

        return outputs

