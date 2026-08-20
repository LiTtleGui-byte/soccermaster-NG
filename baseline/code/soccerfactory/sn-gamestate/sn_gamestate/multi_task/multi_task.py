import sys
sys.path.append('/remote-home/haolinyang/sports/Soccer-Backbone')  # 替换为实际的目录B路径

import torch
import pandas as pd
from tracklab.pipeline.imagelevel_module import ImageLevelModule
from typing import Any
import torch.nn.functional as F
import numpy as np
from PIL import Image

from runtime_option import runtime_option
from configs.util import load_super_config, update_config, yaml_to_dict
from models.multi_task import MultiTaskingSigLIP
from tracklab.utils.coordinates import ltrb_to_ltwh
from data.soccernet_gsr_reid import role_mapping, jn_mapping, digit_head_mapping, digit_tail_mapping
from nbjw_calib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, \
                                            get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, \
                                            coords_to_dict)

import torchvision.transforms as T

def kp_to_line(keypoints):
    line_keypoints_match = {"Big rect. left bottom": [24, 68, 25],
                            "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
                            "Big rect. left top": [4, 62, 5],
                            "Big rect. right bottom": [26, 69, 27],
                            "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
                            "Big rect. right top": [6, 63, 7],
                            "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
                            "Circle left": [31,37, 47, 41, 34],
                            "Circle right": [33, 40, 55, 44, 36],
                            "Goal left crossbar": [16, 12],
                            "Goal left post left": [16, 17],
                            "Goal left post right": [12, 13],
                            "Goal right crossbar": [15, 19],
                            "Goal right post left": [15, 14],
                            "Goal right post right": [19, 18],
                            "Middle line": [2, 32, 51, 35, 29],
                            "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
                            "Side line left": [1, 4, 8, 13,17, 20, 24, 28],
                            "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
                            "Side line top": [1, 58, 59, 2, 60, 61, 3],
                            "Small rect. left bottom": [20, 21],
                            "Small rect. left main": [9, 21],
                            "Small rect. left top": [8, 9],
                            "Small rect. right bottom": [22, 23],
                            "Small rect. right main": [10, 22],
                            "Small rect. right top": [10, 11]}

    lines = {}
    for line_name, kp_indices in line_keypoints_match.items():
        line = []
        for idx in kp_indices:
            if idx in keypoints.keys():
                line.append({'x': keypoints[idx]['x'], 'y': keypoints[idx]['y']})

        if line:
            lines[line_name] = line

    return lines

def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = torch.stack([b["image"] for _, b in batch])
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)

class MultiTask(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": ["keypoints", 
                  "lines"],
        "detection": ["image_id", 
                      "video_id", 
                      "category_id", 
                      "bbox_ltwh", 
                      "bbox_conf", 
                      "embeddings", 
                      "role_detection", 
                      "role_confidence",
                      "jersey_number_detection",
                      "jersey_number_confidence",
                      "visibility_scores"
                      ]
    }

    def __init__(self, config_path, STAGE_1_CKPT_DIR, batch_size, device, **kwargs):
        super().__init__(batch_size)
        cfg = yaml_to_dict(config_path)
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
        self.cfg = cfg
        self.batch_size = batch_size
        
        self.device = device
        self.model = MultiTaskingSigLIP(config=cfg)
        self.model.load_checkpoint(STAGE_1_CKPT_DIR)
        self.model.to(device)
        self.model.eval()
        self.id = 0
        
        self.tfms_resize = T.Compose([T.ToTensor(), T.Resize((512, 512))])
        
        datasets_to_heads = cfg["DATASETS_TO_HEADS"]
        all_heads = []
        for dataset_name, heads in datasets_to_heads.items():
            all_heads.extend(heads)
        all_heads = list(set(all_heads))
        all_heads.sort()
        self.all_heads = all_heads
        
        self.role_mapping_reverse = {v: k for k, v in role_mapping.items()}
        self.jn_mapping_reverse = {v: k for k, v in jn_mapping.items()}
        self.digit_head_mapping_reverse = {v: k for k, v in digit_head_mapping.items()}
        self.digit_tail_mapping_reverse = {v: k for k, v in digit_tail_mapping.items()}

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        shape = (image.shape[1], image.shape[0])
        image = self.tfms_resize(image)
        return {
            "image": image,
            "shape": shape,
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        images = images.to(self.device)
        results = self.model(images, dataset_name='SoccerNetGSR_Detection')
        SoccerNetGSR_Detection = results['SoccerNetGSR_Detection']
        heatmaps = results['KeypointsDetection']['pred_keypoints_heatmap']
        heatmaps_l = results['LinesDetection']['pred_lines_heatmap']
        
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)
        image_width = images.size()[-1]
        image_height = images.size()[-2]
        final_dict = complete_keypoints(kp_dict, lines_dict, w=image_width, h=image_height, normalize=True)
        all_images_pred = []
        for result, idx in zip(final_dict, metadatas.index):
            all_images_pred.append(pd.Series({"keypoints": result, "lines": kp_to_line(result)}, name=idx,))
        
        # 处理人员检测结果
        all_detections = []
        
        for batch_idx, (_, metadata) in enumerate(metadatas.iterrows()):
            # 获取当前batch的预测结果
            pred_logits = SoccerNetGSR_Detection['pred_logits'][batch_idx]  # [num_queries, num_classes]
            pred_boxes = SoccerNetGSR_Detection['pred_boxes'][batch_idx]    # [num_queries, 4]
            pred_roles = SoccerNetGSR_Detection['pred_roles'][batch_idx]    # [num_queries, num_role_classes]
            pred_jn_holistic = SoccerNetGSR_Detection['pred_jn_holistic'][batch_idx]  # [num_queries, num_jn_classes]
            pred_digit_head = SoccerNetGSR_Detection['pred_digit_head'][batch_idx]    # [num_queries, num_digit_head_classes]
            pred_digit_tail = SoccerNetGSR_Detection['pred_digit_tail'][batch_idx]    # [num_queries, num_digit_tail_classes]
            
            # 对分类结果进行softmax
            pred_probs = pred_logits.sigmoid()
            
            # 筛选person类别（类别0）且概率大于0.5的检测结果
            person_probs = pred_probs[:, 0]  # person类别的概率
            valid_mask = person_probs > 0.5
            
            if valid_mask.sum() > 0:
                # 获取有效的检测结果
                valid_boxes = pred_boxes[valid_mask]        # [N, 4]
                valid_probs = person_probs[valid_mask]      # [N]
                valid_roles = pred_roles[valid_mask]        # [N, num_role_classes]
                valid_jn_holistic = pred_jn_holistic[valid_mask]  # [N, num_jn_classes]
                valid_digit_head = pred_digit_head[valid_mask]    # [N, num_digit_head_classes]
                valid_digit_tail = pred_digit_tail[valid_mask]    # [N, num_digit_tail_classes]
                
                # 获取属性预测（取最大概率的索引）
                role_preds = torch.argmax(valid_roles, dim=-1)         # [N]
                jn_preds = torch.argmax(valid_jn_holistic, dim=-1)     # [N]
                digit_head_preds = torch.argmax(valid_digit_head, dim=-1)  # [N]
                digit_tail_preds = torch.argmax(valid_digit_tail, dim=-1)  # [N]
                
                # 获取属性的置信度
                role_confs = torch.max(F.softmax(valid_roles, dim=-1), dim=-1)[0]  # [N]
                jn_confs = torch.max(F.softmax(valid_jn_holistic, dim=-1), dim=-1)[0]  # [N]
                digit_head_confs = torch.max(F.softmax(valid_digit_head, dim=-1), dim=-1)[0]  # [N]
                digit_tail_confs = torch.max(F.softmax(valid_digit_tail, dim=-1), dim=-1)[0]  # [N]
                
                # 转换边界框格式从cxcywh到ltwh，并转换到原始图像尺寸
                original_shape = shapes[batch_idx]  # (width, height)
                img_w, img_h = original_shape
                
                # 将归一化的边界框转换为绝对坐标
                boxes_abs = valid_boxes.clone()
                boxes_abs[:, 0] *= img_w  # center_x
                boxes_abs[:, 1] *= img_h  # center_y
                boxes_abs[:, 2] *= img_w  # width
                boxes_abs[:, 3] *= img_h  # height
                
                # 从cxcywh转换为ltwh格式
                boxes_ltwh = boxes_abs.clone()
                boxes_ltwh[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2  # left = center_x - width/2
                boxes_ltwh[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2  # top = center_y - height/2
                # width和height保持不变
                
                # 获取原始图像用于crop
                original_image = images[batch_idx]  # tensor格式 [C, H, W]
                
                # 从检测框中crop子图并收集当前图像的所有crops
                num_detections = len(valid_boxes)
                current_crops = []
                current_detections = []
                
                for i in range(num_detections):
                    # 获取边界框坐标 (ltwh格式)
                    left, top, width, height = boxes_ltwh[i].cpu().numpy()
                    
                    # 转换为xyxy格式并进行边界检查
                    x1 = max(0, int(left))
                    y1 = max(0, int(top))
                    x2 = min(img_w, int(left + width))
                    y2 = min(img_h, int(top + height))
                    
                    # 确保crop区域有效
                    if x2 > x1 and y2 > y1:
                        # 从tensor中crop子图 (原始图像是512x512，需要按比例计算)
                        # 注意：原始图像已经被resize到512x512
                        resize_scale_w = 512.0 / img_w
                        resize_scale_h = 512.0 / img_h
                        
                        crop_x1 = int(x1 * resize_scale_w)
                        crop_y1 = int(y1 * resize_scale_h)
                        crop_x2 = int(x2 * resize_scale_w)
                        crop_y2 = int(y2 * resize_scale_h)
                        
                        # 确保crop坐标在有效范围内
                        crop_x1 = max(0, min(crop_x1, 511))
                        crop_y1 = max(0, min(crop_y1, 511))
                        crop_x2 = max(crop_x1 + 1, min(crop_x2, 512))
                        crop_y2 = max(crop_y1 + 1, min(crop_y2, 512))
                        
                        # 进行crop
                        cropped_image = original_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # Resize到512x512
                        cropped_image = F.interpolate(
                            cropped_image.unsqueeze(0), 
                            size=(512, 512), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        
                        current_crops.append(cropped_image)
                    else:
                        # 如果crop区域无效，使用全零tensor
                        current_crops.append(torch.zeros_like(original_image))
                        
                    # if jn_confs[i].cpu().item() > 0.5:
                    #     jn_detection = self.jn_mapping_reverse[jn_preds[i].cpu().item()]
                    # else:
                    #     jn_detection = None
                    # jn_confidence = jn_confs[i].cpu().item()
                    
                    if digit_head_confs[i].cpu().item() > 0.5 and digit_tail_confs[i].cpu().item() > 0.5:
                        digit_head_detection = self.digit_head_mapping_reverse[digit_head_preds[i].cpu().item()]
                        digit_tail_detection = self.digit_tail_mapping_reverse[digit_tail_preds[i].cpu().item()]
                        if digit_head_detection == None:
                            digit_head_detection = ''
                        if digit_tail_detection == None:
                            digit_tail_detection = ''
                        jn_detection = digit_head_detection + digit_tail_detection
                        if jn_detection == '':
                            jn_detection = None
                        jn_confidence = digit_head_confs[i].cpu().item() * digit_tail_confs[i].cpu().item()
                    else:
                        jn_detection = None
                        jn_confidence = 0.25
                    
                    # 创建detection记录（暂时没有embeddings）
                    detection = pd.Series({
                        'image_id': metadata.name,
                        'video_id': metadata.video_id,
                        'category_id': 1,  # person类别
                        'bbox_ltwh': boxes_ltwh[i].cpu().numpy(),
                        'bbox_conf': valid_probs[i].cpu().item(),
                        'role_detection': self.role_mapping_reverse[role_preds[i].cpu().item()],
                        'role_confidence': role_confs[i].cpu().item(),
                        'jersey_number_detection': jn_detection,
                        'jersey_number_confidence': jn_confidence,
                        'visibility_scores': [True],
                    }, name=self.id)
                    self.id += 1
                    current_detections.append(detection)
                
                # 如果当前图像有有效检测，进行ReID特征提取
                if len(current_crops) > 0:
                    # 将当前图像的crops组织成batch
                    crops_batch = torch.stack(current_crops)  # [N, C, H, W]
                    crops_batch = crops_batch.to(self.device)
                    
                    # 根据batch_size进行切分，避免显存溢出
                    num_crops = crops_batch.size(0)
                    reid_embeddings_list = []
                    
                    for start_idx in range(0, num_crops, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, num_crops)
                        crops_sub_batch = crops_batch[start_idx:end_idx]
                        
                        # 输入到模型进行ReID特征提取
                        reid_results = self.model(crops_sub_batch, dataset_name='SoccerNetGSR_ReID')
                        sub_reid_embeddings = reid_results['SoccerNetGSR_ReID']['reid_embeddings']  # [sub_batch_size, embedding_dim]
                        reid_embeddings_list.append(sub_reid_embeddings)
                    
                    # 合并所有子batch的特征
                    reid_embeddings = torch.cat(reid_embeddings_list, dim=0)  # [N, embedding_dim]
                    
                    # 将ReID特征添加到对应的detection中
                    for i, detection in enumerate(current_detections):
                        detection['embeddings'] = reid_embeddings[i].unsqueeze(0).cpu().numpy()
                
                # 将当前图像的所有detections添加到总列表中
                all_detections.extend(current_detections)
        
        return pd.DataFrame(all_detections), pd.DataFrame(all_images_pred)
        
        