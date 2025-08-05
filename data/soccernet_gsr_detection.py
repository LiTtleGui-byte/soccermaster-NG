# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# TODO: 对没有gt目录的支持，比如challenge

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from collections import defaultdict
from torch.utils.data import Dataset
import random
import math
from math import floor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xywh, bbox_xywh_to_cxcywh
from data.utils import Compose, ToTensor, RandomResize, Normalize, get_image_hw
from data.soccernet_gsr_reid import role_mapping, jn_mapping, digit_head_mapping, digit_tail_mapping
from data.pnlcalib_utils.utils_keypoints import KeypointsDB
from data.pnlcalib_utils.utils_lines import LineKeypointsDB
import copy

from sn_calibration.src.evaluate_extremities import mirror_labels

class SoccerNetGSR_Detection(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            load_annotation: bool = True,
            transforms=None,
            num_keypoints: int = 58,
            num_lines: int = 24,
            detection_data_type: str = "image",
            backbone_type: str = "image",
            num_frames: int = 30,
            image_input_size: int = 512,
            detect_ball: bool = False,
            detect_ball_only: bool = False,
    ):
        super(SoccerNetGSR_Detection, self).__init__()
        assert split in ['train', 'valid', 'test']
        
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation
        self.transforms = transforms
        self.num_keypoints = num_keypoints
        self.num_lines = num_lines
        self.detection_data_type = detection_data_type
        self.backbone_type = backbone_type
        self.num_frames = num_frames
        self.image_input_size = image_input_size
        self.detect_ball = detect_ball
        self.detect_ball_only = detect_ball_only
        
        # Validate configuration
        if self.detect_ball_only and self.detect_ball:
            print("Warning: Both detect_ball_only and detect_ball are set to True. detect_ball_only takes precedence.")

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        
        assert self.load_annotation, "For SoccerNetGSR_Detection, annotations are required."
        
        if self.load_annotation:
            self.annotations = self._get_annotations()
            self.ann_is_legals = self._decouple_is_legal()
            self.set_sample_position()
            
        return

    def get_sequence_infos(self):
        return self.sequence_infos

    def get_image_paths(self):
        return self.image_paths

    def get_annotations(self):
        if self.load_annotation:
            return self.annotations
        else:
            raise ValueError("Annotations are not loaded.")

    def _get_sequence_names(self):
        sequence_names = os.listdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split))
        return [name for name in sequence_names if os.path.isdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split, name))]
        # if self.split == 'train':
        #     return [name for name in sequence_names if os.path.isdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split, name))][:1]
        # elif self.split == 'test':
        #     sequence_names = [name for name in sequence_names if os.path.isdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split, name))]
        #     sequence_names.sort()
        #     return sequence_names

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            metadata_path = os.path.join(sequence_dir, "Labels-GameState.json")
            metadata = json.load(open(metadata_path))
            sequence_infos[sequence_name] = {
                "width": 1920,
                "height": 1080,
                "length": int(metadata['info']['seq_length']),
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, i))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, 'SoccerNetGS', split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx+1:06d}.jpg"))    # the image name is 1-indexed
            
    def _init_annotations(self, sequence_names):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": [],
                    "category": [],
                    "bbox": [],
                    "visibility": [],
                    "role": [],
                    "jersey": [],
                    "digit_head": [],
                    "digit_tail": [],
                    "legibility_score": [],
                    "valid_camera": torch.tensor(False, dtype=torch.bool),
                    "intrinsic": torch.zeros((3, 3), dtype=torch.float32),
                    "translation": torch.zeros((3, ), dtype=torch.float32),
                    "rotation_matrix": torch.eye(3, dtype=torch.float32),
                    "lines": {},
                })
        return annotations
    
    def _get_annotations(self):
        
        # get legibility jn info
        legibility_jn_json_path = os.path.join(self.data_dir, 'legibility_jn', f'{self.split}.json')
        with open(legibility_jn_json_path, 'r') as f:
            legibility_jn_info = json.load(f)
        legibility_jn_dict = {}
        for [sequence_id, image_id, track_id, jn, legibility] in legibility_jn_info:
            legibility_jn_dict.update({(sequence_id, image_id, track_id): (jn, legibility)})
        
        sequence_names = self._get_sequence_names()
        # Init the annotations:
        annotations = self._init_annotations(sequence_names)
        # Load the annotations:
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "Labels-GameState.json")
            gt = json.load(open(gt_file_path))
            annos = gt['annotations']
            for anno in annos:
                # Filter based on detect_ball and detect_ball_only parameters
                if self.detect_ball_only:
                    # Only include ball (exclude person)
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] == 'ball') or (anno['supercategory']== 'pitch')):
                        continue
                elif self.detect_ball:
                    # Include both person and ball
                    if not ((anno['supercategory'] == 'object') or (anno['supercategory']== 'pitch')):
                        continue
                else:
                    # Only include person (exclude ball)
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] != 'ball') or (anno['supercategory']== 'pitch')):
                        continue
                
                frame_idx = int(anno['image_id'][-6:]) - 1
                if anno['supercategory'] == 'object':
                    obj_id = anno['track_id']
                    x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                    bbox = [x, y, w, h]
                    visibility = 1.0
                    
                    # Set category based on detection mode
                    if self.detect_ball_only:
                        # Only ball detection: ball -> 0
                        if anno['attributes']['role'] == 'ball':
                            category = 0
                        else:
                            # This should not happen due to filtering, but handle it gracefully
                            continue
                    else:
                        # Normal or ball+person detection: person -> 0, ball -> 1
                        if anno['attributes']['role'] == 'ball':
                            category = 1
                        else:
                            category = 0
                    
                    # Append to lists instead of using torch.cat
                    annotations[sequence_name][frame_idx]["id"].append(obj_id)
                    annotations[sequence_name][frame_idx]["category"].append(category)
                    annotations[sequence_name][frame_idx]["bbox"].append(bbox)
                    annotations[sequence_name][frame_idx]["visibility"].append(visibility)
                    
                    # For ball, set all attributes to default values
                    if anno['attributes']['role'] == 'ball':
                        annotations[sequence_name][frame_idx]["role"].append(role_mapping['ball'])
                        annotations[sequence_name][frame_idx]["legibility_score"].append(0.0)
                        annotations[sequence_name][frame_idx]["jersey"].append(jn_mapping[None])
                        annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                        annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                    else:
                        # For person, process attributes as before
                        annotations[sequence_name][frame_idx]["role"].append(role_mapping[anno['attributes']['role']])
                        
                        # get legibility score $$ filtered jn
                        sequence_id = sequence_name[-3:]
                        image_id = anno['image_id']
                        track_id = anno['track_id']
                        legibility_score = legibility_jn_dict[(sequence_id, image_id, track_id)][1]
                        annotations[sequence_name][frame_idx]["legibility_score"].append(legibility_score)
                        jn = anno['attributes']['jersey'] if legibility_score > 0.5 else None
                        annotations[sequence_name][frame_idx]["jersey"].append(jn_mapping[jn])
                        # get digit head and digit tail
                        if jn is not None:
                            if len(jn) == 1:
                                # 对于1位数，设置tail为该数字
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn])
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            elif len(jn) == 2:
                                # 对于2位数，设置head为高位，tail为低位
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[jn[0]])
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn[1]])
                            else:
                                # 对于其他情况，设置为None
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                        else:
                            # 如果jersey number为None，则digit_head和digit_tail也设置为None
                            annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                elif anno['supercategory']== 'pitch':
                    # annotations[sequence_name][frame_idx]['lines'] = self.correct_lines_labels(anno['lines'])
                    annotations[sequence_name][frame_idx]['lines'] = anno['lines']
                else:
                    raise ValueError(f"Unknown annotation: {anno}")
                
            # Load the camera parameters:
            camera_path = os.path.join(self.data_dir, "camera_params", self.split, f"{sequence_name}.json")
            camera_params = json.load(open(camera_path))
            for frame_id, value in camera_params.items():
                frame_idx = int(frame_id[-6:]) - 1
                
                params = None
                if value["ransac_params"] is None:
                    params = value["all_points_params"]
                else:
                    all_reprojection_error_by_ransac = value["all_reprojection_error_by_ransac"]
                    all_points_params_reprojection_error = value["all_points_params"]["reprojection_error"]
                    if all_reprojection_error_by_ransac < all_points_params_reprojection_error:
                        params = value["ransac_params"]
                    else:
                        params = value["all_points_params"]
                assert params is not None, f"Camera parameters are not found for frame {frame_id} in sequence {sequence_name}."
                
                annotations[sequence_name][frame_idx]["valid_camera"] = torch.tensor(True, dtype=torch.bool)
                # annotations[sequence_name][frame_idx]["fxy"] = torch.tensor([params["x_focal_length"], params["y_focal_length"]])
                # annotations[sequence_name][frame_idx]["pxy"] = torch.tensor(params["principal_point"])
                annotations[sequence_name][frame_idx]["intrinsic"] = torch.tensor([[params["x_focal_length"], 0, params["principal_point"][0]], [0, params["y_focal_length"], params["principal_point"][1]], [0, 0, 1]])
                annotations[sequence_name][frame_idx]["translation"] = torch.tensor(params["position_meters"])
                annotations[sequence_name][frame_idx]["rotation_matrix"] = torch.tensor(params["rotation_matrix"])
        
        # Convert lists to tensors in a single operation per frame
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                frame_annotation = annotations[sequence_name][i]
                if len(frame_annotation["id"]) > 0:
                    frame_annotation["id"] = torch.tensor(frame_annotation["id"], dtype=torch.int64)
                    frame_annotation["category"] = torch.tensor(frame_annotation["category"], dtype=torch.int64)
                    frame_annotation["bbox"] = torch.tensor(frame_annotation["bbox"], dtype=torch.float32)
                    frame_annotation["visibility"] = torch.tensor(frame_annotation["visibility"], dtype=torch.float32)
                    frame_annotation["role"] = torch.tensor(frame_annotation["role"], dtype=torch.int64)
                    frame_annotation["jersey"] = torch.tensor(frame_annotation["jersey"], dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.tensor(frame_annotation["digit_head"], dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.tensor(frame_annotation["digit_tail"], dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.tensor(frame_annotation["legibility_score"], dtype=torch.float32)
                else:
                    # Empty frame
                    frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                    frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)
                    frame_annotation["role"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["jersey"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.zeros((0, ), dtype=torch.float32)

        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations
    
    def correct_lines_labels(self, data):
        if 'Goal left post left' in data.keys():
            data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
            del data['Goal left post left']

        return data
    
    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(list)
        for sequence_name in self.annotations:
            for frame_id, annotation in enumerate(self.annotations[sequence_name]):
                decoupled_is_legal[sequence_name].append(annotation["is_legal"])
        # Reformat the 'is_legal' attribute from a list to a tensor,
        # which is more convenient for the sampling process (calculation-friendly).
        decoupled_is_legal_in_tensor = defaultdict(torch.Tensor)
        for sequence_name in decoupled_is_legal:
            decoupled_is_legal_in_tensor[sequence_name] = torch.tensor(
                decoupled_is_legal[sequence_name], dtype=torch.bool
            )
        return decoupled_is_legal_in_tensor

    def set_sample_position(self):
        """
        Set the position of each legal sample.
        For test split in video mode, only frames where frame_idx % num_frames == 0 can be starting points.
        Also ensures that starting position + num_frames doesn't exceed sequence length.
        """
        self.sample_position = list()
        for sequence_name in self.annotations:
            sequence_length = self.sequence_infos[sequence_name]["length"]
            for frame_idx in range(len(self.annotations[sequence_name])):
                if self.annotations[sequence_name][frame_idx]["is_legal"]:
                    # 在test阶段的video模式下，只有frame_idx能被num_frames整除的才能作为起点
                    if (self.detection_data_type == "video" and 
                        self.backbone_type == "video" and 
                        self.split == "test"):
                        # 只有当frame_idx能被num_frames整除时，且不会超出序列长度时，才能作为起点
                        if (frame_idx % self.num_frames == 0 and 
                            frame_idx + self.num_frames <= sequence_length):
                            self.sample_position.append((sequence_name, frame_idx))
                    elif self.detection_data_type == "video" and self.backbone_type == "video":
                        # 其他video模式下，确保不会超出序列长度
                        if frame_idx + self.num_frames <= sequence_length:
                            self.sample_position.append((sequence_name, frame_idx))
                    else:
                        # image模式，保持原有逻辑
                        self.sample_position.append((sequence_name, frame_idx))
        return
    
    def __len__(self):
        return len(self.sample_position)
    
    def format_data(self, image, annotation, metas):
        if self.transforms is not None:
                image, annotation, metas = self.transforms(image, annotation, metas)
            
        # used for DETR loss:
        annotation['boxes'] = annotation['bbox']
        annotation['labels'] = annotation['category']
        annotation['roles'] = annotation['role']
        # use for camera loss:
        annotation['quaternion'] = mat_to_quat(annotation['rotation_matrix'].unsqueeze(0)).squeeze(0)
        H, W = metas['image_size']
        fov_h = 2 * torch.atan((H / 2) / annotation['intrinsic'][1, 1])
        fov_w = 2 * torch.atan((W / 2) / annotation['intrinsic'][0, 0])
        annotation['fov_hw'] = torch.stack([fov_h, fov_w])
        # use for keypoints detection:
        # print(annotation['lines'])
        try:
            line_db = LineKeypointsDB(annotation['lines'], image)
            lines_target = line_db.get_tensor()
            annotation['lines_target'] = torch.tensor(lines_target, dtype=torch.float32)
            annotation['valid_lines'] = torch.tensor(True, dtype=torch.bool)
        except Exception as e:
            annotation['lines_target'] = torch.zeros((self.num_lines, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
            annotation['valid_lines'] = torch.tensor(False, dtype=torch.bool)
        try:
            keypoints = KeypointsDB(self.correct_lines_labels(annotation['lines']), image)
            keypoints_target, keypoints_mask = keypoints.get_tensor_w_mask()
            annotation['keypoints_target'] = torch.tensor(keypoints_target, dtype=torch.float32)
            annotation['keypoints_mask'] = torch.tensor(keypoints_mask, dtype=torch.float32)
            annotation['valid_keypoints'] = torch.tensor(True, dtype=torch.bool)
        except Exception as e:
            annotation['keypoints_target'] = torch.zeros((self.num_keypoints, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
            annotation['keypoints_mask'] = torch.zeros((self.num_keypoints), dtype=torch.float32)
            annotation['valid_keypoints'] = torch.tensor(False, dtype=torch.bool)
        
        return image, annotation, metas
        
    
    def __getitem__(self, index):
        sequence_name, frame_idx = self.sample_position[index]
        
        # Check if we need to collect multiple frames for video mode
        if self.detection_data_type == "video" and self.backbone_type == "video":
            # Collect consecutive frames starting from frame_idx
            sequence_length = self.sequence_infos[sequence_name]["length"]
            
            # Calculate the range of frames to collect
            start_frame = frame_idx
            end_frame = min(start_frame + self.num_frames, sequence_length)
            actual_num_frames = end_frame - start_frame
            
            # Collect images for all frames
            images = []
            for i in range(start_frame, end_frame):
                image_path = self.image_paths[sequence_name][i]
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            
            # If we don't have enough frames, repeat the last frame to reach num_frames
            # while len(images) < self.num_frames:
            #     images.append(images[-1])  # Repeat the last frame
            
            annotations = []
            for i in range(start_frame, end_frame):
                annotations.append(copy.deepcopy(self.annotations[sequence_name][i]))
            
            metas = {"task": 'SoccerNetGSR_Detection',
                    "split": self.split,
                    "sequence": sequence_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "actual_num_frames": actual_num_frames,
                    "total_frames": self.num_frames,
                    "is_static": self.sequence_infos[sequence_name]["is_static"],
                    "size_divisibility": 1,}
            
            for i in range(len(images)):
                images[i], annotations[i], _ = self.format_data(images[i], annotations[i], {})
                
            # 需要做padding和collate吗？暂时先不做
            # if len(images) < self.num_frames:
            #     last_image = images[-1]
            #     images.append(last_image)
            #     annotations.append(copy.deepcopy(annotations[-1]))
            images = torch.stack(images, dim=0)
            # for k, v in annotations.items():
            #     annotations[k] = torch.stack(v, dim=0)
            
            return images, annotations, metas
        else:
            # Original single-frame mode
            image_path = self.image_paths[sequence_name][frame_idx]
            image = Image.open(image_path).convert("RGB")
            annotation = copy.deepcopy(self.annotations[sequence_name][frame_idx])
            metas = {"task": 'SoccerNetGSR_Detection',
                    "split": self.split,
                    "sequence": sequence_name,
                    "frame_idx": frame_idx,
                    "is_static": self.sequence_infos[sequence_name]["is_static"],
                    "size_divisibility": 1,}
            image, annotation, metas = self.format_data(image, annotation, metas)
            return image, annotation, metas

def build_gsr_detection_dataset(config: dict, split: str):
    dataset = SoccerNetGSR_Detection(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        load_annotation=True,
        transforms=build_transforms(config),
        num_keypoints=config["NUM_KEYPOINTS"],
        num_lines=config["NUM_LINES"],
        detection_data_type=config["DETECTION_DATA_TYPE"],
        backbone_type=config["BACKBONE_TYPE"],
        num_frames=config["NUM_FRAMES"],
        image_input_size=config["AUG_MAX_SIZE"],
        detect_ball=config["DETR_DETECT_BALL"],
        detect_ball_only=config["DETECT_BALL_ONLY"],
    )
    return dataset

def build_gsr_detection_dataloader(config: dict, split: str):
    dataset = build_gsr_detection_dataset(config, split)
    shuffle = True if split == "train" else False
    prefetch_factor = config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None
    persistent_workers = config["NUM_WORKERS"] > 0
    return DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=shuffle, collate_fn=collate_fn, num_workers=config["NUM_WORKERS"], prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."

    assert len(annotation["id"]) == len(annotation["category"]) \
           == len(annotation["bbox"]) == len(annotation["visibility"]), \
           "The length of 'id', 'category', 'bbox', 'visibility' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(0)     # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as an parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    # return len(annotation["id"]) > 0
    return len(annotation["id"]) > 0 and _id_unique and leq_300

def append_annotation(
        annotation: dict,
        obj_id: int,
        category: int,
        bbox: list,
        visibility: float,
):
    annotation["id"] = torch.cat([
        annotation["id"],
        torch.tensor([obj_id], dtype=torch.int64)
    ])
    annotation["category"] = torch.cat([
        annotation["category"],
        torch.tensor([category], dtype=torch.int64)
    ])
    annotation["bbox"] = torch.cat([
        annotation["bbox"],
        torch.tensor([bbox], dtype=torch.float32)
    ])
    annotation["visibility"] = torch.cat([
        annotation["visibility"],
        torch.tensor([visibility], dtype=torch.float32)
    ])
    return annotation

class BoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_xywh_to_xyxy(annotation["bbox"])
        return image, annotation, metas


class BoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_xyxy_to_cxcywh(annotation["bbox"])
        return image, annotation, metas

class BoxCXCYWHtoXYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_cxcywh_to_xywh(annotation["bbox"])
        return image, annotation, metas

class BoxXYWHtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = bbox_xywh_to_cxcywh(annotation["bbox"])
        return image, annotation, metas

FLIP_POSTS = {
    'Goal left post right': 'Goal left post left ',
    'Goal left post left ': 'Goal left post right',
    'Goal right post right': 'Goal right post left',
    'Goal right post left': 'Goal right post right'
}

h_lines = ['Goal left crossbar', 'Side line left', 'Small rect. left main', 'Big rect. left main', 'Middle line',
                   'Big rect. right main', 'Small rect. right main', 'Side line right', 'Goal right crossbar']

v_lines = ['Side line top', 'Big rect. left top', 'Small rect. left top', 'Small rect. left bottom',
                   'Big rect. left bottom', 'Big rect. right top', 'Small rect. right top', 'Small rect. right bottom',
                              'Big rect. right bottom', 'Side line bottom']

def swap_top_bottom_names(line_name: str) -> str:
    x: str = 'top'
    y: str = 'bottom'
    if x in line_name or y in line_name:
        return y.join(part.replace(y, x) for part in line_name.split(x))
    return line_name


def swap_posts_names(line_name: str) -> str:
    if line_name in FLIP_POSTS:
        return FLIP_POSTS[line_name]
    return line_name

def flip_annot_names(annot, swap_top_bottom: bool = True, swap_posts: bool = True):
    annot = mirror_labels(annot)
    if swap_top_bottom:
        annot = {swap_top_bottom_names(k): v for k, v in annot.items()}
    if swap_posts:
        annot = {swap_posts_names(k): v for k, v in annot.items()}
    return annot

class LRAmbiguityFix():
    def __init__(self, v_th=70, h_th=20):
        self.v_th = v_th
        self.h_th = h_th

    def __call__(self, image, annotation, metas):
        data = annotation['lines']

        if len(data) == 0:
            return image, annotation, metas

        n_left, n_right = self.compute_n_sides(data)

        angles_v, angles_h = [], []
        for line in data.keys():
            line_points = []
            for point in data[line]:
                line_points.append((point['x'], point['y']))

            sorted_points = sorted(line_points, key=lambda point: (point[0], point[1]))
            pi, pf = sorted_points[0], sorted_points[-1]
            if line in h_lines:
                angle_h = self.calculate_angle_h(pi[0], pi[1], pf[0], pf[1])
                if angle_h:
                    angles_h.append(abs(angle_h))
            if line in v_lines:
                angle_v = self.calculate_angle_v(pi[0], pi[1], pf[0], pf[1])
                if angle_v:
                    angles_v.append(abs(angle_v))


        if len(angles_h) > 0 and len(angles_v) > 0:
            if np.mean(angles_h) < self.h_th and np.mean(angles_v) < self.v_th:
                if n_right > n_left:
                    data = flip_annot_names(data, swap_top_bottom=False, swap_posts=False)
        annotation['lines'] = data

        return image, annotation, metas

    def calculate_angle_h(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(slope)
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None
    def calculate_angle_v(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(1 / slope) if slope != 0 else math.pi / 2  # Avoid division by zero
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None

    def compute_n_sides(self, data):
        n_left, n_right = 0, 0
        for line in data:
            line_words = line.split()[:3]
            if 'left' in line_words:
                n_left += 1
            elif 'right' in line_words:
                n_right += 1
        return n_left, n_right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(v_th={self.v_th}, h_th={self.h_th})"

def build_transforms(config: dict):

    return Compose([
        ToTensor(),
        RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
        BoxXYWHtoCXCYWH(),
        LRAmbiguityFix(),
    ])
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    images = torch.stack(images)
    
    # keys need to concat
    # concat_keys = ['valid_camera', 'quaternion', 'translation', 'fov_hw']
    
    # new_annotations = []
    # for key in annotations[0]:
    #     new_annotations[key] = torch.stack([anno[key] for anno in annotations])

    return {
        "images": images,
        "annotations": annotations,
        # "annotations": new_annotations,
        "metas": metas,
    }
    
def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
    
def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)
