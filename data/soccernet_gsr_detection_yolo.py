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
import cv2
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xywh, bbox_xywh_to_cxcywh
from data.utils import Compose, ToTensor
import copy

class Normalize:
    def __init__(self):
        pass
    def __call__(self, image, annotation, metas):
        # image = image.to(torch.float32)
        if "bbox" in annotation:
            h, w = image.shape[:2]
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return image, annotation, metas

class SoccerNetGSR_Detection_YOLO(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            load_annotation: bool = True,
            transforms=None,
            detection_data_type: str = "image",
            backbone_type: str = "image",
            num_frames: int = 30,
            detect_ball: bool = False,
            detect_ball_only: bool = False,
    ):
        super(SoccerNetGSR_Detection_YOLO, self).__init__()
        assert split in ['train', 'valid', 'test']
        
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation
        self.transforms = transforms
        self.detection_data_type = detection_data_type
        self.backbone_type = backbone_type
        self.num_frames = num_frames
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
                })
        return annotations
    
    def _get_annotations(self):
        
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
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] == 'ball')):
                        continue
                elif self.detect_ball:
                    # Include both person and ball
                    if not ((anno['supercategory'] == 'object')):
                        continue
                else:
                    # Only include person (exclude ball)
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] != 'ball')):
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
                else:
                    raise ValueError(f"Unknown annotation: {anno}")
                
        
        # Convert lists to tensors in a single operation per frame
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                frame_annotation = annotations[sequence_name][i]
                if len(frame_annotation["id"]) > 0:
                    frame_annotation["id"] = torch.tensor(frame_annotation["id"], dtype=torch.int64)
                    frame_annotation["category"] = torch.tensor(frame_annotation["category"], dtype=torch.int64)
                    frame_annotation["bbox"] = torch.tensor(frame_annotation["bbox"], dtype=torch.float32)
                    frame_annotation["visibility"] = torch.tensor(frame_annotation["visibility"], dtype=torch.float32)
                else:
                    # Empty frame
                    frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                    frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)

        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations
    
    
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
                
            images = torch.stack(images, dim=0)
            
            return images, annotations, metas
        else:
            # Original single-frame mode
            image_path = self.image_paths[sequence_name][frame_idx]
            # image = Image.open(image_path).convert("RGB")
            image = cv2.imread(image_path)
            annotation = copy.deepcopy(self.annotations[sequence_name][frame_idx])
            metas = {"task": 'SoccerNetGSR_Detection',
                    "split": self.split,
                    "sequence": sequence_name,
                    "frame_idx": frame_idx,
                    "is_static": self.sequence_infos[sequence_name]["is_static"],
                    "size_divisibility": 1,}
            image, annotation, metas = self.format_data(image, annotation, metas)
            return image, annotation, metas

def build_gsr_detection_yolo_dataset(config: dict, split: str):
    dataset = SoccerNetGSR_Detection_YOLO(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        load_annotation=True,
        transforms=build_transforms(config),
        detection_data_type=config["DETECTION_DATA_TYPE"],
        backbone_type=config["BACKBONE_TYPE"],
        num_frames=config["NUM_FRAMES"],
        detect_ball=config["DETR_DETECT_BALL"],
        detect_ball_only=config["DETECT_BALL_ONLY"],
    )
    return dataset

def build_gsr_detection_yolo_dataloader(config: dict, split: str):
    dataset = build_gsr_detection_yolo_dataset(config, split)
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


def build_transforms(config: dict):
    return Compose([
        # ToTensor(),
        Normalize(),
        BoxXYWHtoCXCYWH(),
    ])
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    # images = torch.stack(images)

    return {
        "images": images,
        "annotations": annotations,
        "metas": metas,
    }
    
