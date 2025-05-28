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
from PIL import Image
from torchvision.transforms import v2
from collections import defaultdict
from torch.utils.data import Dataset
from utils.nested_tensor import nested_tensor_from_tensor_list
import copy
import math
import torch
import einops
import random
from torchvision.transforms import v2
import torchvision.transforms as T
from math import floor
from PIL import Image
from triton.language import dtype
import numpy as np
from torch.utils.data import DataLoader
from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xywh, bbox_xywh_to_cxcywh

class SoccerNetGSR_Detection(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            load_annotation: bool = True,
            transforms=None,
    ):
        super(SoccerNetGSR_Detection, self).__init__()
        assert split in ['train', 'valid', 'test']
        
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation
        self.transforms = transforms

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
                if anno['supercategory'] != 'object' or anno['attributes']['role'] == 'ball':
                    continue
                frame_idx = int(anno['image_id'][-6:]) - 1
                obj_id = anno['track_id']
                x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                bbox = [x, y, w, h]
                category, visibility = 0, 1.0
                annotations[sequence_name][frame_idx] = append_annotation(
                    annotation=annotations[sequence_name][frame_idx],
                    obj_id=obj_id,
                    category=category,
                    bbox=bbox,
                    visibility=visibility,
                )
            
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

    def _init_annotations(self, sequence_names):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": torch.zeros((0, ), dtype=torch.int64),
                    "category": torch.zeros((0, ), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0, ), dtype=torch.float32),
                })
        return annotations
    
    def set_sample_position(self):
        """
        Set the position of each legal sample.
        """
        self.sample_position = list()
        for sequence_name in self.annotations:
            for frame_idx in range(len(self.annotations[sequence_name])):
                if self.annotations[sequence_name][frame_idx]["is_legal"]:
                    self.sample_position.append((sequence_name, frame_idx))
        return
    
    def __len__(self):
        return len(self.sample_position)
    
    def __getitem__(self, index):
        sequence_name, frame_idx = self.sample_position[index]
        image_path = self.image_paths[sequence_name][frame_idx]
        image = Image.open(image_path).convert("RGB")
        annotation = self.annotations[sequence_name][frame_idx]
        metas = {"task": 'SoccerNetGSR_Detection',
                "split": self.split,
                "sequence": sequence_name,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[sequence_name]["is_static"],
                "size_divisibility": 1,}
        if self.transforms is not None:
            image, annotation, metas = self.transforms(image, annotation, metas)
            
        # used for DETR loss:
        annotation['boxes'] = annotation['bbox']
        annotation['labels'] = annotation['category']
            
        return image, annotation, metas

def build_gsr_detection_dataset(config: dict, split: str):
    dataset = SoccerNetGSR_Detection(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        load_annotation=True,
        transforms=build_transforms(config),
    )
    return dataset

def build_gsr_detection_dataloader(config: dict, split: str):
    dataset = build_gsr_detection_dataset(config, split)
    return DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn, num_workers=config["NUM_WORKERS"])

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

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, annotation, metas):
        for transform in self.transforms:
            image, annotation, metas = transform(image, annotation, metas)
        return image, annotation, metas

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, annotation, metas):
        image = image.to(torch.float32).div(255)
        image = v2.functional.normalize(image, mean=self.mean, std=self.std)
        h, w = image.shape[-2:]
        annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return image, annotation, metas

def get_image_hw(image: torch.Tensor | list | Image.Image):
    if isinstance(image, torch.Tensor):
        return image.shape[-2], image.shape[-1]
    elif isinstance(image, list):
        return get_image_hw(image[0])
    elif isinstance(image, Image.Image):
        return image.height, image.width
    else:
        raise NotImplementedError("The input image type is not supported.")

class RandomResize:
    def __init__(self, sizes: list, max_size: int | None = None, keep_aspect_ratio: bool = True):
        self.sizes = sizes
        self.max_size = max_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, image, annotation, metas):
        new_size = random.choice(self.sizes)  # choose the size for images

        def get_new_hw(_curr_hw: list, _new_size) -> tuple[int, int]:
            _curr_h, _curr_w = _curr_hw
            if self.keep_aspect_ratio:
                if self.max_size is not None:  # need to restrict the longer side length
                    _min_hw, _max_hw = float(min(_curr_h, _curr_w)), float(max(_curr_h, _curr_w))
                    if _max_hw / _min_hw * _new_size > self.max_size:  # need to restrict the resize size
                        _new_size = int(floor(self.max_size * _min_hw / _max_hw))
                # Calculate the new height and width while maintaining aspect ratio:
                if _curr_w < _curr_h:
                    _new_w = _new_size
                    _new_h = int(round(_new_size * _curr_h / _curr_w))
                else:
                    _new_h = _new_size
                    _new_w = int(round(_new_size * _curr_w / _curr_h))
                return _new_h, _new_w
            else:
                # When not keeping aspect ratio, just use the same size for both dimensions
                return _new_size, _new_size

        new_hw = get_new_hw(get_image_hw(image), _new_size=new_size)    # new yx
        scale_ratio_x = new_hw[1] / get_image_hw(image)[1]
        scale_ratio_y = new_hw[0] / get_image_hw(image)[0]
        # Resize images:
        if isinstance(image, torch.Tensor):
            image = v2.functional.resize(image, new_hw)
        else:
            raise NotImplementedError(f"The input image type {type(image)} is not supported.")
        # Resize annotations:
        annotation["bbox"] = annotation["bbox"] * torch.as_tensor([scale_ratio_x, scale_ratio_y] * 2)
        return image, annotation, metas

class ToTensor:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        assert isinstance(image, Image.Image)
        image = v2.functional.to_image(image)
        return image, annotation, metas

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
        ToTensor(),
        RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
        BoxXYWHtoCXCYWH(),
    ])
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    images = torch.stack(images)
    
    # new_annotations = []
    # for key in annotations[0]:
    #     new_annotations[key] = torch.stack([anno[key] for anno in annotations])

    return {
        "images": images,
        "annotations": annotations,
        # "annotations": new_annotations,
        "metas": metas,
    }
