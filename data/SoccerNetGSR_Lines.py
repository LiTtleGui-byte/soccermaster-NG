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
from data.pnlcalib_utils.utils_keypoints import KeypointsDB
from data.pnlcalib_utils.utils_lines import LineKeypointsDB
import copy

from sn_calibration.src.evaluate_extremities import mirror_labels

class SoccerNetGSR_Lines(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            load_annotation: bool = True,
            transforms=None,
    ):
        super(SoccerNetGSR_Lines, self).__init__()
        assert split in ['train', 'valid', 'test']
        
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation
        self.transforms = transforms

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        
        assert self.load_annotation, "For SoccerNetGSR_Lines, annotations are required."
        
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
                    "lines": {},
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
                # 只处理pitch相关的标注
                if anno['supercategory'] == 'pitch':
                    frame_idx = int(anno['image_id'][-6:]) - 1
                    # annotations[sequence_name][frame_idx]['lines'] = self.correct_lines_labels(anno['lines'])
                    annotations[sequence_name][frame_idx]['lines'] = anno['lines']

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
        annotation = copy.deepcopy(self.annotations[sequence_name][frame_idx])
        metas = {"task": 'SoccerNetGSR_Lines',
                "split": self.split,
                "sequence": sequence_name,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[sequence_name]["is_static"],
                "size_divisibility": 1,}
        if self.transforms is not None:
            image, annotation, metas = self.transforms(image, annotation, metas)
            
        # use for keypoints detection:
        line_db = LineKeypointsDB(annotation['lines'], image)
        try:
            lines_target = line_db.get_tensor()
            annotation['lines_target'] = torch.tensor(lines_target, dtype=torch.float32)
        except Exception as e:
            if isinstance(e, OverflowError):
                # If overflow error occurs, try getting a different sample
                new_index = (index + 1) % len(self)
                return self.__getitem__(new_index)
            else:
                # For other exceptions, print error and raise
                print(e)
                print(annotation['lines'])
                raise e
        
        return image, annotation, metas

def build_gsr_lines_dataset(config: dict, split: str):
    dataset = SoccerNetGSR_Lines(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        load_annotation=True,
        transforms=build_transforms(config),
    )
    return dataset

def build_gsr_lines_dataloader(config: dict, split: str):
    dataset = build_gsr_lines_dataset(config, split)
    shuffle = True if split == "train" else False
    prefetch_factor = config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None
    return DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=shuffle, collate_fn=collate_fn, num_workers=config["NUM_WORKERS"], prefetch_factor=prefetch_factor)

def is_legal(annotation: dict):
    # 只检查是否有lines数据
    return len(annotation["lines"]) > 0

# 移除不需要的bbox变换类
class BoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        return image, annotation, metas


class BoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        return image, annotation, metas

class BoxCXCYWHtoXYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        return image, annotation, metas

class BoxXYWHtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
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
        LRAmbiguityFix(),
    ])
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    images = torch.stack(images)

    return {
        "images": images,
        "annotations": annotations,
        "metas": metas,
    }
