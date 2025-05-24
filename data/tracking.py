# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------

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

from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh

class OneDataset:
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "OneDataset",
            split: str = "train",
            load_annotation: bool = True,
    ):
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.load_annotation = load_annotation

        # Null data:
        self.sequence_infos, self.image_paths, self.annotations = None, None, None
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

class SeqDataset(Dataset):
    def __init__(
            self,
            seq_info,
            image_paths,
            max_shorter: int = 800,
            max_longer: int = 1536,
            keep_aspect_ratio: bool = True,
            size_divisibility: int = 0,
            dtype=torch.float32,
            mean: list[float] = [0.485, 0.456, 0.406],
            std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.seq_info = seq_info
        self.image_paths = image_paths
        self.max_shorter = max_shorter
        self.max_longer = max_longer
        self.size_divisibility = size_divisibility
        self.dtype = dtype

        self.transform = v2.Compose([
            v2.Resize(size=self.max_shorter, max_size=self.max_longer) if keep_aspect_ratio else v2.Resize(size=(self.max_shorter, self.max_longer)), # compatible with resnet and siglip
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = self._load(self.image_paths[item])
        transformed_image = self.transform(image)
        if self.dtype != torch.float32:
            transformed_image = transformed_image.to(self.dtype)
        transformed_image = nested_tensor_from_tensor_list([transformed_image], self.size_divisibility)
        return transformed_image, self.image_paths[item]

    def seq_hw(self):
        return self.seq_info["height"], self.seq_info["width"]

    @staticmethod
    def _load(path):
        image = Image.open(path)
        return image

class SoccerNetGSR(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            load_annotation: bool = True,
            save_mot_gt: bool = True,
            overwrite_mot_gt: bool = False,
    ):
        super(SoccerNetGSR, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        
        assert split in ['train', 'valid', 'test']

        # Prepare the data:
        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        if save_mot_gt:
            self._save_mot_gt(overwrite_mot_gt)
        return

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
    
    def _save_mot_gt(self, overwrite_mot_gt):
        sequence_names = self._get_sequence_names()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "Labels-GameState.json")
            gt = json.load(open(gt_file_path))
            annos = gt['annotations']
            
            mot_gt_dir = os.path.join(sequence_dir, "gt")
            os.makedirs(mot_gt_dir, exist_ok=True)
            mot_gt_file_path = os.path.join(mot_gt_dir, "gt.txt")
            if overwrite_mot_gt or not os.path.exists(mot_gt_file_path):
                with open(mot_gt_file_path, "w") as mot_gt_file:
                    for anno in annos:
                        if anno['supercategory'] != 'object' or anno['attributes']['role'] == 'ball':
                            continue
                        t = int(anno['image_id'][-6:])
                        obj_id = anno['track_id']
                        x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                        mot_gt_file.write(f"{t},{obj_id},{x},{y},{w},{h},1,1,1\n")

            ini_file_path = os.path.join(sequence_dir, "seqinfo.ini")
            if overwrite_mot_gt or not os.path.exists(ini_file_path):
                with open(ini_file_path, "w") as ini_file:
                    ini_file.write(f"[Sequence]\nname={sequence_name}\n")
                    ini_file.write(f"imDir=img1\n")
                    ini_file.write(f"frameRate=25\n")
                    ini_file.write(f"seqLength={self.sequence_infos[sequence_name]['length']}\n")
                    ini_file.write(f"imWidth={self.sequence_infos[sequence_name]['width']}\n")
                    ini_file.write(f"imHeight={self.sequence_infos[sequence_name]['height']}\n")
                    ini_file.write(f"imExt=.jpg\n")
                

        seqmap_file_path = os.path.join(self.data_dir, "SoccerNetGS", f"{self.split}_seqmap.txt")
        if not overwrite_mot_gt and os.path.exists(seqmap_file_path):
            return
        with open(seqmap_file_path, "w") as seqmap_file:
            seqmap_file.write('name\n')
            for sequence_name in sequence_names:
                seqmap_file.write(f"{sequence_name}\n")
            
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
                ann_index = int(anno['image_id'][-6:]) - 1
                obj_id = anno['track_id']
                x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                bbox = [x, y, w, h]
                category, visibility = 0, 1.0
                annotations[sequence_name][ann_index] = append_annotation(
                    annotation=annotations[sequence_name][ann_index],
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

dataset_classes = {
    # "DanceTrack": DanceTrack,
    # "SportsMOT": SportsMOT,
    # "CrowdHuman": CrowdHuman,
    # "BFT": BFT,
    "SoccerNetGSR": SoccerNetGSR,
}

class JointTrackingDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            datasets: list,
            splits: list,
            transforms=None,
            **kwargs,
    ):
        """
        Args:
            data_root: The root directory of datasets.
            datasets: The list of dataset names, e.g., ["DanceTrack", "SportsMOT"].
            splits: The list of (dataset) split names, e.g., ["train", "train"].
        """
        super().__init__()
        assert len(datasets) == len(splits), "The number of datasets and splits should be the same."
        self.transforms = transforms

        # Handle the parameters **kwargs:
        self.size_divisibility = kwargs.get("size_divisibility", 0)

        # Load the datasets into "sequence_infos", "image_paths", and "annotations",
        # each of which is a dictionary with the dataset name and split as the key.
        # e.g., sequence_infos["DanceTrack"]["train"]["sequence_name"] = {}.
        self.sequence_infos = defaultdict(lambda: defaultdict(dict))
        self.image_paths = defaultdict(lambda: defaultdict(dict))
        self.annotations = defaultdict(lambda: defaultdict(dict))
        for dataset, split in zip(datasets, splits):
            try:
                dataset_class = dataset_classes[dataset](
                    data_root=data_root,
                    split=split,
                    load_annotation=True,
                )
                self.sequence_infos[dataset][split] = dataset_class.get_sequence_infos()
                self.image_paths[dataset][split] = dataset_class.get_image_paths()
                self.annotations[dataset][split] = dataset_class.get_annotations()
            except KeyError:
                raise AttributeError(f"Dataset {dataset} is not supported.")
        # Decouple the 'is_legal' attribute from the annotations,
        # I believe it is more flexible to check the legality of the annotations in the sampling process.
        self.ann_is_legals = self._decouple_is_legal()

        # Init the sampling details:
        # Here, they are not ready for sampling,
        # you should call "self.set_sample_details()" to prepare them.
        self.sample_begins: list | None = None      # a tuple: (dataset, split, sequence_name, begin_index)
        return

    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for frame_id, annotation in enumerate(self.annotations[dataset][split][sequence_name]):
                        decoupled_is_legal[dataset][split][sequence_name].append(annotation["is_legal"])
        # Reformat the 'is_legal' attribute from a list to a tensor,
        # which is more convenient for the sampling process (calculation-friendly).
        decoupled_is_legal_in_tensor = defaultdict(lambda: defaultdict(lambda: defaultdict(torch.Tensor)))
        for dataset in decoupled_is_legal:
            for split in decoupled_is_legal[dataset]:
                for sequence_name in decoupled_is_legal[dataset][split]:
                    decoupled_is_legal_in_tensor[dataset][split][sequence_name] = torch.tensor(
                        decoupled_is_legal[dataset][split][sequence_name], dtype=torch.bool
                    )
        return decoupled_is_legal_in_tensor

    def set_sample_details(
            self,
            sample_length: int,
            sample_interval: int,
            sample_mode: str = "random_interval",
    ):
        """
        Set the details for sampling.
        Now we only have "self.sample_begins" to store the beginning of each legal sample.
        NOTE: You should call this function at the start of each epoch.
        Args:
            sample_length: The length of each sample.
            sample_interval: The interval between two adjacent samples, currently not used.
            sample_mode: The mode of sampling, e.g., "random_interval", "fixed_interval".
        """
        assert sample_mode in ["random_interval"], f"Sample mode '{sample_mode}' is not supported."
        self.sample_begins = list()
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for frame_id in range(self.sequence_infos[dataset][split][sequence_name]["length"]):
                        if self.sequence_infos[dataset][split][sequence_name]["is_static"] is True:     # static image
                            self.sample_begins.append((dataset, split, sequence_name, frame_id))
                        else:   # real-world video
                            if frame_id + sample_length <= self.sequence_infos[dataset][split][sequence_name]["length"]:
                                if self.ann_is_legals[dataset][split][sequence_name][frame_id: frame_id + sample_length].all():
                                    # TODO: We may support different sampling ratio for each dataset, need to add code.
                                    self.sample_begins.append((dataset, split, sequence_name, frame_id))
        return

    def __len__(self):
        assert self.sample_begins is not None, "Please use 'self.set_sample_details()' at the start of each epoch."
        return len(self.sample_begins)

    def __getitem__(self, info):
        dataset = info["dataset"]
        split = info["split"]
        sequence = info["sequence"]
        frame_idxs = info["frame_idxs"]
        # Get image paths:
        image_paths = [
            self.image_paths[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]
        # Read images:
        # images = [
        #     read_image(image_path) for image_path in image_paths
        # ]   # a list of tensors, shape=(C, H, W), dtype=torch.uint8
        images = [
            Image.open(image_path) for image_path in image_paths
        ]  # a list of tensors, shape=(C, H, W), dtype=torch.uint8
        # images = torch.stack(images, dim=0)     # shape=(N, C, H, W), dtype=torch.uint8
        # Get annotations:
        annotations = [
            self.annotations[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]   # "bbox", "category", "id", "visibility", "is_legal"
        # Get metas:
        metas = [
            {
                "dataset": dataset,
                "split": split,
                "sequence": sequence,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[dataset][split][sequence]["is_static"],
                "is_begin": False,      # whether the frame is the beginning of a video clip
                "size_divisibility": self.size_divisibility,
            } for frame_idx in frame_idxs
        ]
        # Do some modifications:
        metas[0]["is_begin"] = True     # the first frame is the beginning of a video clip
        # Deep copy:
        annotations = [copy.deepcopy(annotation) for annotation in annotations]
        metas = [copy.deepcopy(meta) for meta in metas]

        # Apply transforms:
        if self.transforms is not None:
            images, annotations, metas = self.transforms(images, annotations, metas)
        # from .tools import visualize_a_batch
        # visualize_a_batch(images, annotations)
        return images, annotations, metas

    def statistics(self):
        """
        Return the statistics of the dataset, in a list.
        Each item is a string: "Dancetrack.train, 35 sequences, 40000 frames."
        """
        statistics = list()
        for dataset in self.sequence_infos:
            for split in self.sequence_infos[dataset]:
                num_sequences = len(self.sequence_infos[dataset][split])
                num_frames = sum([info["length"] for info in self.sequence_infos[dataset][split].values()])
                statistics.append(f"{dataset}.{split}, {num_sequences} sequences, {num_frames} frames.")
        return statistics

def build_tracking_dataset(config: dict):
    dataset_names = config["DATASET_NAME"]
    datasets = [dataset_name for dataset_name in dataset_names if dataset_name in dataset_classes]
    assert len(datasets) > 0, "No valid dataset names provided for tracking task."

    # for dataset_name in dataset_names:
    #     if dataset_name == "SoccerNetGSR":
    #         return SoccerNetGSR(
    #         data_root=config["DATA_ROOT"],
    #         sub_dir=config["SoccerNetGSR_SUB_DIR"],
    #         split=config["SoccerNetGSR_DATA_SPLITS"] if "SoccerNetGSR_DATA_SPLITS" in config else config["DATASET_SPLITS"],
    #     )

    # transform_config
    transform_config = config

    # TODO: add polish code in this py file
    dataset = JointTrackingDataset(
        data_root=config["DATA_ROOT"],
        datasets=datasets,
        splits=[config["DATASET_SPLITS"]] * len(datasets),
        transforms=build_transforms(transform_config),
    )
    return dataset

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

# Copyright (c) Ruopeng Gao. All Rights Reserved.




class MultiIdentity:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        return images, annotations, metas


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, annotations, metas):
        for transform in self.transforms:
            images, annotations, metas = transform(images, annotations, metas)
        return images, annotations, metas


class MultiSimulate:
    """
    Simulate a video clip from a sequence of images.
    """
    def __init__(self, max_shift_ratio: float, overflow_bbox: bool):
        self.max_shift_ratio = max_shift_ratio
        self.overflow_bbox = overflow_bbox
        return

    def __call__(self, images, annotations, metas):
        if metas[0]["is_static"] is False:
            return images, annotations, metas
        else:
            # Currently, we simulate a video clip by shifting the images.
            # However, as discussed in MOTIP Appendix C.2, we need more advanced methods to simulate a video clip.

            # Calculate the shift meta infos:
            w, h = images[0].size
            max_x_shift = math.ceil(self.max_shift_ratio * w)
            max_y_shift = math.ceil(self.max_shift_ratio * h)
            x_shift = random.randint(-max_x_shift, max_x_shift)
            y_shift = random.randint(-max_y_shift, max_y_shift)
            # Prepare for the shifted sequence:
            shifted_images, shifted_annotations = [], []
            shifted_images.append(copy.deepcopy(images[0]))
            shifted_annotations.append(copy.deepcopy(annotations[0]))
            # Shifting for the rest images:
            for _idx in range(1, len(images)):
                x_min, x_max = max(0, x_shift), min(w, w + x_shift)
                y_min, y_max = max(0, y_shift), min(h, h + y_shift)
                _image = copy.deepcopy(shifted_images[_idx - 1])
                _ann = copy.deepcopy(shifted_annotations[_idx - 1])
                # Crop:
                _i, _j, _h, _w = y_min, x_min, y_max - y_min, x_max - x_min
                _ann["bbox"] = _ann["bbox"] - torch.tensor([_j, _i, _j, _i])
                _bbox = _ann["bbox"].clone()
                _max_wh = torch.tensor([_w, _h])
                _bbox = torch.min(_bbox.reshape(-1, 2, 2), _max_wh)
                _bbox = _bbox.clamp(min=0)
                _legal_idxs = torch.all(_bbox[:, 1, :] > _bbox[:, 0, :], dim=1)
                # Reshape to the original format:
                _bbox = _bbox.reshape(-1, 4)
                _need_to_select_fields = ["bbox", "category", "id", "visibility"]
                if self.overflow_bbox is False:
                    _ann["bbox"] = _bbox
                for _field in _need_to_select_fields:
                    _ann[_field] = _ann[_field][_legal_idxs]
                _ann["is_legal"] = is_legal(_ann)
                _image = v2.functional.crop(_image, _i, _j, _h, _w)
                # Resize:
                _h_ratio = h / _h
                _w_ratio = w / _w
                _bbox_ratio = torch.tensor([_w_ratio, _h_ratio] * 2)
                _ann["bbox"] = _ann["bbox"] * _bbox_ratio
                _image = v2.functional.resize(_image, [h, w])
                # Put into the shifted sequence:
                shifted_images.append(_image)
                shifted_annotations.append(_ann)
            # Check if the shifted sequence is legal:
            _is_legals = torch.tensor([_ann["is_legal"] for _ann in shifted_annotations])
            if not _is_legals.all().item():
                return images, annotations, metas
            else:
                if random.random() < 0.5:
                    return shifted_images, shifted_annotations, metas
                else:
                    # Inverse the sequence:
                    shifted_images = shifted_images[::-1]
                    shifted_annotations = shifted_annotations[::-1]
                    metas = metas[::-1]
                    # We need to fix the "is_begin" field:
                    _meta_begins = [meta["is_begin"] for meta in metas]
                    _meta_begins = _meta_begins[::-1]
                    for _ in range(len(metas)):
                        metas[_]["is_begin"] = _meta_begins[_]
                    return shifted_images, shifted_annotations, metas


class MultiStack:
    """
    Stack a sequence of images into a single tensor, (T, C, H, W).
    The result tensor is more suitable for multi-image processing.
    """
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                images = torch.stack(images, dim=0)
        return images, annotations, metas


class MultiBoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xywh_to_xyxy(annotations[_]["bbox"])
        return images, annotations, metas


class MultiBoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        for _ in range(len(annotations)):
            annotations[_]["bbox"] = box_xyxy_to_cxcywh(annotations[_]["bbox"])
        return images, annotations, metas


class MultiRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, annotations, metas):
        # Here, the boxes in annotations are in the format of (x1, y1, x2, y2).
        if torch.rand(1).item() < self.p:
            if isinstance(images, torch.Tensor):
                images = v2.functional.horizontal_flip_image(images)
            elif isinstance(images, list):
                assert isinstance(images[0], Image.Image)
                images = [v2.functional.hflip(_) for _ in images]
            else:
                raise NotImplementedError(f"The input image type {type(images)} is not supported.")
            h, w = get_image_hw(images)
            for annotation in annotations:
                annotation["bbox"] = (
                    annotation["bbox"][:, [2, 1, 0, 3]]
                    * torch.as_tensor([-1, 1, -1, 1])
                    + torch.as_tensor([w, 0, w, 0])
                )
        return images, annotations, metas


class MultiRandomSelect:
    def __init__(self, transform1, transform2, p: float = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, images, annotations, metas):
        if torch.rand(1).item() < self.p:
            return self.transform1(images, annotations, metas)
        else:
            return self.transform2(images, annotations, metas)


class MultiRandomResize:
    def __init__(self, sizes: list, max_size: int | None = None, keep_aspect_ratio: bool = True):
        self.sizes = sizes
        self.max_size = max_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, images, annotations, metas):
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

        new_hw = get_new_hw(get_image_hw(images), _new_size=new_size)    # new yx
        scale_ratio_x = new_hw[1] / get_image_hw(images)[1]
        scale_ratio_y = new_hw[0] / get_image_hw(images)[0]
        # Resize images:
        if isinstance(images, torch.Tensor):
            images = v2.functional.resize(images, new_hw)
        elif isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            images = [v2.functional.resize(_, new_hw) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        # Resize annotations:
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] * torch.as_tensor([scale_ratio_x, scale_ratio_y] * 2)
        return images, annotations, metas


class MultiRandomCrop:
    def __init__(self, min_size: int, max_size: int, overflow_bbox: bool):
        self.min_size = min_size
        self.max_size = max_size
        self.overflow_bbox = overflow_bbox

    def __call__(self, images, annotations, metas):
        # Calculate the crop box:
        curr_h, curr_w = get_image_hw(images)
        crop_h = random.randint(self.min_size, min(self.max_size, curr_h))
        crop_w = random.randint(self.min_size, min(self.max_size, curr_w))
        crop_ijhw = T.RandomCrop.get_params(images[0], (crop_h, crop_w))

        # Crop the cropped annotations:
        _annotations = copy.deepcopy(annotations)
        _i, _j, _h, _w = crop_ijhw
        for _annotation in _annotations:
            _annotation["bbox"] = _annotation["bbox"] - torch.tensor([_j, _i, _j, _i])  # (x1,y1,x2,y2) - (j,i,j,i)
            _bbox = _annotation["bbox"].clone()
            _max_wh = torch.tensor([_w, _h])
            # If the crop box is out of the image, we need to adjust the bbox:
            _bbox = torch.min(_bbox.reshape(-1, 2, 2), _max_wh)
            _bbox = _bbox.clamp(min=0)
            # We need to find the legal bbox:
            # _legal_idxs = torch.all(
            #     torch.tensor(_bbox[:, 1, :] > _bbox[:, 0, :]), dim=1
            # )
            _legal_idxs = torch.all(
                _bbox[:, 1, :] > _bbox[:, 0, :], dim=1
            )
            # Reshape to the original format:
            _bbox = _bbox.reshape(-1, 4)
            _need_to_select_fields = ["bbox", "category", "id", "visibility"]
            if self.overflow_bbox is False:
                _annotation["bbox"] = _bbox
            for _field in _need_to_select_fields:
                _annotation[_field] = _annotation[_field][_legal_idxs]
            _annotation["is_legal"] = is_legal(_annotation)

        # Check all annotations' legality:
        _is_legals = torch.tensor([_annotation["is_legal"] for _annotation in _annotations])
        # If all annotations are illegal, we need to return the original images and annotations:
        if not _is_legals.all().item():
            return images, annotations, metas
        else:
            if isinstance(images, torch.Tensor):
                images = v2.functional.crop(images, _i, _j, _h, _w)
            elif isinstance(images, list):
                assert isinstance(images[0], Image.Image)
                images = [v2.functional.crop(_, _i, _j, _h, _w) for _ in images]
            else:
                raise NotImplementedError(f"The input image type {type(images)} is not supported.")
            annotations = _annotations
            return images, annotations, metas


class MultiColorJitter:
    def __init__(
            self,
            brightness: float = 0.0,
            contrast: float = 0.0,
            saturation: float = 0.0,
            hue: float = 0.0
    ):
        self.color_jitter = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, images, annotations, metas):
        if isinstance(images, torch.Tensor):
            images = self.color_jitter(images)
        elif isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            params = self.color_jitter._get_params([images[0]])
            images = [self.color_jitter._transform(_, params=params) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        return images, annotations, metas


class MultiRandomPhotometricDistort:
    def __init__(self):
        self.ramdom_photometric_distort = v2.RandomPhotometricDistort()

    def __call__(self, images, annotations, metas):
        _params = self.ramdom_photometric_distort._get_params([images[0]])
        images = [self.ramdom_photometric_distort._transform(_, _params) for _ in images]
        return images, annotations, metas


class MultiToTensor:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, list):
            assert isinstance(images[0], Image.Image)
            images = [v2.functional.to_image(_) for _ in images]
        return images, annotations, metas


class MultiToDtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        return

    def __call__(self, images, annotations, metas):
        if isinstance(images, torch.Tensor):
            images = v2.functional.to_dtype(images, dtype=torch.float32, scale=True)
        elif isinstance(images, list):
            assert isinstance(images[0], torch.Tensor)
            images = [v2.functional.to_dtype(_, dtype=torch.float32, scale=True) for _ in images]
        else:
            raise NotImplementedError(f"The input image type {type(images)} is not supported.")
        return images, annotations, metas


class MultiNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, annotations, metas):
        # images = images.to(torch.float32).div(255)
        # images = v2.functional.normalize(images, mean=self.mean, std=self.std)
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return images.contiguous(), annotations, metas


class MultiNormalizeBoundingBoxes:
    def __init__(self):
        return

    def __call__(self, images, annotations, metas):
        # Only normalize the bounding boxes,
        # the images will be normalized in the training loop (on cuda).
        h, w = images.shape[-2:]
        for annotation in annotations:
            annotation["bbox"] = annotation["bbox"] / torch.tensor([w, h, w, h])
        return images.contiguous(), annotations, metas


# For MOTIP only, biding the ID label:

class GenerateIDLabels:
    def __init__(self, num_id_vocabulary: int, aug_num_groups: int, num_training_ids: int):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_num_groups = aug_num_groups
        self.num_training_ids = num_training_ids

    def __call__(self, images, annotations, metas):
        _T = len(images)
        _G = self.aug_num_groups
        # Collect all IDs:
        ids_set = set()
        for annotation in annotations:
            ids_set.update(set(annotation["id"].tolist()))
        _N = len(ids_set)

        # ID anns consist of the following parts:
        # (1): a (_G, _T, _N) tensor, representing the ID labels of each object in each frame.
        # (2): a (_G, _T, _N) tensor, representing the corresponding index of each object in detection annotations.
        # (3): a (_G, _T, _N) tensor, representing the mask of ID labels in each frame.
        # (4): a (_G, _T, _N) tensor, representing the time index of each object.

        ids_list = list(ids_set)
        id_to_idx = {ids_list[_]: _ for _ in range(_N)}     # the idx in the final ID labels
        base_id_masks = torch.ones((_T, _N), dtype=torch.bool)
        base_ann_idxs = - torch.ones((_T, _N), dtype=torch.int64)
        # These "base" ID anns are used to generate the final ID anns, do not directly use them.
        for t in range(_T):
            annotation = annotations[t]
            for i in range(len(annotation["id"])):
                _id = annotation["id"][i].item()
                _ann_idx = i
                _n = id_to_idx[_id]
                # generate the corresponding ID ann:
                base_id_masks[t, _n] = False
                base_ann_idxs[t, _n] = _ann_idx

        # Generate the final ID anns
        # If the number of IDs is larger than `num_id_vocabulary`, we need to randomly select a subset of IDs.
        # Also, if the number of IDs is larger than `num_training_ids`, we need to randomly select a subset of IDs.
        if _N > self.num_id_vocabulary or _N > self.num_training_ids:
            _random_select_idxs = torch.randperm(_N)[:self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary]
            base_id_masks = base_id_masks[:, _random_select_idxs]
            base_ann_idxs = base_ann_idxs[:, _random_select_idxs]
            _N = self.num_training_ids if _N > self.num_training_ids else self.num_id_vocabulary
            pass
        # Normal processing:
        id_labels = torch.zeros((_G, _T, _N), dtype=torch.int64)
        id_masks = torch.ones((_G, _T, _N), dtype=torch.bool)
        ann_idxs = - torch.ones((_G, _T, _N), dtype=torch.int64)
        for group in range(_G):
            _random_id_labels = torch.randperm(self.num_id_vocabulary)[:_N]
            _random_id_labels = _random_id_labels[None, ...].repeat(_T, 1)
            # _random_id_labels[base_id_masks] = -1
            id_labels[group] = _random_id_labels.clone()
            id_masks[group] = base_id_masks.clone()
            ann_idxs[group] = base_ann_idxs.clone()
        # Generate the time indexes:
        times = torch.arange(_T, dtype=torch.int64)[None, :, None].repeat(_G, 1, _N)
        # Check the shapes:
        assert id_labels.shape == id_masks.shape == ann_idxs.shape == times.shape

        # Split the ID anns into each frame:
        id_labels_list = torch.split(id_labels, split_size_or_sections=1, dim=1)    # each item is in (_G, 1, _N)
        id_masks_list = torch.split(id_masks, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        ann_idxs_list = torch.split(ann_idxs, split_size_or_sections=1, dim=1)      # each item is in (_G, 1, _N)
        times_list = torch.split(times, split_size_or_sections=1, dim=1)            # each item is in (_G, 1, _N)

        # Update the annotations (put the ID anns into the annotations):
        for t in range(_T):
            annotations[t]["id_labels"] = id_labels_list[t]
            annotations[t]["id_masks"] = id_masks_list[t]
            annotations[t]["ann_idxs"] = ann_idxs_list[t]
            annotations[t]["times"] = times_list[t]
        pass
        return images, annotations, metas


class TurnIntoTrajectoryAndUnknown:
    def __init__(
            self,
            num_id_vocabulary: int,
            aug_trajectory_occlusion_prob: float,
            aug_trajectory_switch_prob: float,
    ):
        self.num_id_vocabulary = num_id_vocabulary
        self.aug_trajectory_occlusion_prob = aug_trajectory_occlusion_prob
        self.aug_trajectory_switch_prob = aug_trajectory_switch_prob
        return

    def __call__(self, images, annotations, metas):
        id_labels = torch.cat([annotation["id_labels"] for annotation in annotations], dim=1)
        id_masks = torch.cat([annotation["id_masks"] for annotation in annotations], dim=1)
        ann_idxs = torch.cat([annotation["ann_idxs"] for annotation in annotations], dim=1)
        times = torch.cat([annotation["times"] for annotation in annotations], dim=1)
        _G, _T, _N = id_labels.shape
        # Del these fields from the annotations:
        for t in range(_T):
            del annotations[t]["id_labels"]
            del annotations[t]["id_masks"]
            del annotations[t]["ann_idxs"]
            del annotations[t]["times"]

        # Copy the ID anns to "trajectory_" and "unknown_":
        trajectory_id_labels = id_labels.clone()
        trajectory_id_masks = id_masks.clone()
        trajectory_ann_idxs = ann_idxs.clone()
        trajectory_times = times.clone()
        unknown_id_labels = id_labels.clone()
        unknown_id_masks = id_masks.clone()
        unknown_ann_idxs = ann_idxs.clone()
        unknown_times = times.clone()

        if self.aug_trajectory_occlusion_prob > 0.0:
            # Make trajectory occlusion:
            # 1. Turn the shape into (_G * _N, _T):
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
            unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
            # 2. Generate the occlusion mask:
            trajectory_occlusion_masks = torch.zeros_like(trajectory_id_masks, dtype=torch.bool)
            unknown_occlusion_masks = torch.zeros_like(unknown_id_masks, dtype=torch.bool)
            for i in range(_G * _N):
                if random.random() < self.aug_trajectory_occlusion_prob:
                    begin_idx = random.randint(0, _T - 1)
                    _max_T = _T - 1 - begin_idx
                    end_idx = begin_idx + math.ceil(_max_T * random.random())
                    trajectory_occlusion_masks[i, begin_idx:end_idx] = True
                    unknown_occlusion_masks[i, begin_idx:end_idx] = True
            # Currently, we do not check the legality of the occlusion mask.
            # However, we did it in the previous version.
            # 3. Apply the occlusion mask:
            trajectory_id_masks = trajectory_id_masks | trajectory_occlusion_masks
            unknown_id_masks = unknown_id_masks | unknown_occlusion_masks
            # 4. Turn the shape back:
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
            unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        if self.aug_trajectory_switch_prob > 0.0:
            # Make trajectory switch:
            # 1. Turn the shape into (_G * _T, _N):
            trajectory_id_labels = einops.rearrange(trajectory_id_labels, "G T N -> (G T) N")
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G T) N")
            trajectory_ann_idxs = einops.rearrange(trajectory_ann_idxs, "G T N -> (G T) N")
            # 2. Switch for each frame:
            #    (switching the ID labels is the same as switching the ann_idxs and masks)
            for g_t in range(_G * _T):
                switch_p = torch.ones((_N, )) * self.aug_trajectory_switch_prob
                switch_map = torch.bernoulli(switch_p)
                switch_idxs = torch.nonzero(switch_map)[:, 0]
                if len(switch_idxs) > 1:    # make sure can be switched
                    shuffled_switch_idxs = switch_idxs[torch.randperm(len(switch_idxs))]
                    # Do switch:
                    trajectory_ann_idxs[g_t, switch_idxs] = trajectory_ann_idxs[g_t, shuffled_switch_idxs]
                    trajectory_id_masks[g_t, switch_idxs] = trajectory_id_masks[g_t, shuffled_switch_idxs]
                    pass
                pass
            # 3. Turn the shape back:
            trajectory_id_labels = einops.rearrange(trajectory_id_labels, "(G T) N -> G T N", G=_G, T=_T)
            trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G T) N -> G T N", G=_G, T=_T)
            trajectory_ann_idxs = einops.rearrange(trajectory_ann_idxs, "(G T) N -> G T N", G=_G, T=_T)
            pass

        # Check all ID labels are legal:
        assert torch.all(trajectory_id_labels >= 0)
        assert torch.all(unknown_id_labels >= 0)

        # Add "newborn" ID label to unknown ID labels for supervision:
        # 1. Turn the shape into (_G * _N, _T):
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "G T N -> (G N) T")
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "G T N -> (G N) T")
        unknown_id_labels = einops.rearrange(unknown_id_labels, "G T N -> (G N) T")
        unknown_id_masks = einops.rearrange(unknown_id_masks, "G T N -> (G N) T")
        # 2. Calculate the already_born masks:
        already_born_masks = torch.cumsum(~trajectory_id_masks, dim=1)
        already_born_masks = already_born_masks > 0
        # 3. Generate the newborn ID labels:
        newborn_id_label_masks = ~ torch.cat(
            [
                torch.zeros((_G * _N, 1), dtype=torch.bool),
                already_born_masks[:, :-1]
            ],
            dim=-1
        )
        unknown_id_labels[newborn_id_label_masks] = self.num_id_vocabulary
        # 4. Turn the shape back:
        trajectory_id_labels = einops.rearrange(trajectory_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        trajectory_id_masks = einops.rearrange(trajectory_id_masks, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_labels = einops.rearrange(unknown_id_labels, "(G N) T -> G T N", G=_G, N=_N)
        unknown_id_masks = einops.rearrange(unknown_id_masks, "(G N) T -> G T N", G=_G, N=_N)

        # Update the annotations:
        for t in range(_T):
            annotations[t]["trajectory_id_labels"] = trajectory_id_labels[:, t:t+1, :]
            annotations[t]["trajectory_id_masks"] = trajectory_id_masks[:, t:t+1, :]
            annotations[t]["trajectory_ann_idxs"] = trajectory_ann_idxs[:, t:t+1, :]
            annotations[t]["trajectory_times"] = trajectory_times[:, t:t+1, :]
            annotations[t]["unknown_id_labels"] = unknown_id_labels[:, t:t+1, :]
            annotations[t]["unknown_id_masks"] = unknown_id_masks[:, t:t+1, :]
            annotations[t]["unknown_ann_idxs"] = unknown_ann_idxs[:, t:t+1, :]
            annotations[t]["unknown_times"] = unknown_times[:, t:t+1, :]

        return images, annotations, metas


def build_transforms(config: dict):

    return MultiCompose([
        MultiBoxXYWHtoXYXY(),
        MultiSimulate(max_shift_ratio=config["AUG_MAX_SHIFT_RATIO"], overflow_bbox=config["AUG_OVERFLOW_BBOX"]),
        MultiStack(),
        MultiRandomHorizontalFlip(p=0.5),
        MultiRandomSelect(
            MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
            MultiCompose([
                MultiRandomResize(sizes=config["AUG_RANDOM_RESIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
                MultiRandomCrop(
                    min_size=config["AUG_RANDOM_CROP_MIN"],
                    max_size=config["AUG_RANDOM_CROP_MAX"],
                    overflow_bbox=config["AUG_OVERFLOW_BBOX"]
                ),
                MultiRandomResize(sizes=config["AUG_RESIZE_SCALES"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"])
            ])
        ),
        MultiBoxXYXYtoCXCYWH(),
        MultiColorJitter(
            brightness=config["AUG_BRIGHTNESS"],
            contrast=config["AUG_CONTRAST"],
            saturation=config["AUG_SATURATION"],
            hue=config["AUG_HUE"],
        ) if not config["AUG_COLOR_JITTER_V2"] else MultiRandomPhotometricDistort(),
        MultiToTensor(),
        MultiStack(),
        # MultiToDtype(torch.float32),
        MultiNormalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
        MultiNormalizeBoundingBoxes(),
        # For MOTIP, biding ID labels:
        # GenerateIDLabels(
        #     num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        #     aug_num_groups=config["AUG_NUM_GROUPS"],
        #     num_training_ids=config.get("NUM_TRAINING_IDS", config["NUM_ID_VOCABULARY"]),
        # ),
        # TurnIntoTrajectoryAndUnknown(
        #     num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        #     aug_trajectory_occlusion_prob=config["AUG_TRAJECTORY_OCCLUSION_PROB"],
        #     aug_trajectory_switch_prob=config["AUG_TRAJECTORY_SWITCH_PROB"],
        # ),
    ])


def get_image_hw(image: torch.Tensor | list | Image.Image):
    if isinstance(image, torch.Tensor):
        return image.shape[-2], image.shape[-1]
    elif isinstance(image, list):
        return get_image_hw(image[0])
    elif isinstance(image, Image.Image):
        return image.height, image.width
    else:
        raise NotImplementedError("The input image type is not supported.")
