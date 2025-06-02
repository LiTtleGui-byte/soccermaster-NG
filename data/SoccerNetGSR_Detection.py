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
import random
from math import floor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xywh, bbox_xywh_to_cxcywh
from data.utils import Compose, ToTensor, RandomResize, Normalize, get_image_hw
from data.SoccerNetGSR_ReID import role_mapping

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
                if anno['supercategory'] != 'object' or anno['attributes']['role'] == 'ball':
                    continue
                frame_idx = int(anno['image_id'][-6:]) - 1
                obj_id = anno['track_id']
                x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                bbox = [x, y, w, h]
                category, visibility = 0, 1.0
                
                # Append to lists instead of using torch.cat
                annotations[sequence_name][frame_idx]["id"].append(obj_id)
                annotations[sequence_name][frame_idx]["category"].append(category)
                annotations[sequence_name][frame_idx]["bbox"].append(bbox)
                annotations[sequence_name][frame_idx]["visibility"].append(visibility)
                annotations[sequence_name][frame_idx]["role"].append(role_mapping[anno['attributes']['role']])
                
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
                else:
                    # Empty frame
                    frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                    frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)
                    frame_annotation["role"] = torch.zeros((0, ), dtype=torch.int64)
                if "intrinsic" not in frame_annotation:
                    frame_annotation["valid_camera"] = torch.tensor(False, dtype=torch.bool)
                    frame_annotation["intrinsic"] = torch.zeros((3, 3), dtype=torch.float32)
                    frame_annotation["translation"] = torch.zeros((3, ), dtype=torch.float32)
                    frame_annotation["rotation_matrix"] = torch.eye(3, dtype=torch.float32)
                else:
                    frame_annotation["valid_camera"] = torch.tensor(True, dtype=torch.bool)

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
        annotation['roles'] = annotation['role']
        # use for camera loss:
        annotation['quaternion'] = mat_to_quat(annotation['rotation_matrix'].unsqueeze(0)).squeeze(0)
        H, W = metas['image_size']
        fov_h = 2 * torch.atan((H / 2) / annotation['intrinsic'][1, 1])
        fov_w = 2 * torch.atan((W / 2) / annotation['intrinsic'][0, 0])
        annotation['fov_hw'] = torch.stack([fov_h, fov_w])
        
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
