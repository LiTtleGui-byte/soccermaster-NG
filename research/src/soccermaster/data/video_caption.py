import torch
import sys
sys.path.append('/remote-home/haolinyang/sports/Soccer-Backbone')
import json
import os
import random
from einops import rearrange
from torch.utils.data import Dataset, Sampler
from decord import VideoReader
import decord
decord.bridge.set_bridge("torch")
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from typing import List
import torch.distributed as dist
import math

from soccermaster.data.utils import Compose, ToTensor, RandomResize, Normalize, get_image_hw, ColorJitter, RandomHorizontalFlip, GaussianNoise, GaussianBlur, ClearAugmentationMetas, RandomCrop, RandomAffine, RandomPerspective

keywords_list = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession", "throw in", "show added time", "shot off target", "start of half game", "substitution", "saved by goal-keeper", "red card", "lead to corner", "ball out of play", "off side", "goal", "penalty", "yellow card", "foul lead to penalty", "corner", "free kick", "foul with no card"]

class VideoCaptionDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            video_caption_datasets: List[str],
            split: str,
            num_frames=30, 
            sample='rand', 
            fix_start=None, 
            max_num_frames=-1, 
            trimmed30=False,
            keywords = keywords_list,
            # require_text = False,
            text_key = "comments_text_anonymized",
            transforms=None,
    ):
        self.num_frames = num_frames
        self.sample = sample
        self.fix_start = fix_start
        self.max_num_frames = max_num_frames
        self.trimmed30 = trimmed30
        self.keywords = keywords
        self.transforms = transforms
        # self.require_text = require_text
        self.text_key = text_key

        # 创建关键词到索引的映射字典，提高查找效率
        self.keyword_to_index = {keyword: i for i, keyword in enumerate(self.keywords)}

        self.data = []

        clip_root = os.path.join(data_root, "video_clip")
        clip_json_root = os.path.join(data_root, "video_clip_json")
        for dataset in video_caption_datasets:
            if dataset in ['SoccerReplay-1988']:
                # clip_base_dir = os.path.join(clip_root, f"{dataset}-high-resolution")
                clip_base_dir = os.path.join(clip_root, "SoccerReplay-1988-high-resolution")
            else:
                clip_base_dir = os.path.join(clip_root, dataset)
            clip_json_path = os.path.join(clip_json_root, dataset, f"classification_{split}.json")
            with open(clip_json_path, 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    if dataset in ['SoccerReplay-1988', 'SoccerNet-v2']:
                        item["video"] = os.path.join(clip_base_dir, item["video"])
                    elif dataset == 'MatchTime':
                        item["video"] = os.path.join(clip_base_dir, split, item["video"]) if split in ['train', 'valid'] else os.path.join(clip_base_dir, 'SN-Caption-test-align', item["video"])
                    else:
                        raise ValueError(f"Invalid dataset: {dataset}")
                self.data.extend(current_data)

        # 预分组：无文本与有文本（以是否存在text_key且非None判断）
        self.no_text_indices = []
        self.has_text_indices = []
        for idx, item in enumerate(self.data):
            if (self.text_key in item) and (item[self.text_key] is not None):
                self.has_text_indices.append(idx)
            else:
                self.no_text_indices.append(idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # try:
        video_info = self.data[idx]
        video_path = video_info['video']
        # caption = self.caption_to_tensor(video_info['caption'])
        # Extract frames using the pre-defined function
        frames, frame_indices, duration = read_frames_decord(
            video_path, self.num_frames, self.sample, self.fix_start, 
            self.max_num_frames, self.trimmed30
        )
        
        metas = {"task": 'VideoCaption',
            "video": video_path,
            # "caption": video_info['caption'],
            # "text": video_info[self.text_key],
            # "size_divisibility": 1,
            }
        
        text = video_info[self.text_key] if self.text_key in video_info else None
        annotation = {'caption': video_info['caption'], 'caption_index': self.caption_to_tensor(video_info['caption']), 'text': text}
        
        frames, annotation, metas = self.transforms(frames, annotation, metas)
        # processed_frames = []
        # frames = frames.asnumpy().astype(np.uint8)
        # bs = frames.shape[0]
        # for frame in frames:
        #     frame_pil = Image.fromarray(frame)
        #     transformed_frame, _, _ = self.transforms(frame_pil, annotation, metas)
        #     processed_frames.append(transformed_frame.unsqueeze(0))
        # frames = torch.cat(processed_frames, dim=0)
        
        return frames, annotation, metas
            
        # except:
        #     idx = random.randint(0, len(self) - 1)
        #     return self.__getitem__(idx)
    
    def caption_to_tensor(self, caption):
        """
        Converts a caption string to a tensor based on the keywords list.
        The tensor will contain the index of the keyword found in the caption.
        If the caption does not match any keyword, the tensor will contain -1.
        """
        caption_index = self.keyword_to_index[caption]
        caption_tensor = torch.tensor(caption_index, dtype=torch.long)
                
        return caption_tensor

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, trimmed30=False):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    # 只使用前 30 秒
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames, frame_indices, duration

def build_transforms(config: dict, split: str = "train"):
    
    transforms = [
        ClearAugmentationMetas(),  # Clear any previous augmentation metadata
        ToTensor(),  # Convert to tensor and float, divide by 255
    ]
    
    # Add random affine after ToTensor and before RandomPerspective
    if split == "train" and config["AUG_ENABLE_RANDOM_AFFINE"]:
        transforms.append(RandomAffine(
            degrees=config["AUG_AFFINE_DEGREES"],
            translate=config["AUG_AFFINE_TRANSLATE"],
            scale=config["AUG_AFFINE_SCALE"],
            shear=config["AUG_AFFINE_SHEAR"],
            p=config["AUG_AFFINE_PROB"]
        ))
    
    # Add random perspective after RandomAffine and before RandomCrop
    if split == "train" and config["AUG_ENABLE_RANDOM_PERSPECTIVE"]:
        transforms.append(RandomPerspective(
            distortion_scale=config["AUG_PERSPECTIVE_DISTORTION_SCALE"],
            p=config["AUG_PERSPECTIVE_PROB"]
        ))
    
    # Add random crop after ToTensor and before RandomResize
    if split == "train" and config["AUG_ENABLE_RANDOM_CROP"]:
        transforms.append(RandomCrop(
            crop_size_ratio_range=config["AUG_RANDOM_CROP_SIZE_RATIO_RANGE"],
            p=config["AUG_RANDOM_CROP_PROB"]
        ))
    
    transforms.append(RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]))
    
    # Add training-specific augmentations after resize
    if split == "train" and config["AUG_ENABLE_TRAINING_AUGMENTATION"]:
        # Color jitter
        if config["AUG_COLOR_JITTER_V2"]:
            transforms.append(ColorJitter(
                brightness=config["AUG_BRIGHTNESS"],
                contrast=config["AUG_CONTRAST"], 
                saturation=config["AUG_SATURATION"],
                hue=config["AUG_HUE"],
                p=1.0  # Always apply if enabled
            ))
        
        # Random horizontal flip
        if config["AUG_RANDOM_HORIZONTAL_FLIP"]:
            transforms.append(RandomHorizontalFlip(p=config["AUG_HORIZONTAL_FLIP_PROB"]))
        
        # Gaussian noise
        if config["AUG_GAUSSIAN_NOISE"]:
            transforms.append(GaussianNoise(
                mean=0.0,
                std=config["AUG_GAUSSIAN_NOISE_STD"],
                p=config["AUG_GAUSSIAN_NOISE_PROB"]
            ))
        
        # Gaussian blur
        if config["AUG_GAUSSIAN_BLUR"]:
            transforms.append(GaussianBlur(
                kernel_size_range=config["AUG_GAUSSIAN_BLUR_KERNEL_SIZE_RANGE"],
                sigma_range=config["AUG_GAUSSIAN_BLUR_SIGMA_RANGE"],
                p=config["AUG_GAUSSIAN_BLUR_PROB"]
            ))
    
    # Add final transforms
    transforms.extend([
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),  # Normalize at the end
    ])
    
    return Compose(transforms)
    
def collate_fn(batch):
    clip, annotations, metas = zip(*batch)
    _B = len(batch)
    clips = torch.stack(clip)
    
    return {
        "images": clips,
        "annotations": annotations,
        # "annotations": new_annotations,
        "metas": metas,
    }
    
def build_video_caption_dataset(config: dict, split: str):
    dataset = VideoCaptionDataset(
        data_root=config["VIDEO_CAPTION_DATA_ROOT"],
        video_caption_datasets=config["VIDEO_CAPTION_DATASETS"],
        split=split,
        num_frames=config["NUM_FRAMES"],
        sample=config["VIDEO_CAPTION_SAMPLE"],
        fix_start=config["VIDEO_CAPTION_FIX_START"],
        max_num_frames=config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
        trimmed30=config["VIDEO_CAPTION_TRIMMED30"],
        # keywords=config["VIDEO_CAPTION_KEYWORDS"],
        text_key=config["VIDEO_CAPTION_TEXT_KEY"],
        transforms=build_transforms(config, split),
    )
    assert config["VIDEO_CAPTION_FIX_START"] == None
    return dataset

def build_video_caption_dataloader(config: dict, split: str):
    dataset = build_video_caption_dataset(config, split)
    prefetch_factor = config["PREFETCH_FACTOR"] if config["VIDEO_CAPTION_NUM_WORKERS"] > 0 else None
    persistent_workers = config["VIDEO_CAPTION_NUM_WORKERS"] > 0

    if split == "test":
        # 评测：无文本样本在前，有文本样本在后，两组内部独立shuffle
        # 使用分布式友好的分组采样器（在__iter__时读取分布式状态，单机也可用）
        sampler = DistributedGroupedShuffleSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=config["VIDEO_CAPTION_TEST_BATCH_SIZE"],
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=config["VIDEO_CAPTION_NUM_WORKERS"],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    else:
        # 训练：保持原先shuffle
        return DataLoader(
            dataset,
            batch_size=config["VIDEO_CAPTION_BATCH_SIZE"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config["VIDEO_CAPTION_NUM_WORKERS"],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

class GroupedShuffleSampler(Sampler[int]):
    """
    将索引分两组：无text在前、有text在后；两组内部分别shuffle。
    """
    def __init__(self, dataset: VideoCaptionDataset):
        self.dataset = dataset

    def __iter__(self):
        device = torch.device("cpu")
        no_text = torch.tensor(self.dataset.no_text_indices, dtype=torch.long, device=device)
        has_text = torch.tensor(self.dataset.has_text_indices, dtype=torch.long, device=device)

        if no_text.numel() > 0:
            perm_nt = torch.randperm(no_text.numel(), device=device)
            no_text = no_text[perm_nt]
        if has_text.numel() > 0:
            perm_ht = torch.randperm(has_text.numel(), device=device)
            has_text = has_text[perm_ht]

        ordered = torch.cat([no_text, has_text], dim=0).tolist()
        return iter(ordered)

    def __len__(self):
        return len(self.dataset)

class DistributedGroupedShuffleSampler(Sampler[int]):
    """
    分布式版本：先按“无text组 + 有text组”的顺序构建全局打乱序列，再按rank划分。
    """
    def __init__(self, dataset: VideoCaptionDataset, drop_last: bool = False):
        self.dataset = dataset
        self.drop_last = drop_last
        self.epoch = 0
        # 延迟到迭代时再读取分布式状态
        self._refresh_dist_params()

    def _refresh_dist_params(self):
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 每次迭代前刷新分布式参数，适配加速库在prepare之后初始化进程组的场景
        self._refresh_dist_params()
        device = torch.device("cpu")
        no_text = torch.tensor(self.dataset.no_text_indices, dtype=torch.long, device=device)
        has_text = torch.tensor(self.dataset.has_text_indices, dtype=torch.long, device=device)

        if no_text.numel() > 0:
            g_nt = torch.Generator(device=device)
            g_nt.manual_seed(self.epoch)
            perm_nt = torch.randperm(no_text.numel(), generator=g_nt, device=device)
            no_text = no_text[perm_nt]
        if has_text.numel() > 0:
            g_ht = torch.Generator(device=device)
            g_ht.manual_seed(self.epoch + 1024)
            perm_ht = torch.randperm(has_text.numel(), generator=g_ht, device=device)
            has_text = has_text[perm_ht]

        indices = torch.cat([no_text, has_text], dim=0).tolist()

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        # 确保长度与当前分布式配置一致
        self._refresh_dist_params()
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

if __name__ == "__main__":
    
    def set_transform(
            model_name = "./pretrained_models/google/siglip2-base-patch16-224"
        ):
        from transformers import AutoProcessor
        # 创建 SiglipProcessor 实例
        processor = AutoProcessor.from_pretrained(model_name)
        return processor
    
    import pdb
    import numpy as np
    from PIL import Image
    video_path = '/remote-home/haolinyang/sports/UniSoccer/train_data/video_clip/MatchTime/SN-Caption-test-align/england_epl_2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea/1_11_37.mp4'
    num_frames = 30
    sample = 'middle'
    fix_start = None
    max_num_frames = -1
    trimmed30 = False
    
    frames, frame_indices, duration = read_frames_decord(
            video_path, num_frames, sample, fix_start, 
            max_num_frames, trimmed30
        )
    
    transforms = Compose([
        ToTensor(),
        RandomResize(sizes=[224,], max_size=224, keep_aspect_ratio=False),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transformed_frame1, _, _ = transforms(frames, {}, {})
    
    model_name = '/remote-home/haolinyang/sports/UniSoccer/pretrained_models/google/siglip-base-patch16-224'
    transform2 = set_transform(model_name)
    transformed_frames2 = torch.cat([transform2(images=frame, return_tensors="pt")["pixel_values"] for frame in frames], dim=0)
    
    processed_frames = []
    frames = frames.numpy().astype(np.uint8)
    bs = frames.shape[0]
    for frame in frames:
        frame = frame.transpose(1, 2, 0)
        frame_pil = Image.fromarray(frame)
        transformed_frame, _, _ = transforms(frame_pil, {}, {})
        processed_frames.append(transformed_frame.unsqueeze(0))
    transformed_frames3 = torch.cat(processed_frames, dim=0)
    
    pdb.set_trace()