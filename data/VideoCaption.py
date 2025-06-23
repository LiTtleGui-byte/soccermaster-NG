import torch
import sys
import json
import os
import random
from einops import rearrange
from torch.utils.data import Dataset
from decord import VideoReader
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from typing import List

from data.utils import Compose, ToTensor, RandomResize, Normalize

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
            keywords = ['corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed', 'red card', 'second yellow card', 'substitution', 'start of game(half)', 'end of game(half)', 'yellow card', 'throw in', 'free kick', 'saved by goal-keeper', 'shot off target', 'clearance', "lead to corner", 'off-side', 'var', 'foul with no card', 'statistics and summary', 'ball possession', 'ball out of play'],
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

        self.data = []

        clip_root = os.path.join(data_root, "video_clip")
        clip_json_root = os.path.join(data_root, "video_clip_json")
        for dataset in video_caption_datasets:
            clip_base_dir = os.path.join(clip_root, f"{dataset}-high-resolution") if '1988' in dataset else os.path.join(clip_root, dataset)
            clip_json_path = os.path.join(clip_json_root, dataset, f"classification_{split}.json")
            with open(clip_json_path, 'r') as file:
                current_data = json.load(file)
                for item in current_data:
                    item["video"] = os.path.join(clip_base_dir, item["video"])
                self.data.extend(current_data)

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
        
        metas = {"task": 'MatchVision',
            # "video": video_path,
            # "caption": video_info['caption'],
            # "text": video_info[self.text_key],
            # "size_divisibility": 1,
            }
        
        annotation = {'caption': video_info['caption'], 'text': video_info[self.text_key]}
        
        processed_frames = []
        frames = frames.asnumpy().astype(np.uint8)
        bs = frames.shape[0]
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            transformed_frame, _, _ = self.transforms(frame_pil, annotation, metas)
            processed_frames.append(transformed_frame.unsqueeze(0))
        frames = torch.cat(processed_frames, dim=0)
        
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
        # Initialize the tensor with a default value of -1 (indicating no match)
        caption_index = -1
        for i, keyword in enumerate(self.keywords):
            if keyword == caption:
                caption_index = i
                break
        
        # Convert the index to a tensor
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
        max_num_frames=-1, trimmed30=False, processor=None
    ):
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

    return frames, frame_indices, duration

def build_transforms(config: dict):
    """
    Build transforms
    """
    return Compose([
        ToTensor(),
        RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
    ])
    
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
        num_frames=config["VIDEO_CAPTION_NUM_FRAMES"],
        sample=config["VIDEO_CAPTION_SAMPLE"],
        fix_start=config["VIDEO_CAPTION_FIX_START"],
        max_num_frames=config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
        trimmed30=config["VIDEO_CAPTION_TRIMMED30"],
        keywords=config["VIDEO_CAPTION_KEYWORDS"],
        text_key=config["VIDEO_CAPTION_TEXT_KEY"],
        transforms=build_transforms(config),
    )
    assert config["VIDEO_CAPTION_FIX_START"] == None
    return dataset

def build_video_caption_dataloader(config: dict, split: str):
    dataset = build_video_caption_dataset(config, split)
    shuffle = True if split == "train" else False
    return DataLoader(dataset, batch_size=config["VIDEO_CAPTION_BATCH_SIZE"], shuffle=shuffle, collate_fn=collate_fn, num_workers=config["VIDEO_CAPTION_NUM_WORKERS"])