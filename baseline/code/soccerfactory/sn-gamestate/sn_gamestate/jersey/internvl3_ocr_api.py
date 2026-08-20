import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image
import math
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from torchvision import models, transforms

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from qwen_vl_utils import process_vision_info

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def preprocess_image(image, transform, input_size=448, max_num=12):
    image = Image.fromarray(image)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class InternVL3_OCR_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
        if self.save_jersey_number_full_detection:
            self.output_columns.append("jersey_number_full_detection")
        self.use_legibility_filter = cfg.use_legibility_filter
        if self.use_legibility_filter:
            self.input_columns.append("legibility_score")
            self.legibility_filter_threshold = cfg.legibility_filter_threshold
        
        device_map = split_model(self.model_path)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.batch_size = batch_size
        self.device = device
        self.transform = build_transform(input_size=448)
        self.questions = "<image>\nAnalyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.generation_config['pad_token_id'] = self.tokenizer.pad_token_id

    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        
        batch = {'imgs': Unbatchable([preprocess_image(crop, self.transform)])}
        
        if self.use_legibility_filter:
            batch['legibility_score'] = detection.legibility_score
        
        return batch

    def extract_numbers(self, text):
        if text.strip() == "?":
            return None
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        real_bs = len(batch['imgs'])
        jersey_number_detection = [None] * real_bs
        jersey_number_confidence = [0.0] * real_bs
        jersey_number_full_detection = [''] * real_bs
        
        # Create a list of valid indices based on legibility filter
        idxs = []
        if self.use_legibility_filter:
            for i, score in enumerate(batch['legibility_score']):
                if score >= self.legibility_filter_threshold:
                    idxs.append(i)
        else:
            idxs = list(range(len(batch['imgs'])))
        
        if len(idxs) > 0:
            pixel_values = [batch['imgs'][idx].to(torch.bfloat16).to(self.device) for idx in idxs]
            num_patches_list = [pixel_values[i].size(0) for i in range(len(pixel_values))]
            pixel_values = torch.cat(pixel_values, dim=0)   

            questions = [self.questions] * len(num_patches_list)
            responses = self.model.batch_chat(self.tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=self.generation_config)
            
            for idx, question, response in zip(idxs, questions, responses):
                jersey_number = self.extract_numbers(response)
                jersey_number_detection[idx] = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = response
            
        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            detections['jersey_number_full_detection'] = jersey_number_full_detection

        return detections