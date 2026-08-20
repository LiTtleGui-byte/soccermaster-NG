import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)
    
class InternVL3_ROLE_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["role_detection", "role_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        self.downsample_factor = cfg.downsample_factor
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
            # device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.batch_size = batch_size
        self.device = device
        
        self.text_prompt = """
            There are an image and a crop from it.
            Analyze this image and determine the role of the person in this image. 
            Respond ONLY with a single word in ['player', 'referee', 'goalkeeper', 'other']. 
            If there is no person in the image, or the person is not an athlete on the pitch, respond 'other'.
            """
        self.role_list = ['player', 'referee', 'goalkeeper', 'other']
        
        self.NUM_BEAMS = 1
        self.TEMPERATURE = 0.0
        self.MAX_NEW_TOKENS = 128
        self.USE_CACHE = True
        self.DO_SAMPLE = True if self.TEMPERATURE > 0 else False

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((28, 28, 3), dtype=np.uint8)
            
        # Downsample the image according to the downsample_factor
        if self.downsample_factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        batch = {'images': Unbatchable([image]), 'crops': Unbatchable([crop])}
        
        return batch

    def extract_role(self, text):
        text = text.lower()
        
        if text in self.role_list:
            return text  # 返回匹配的小写形式
    
        # 如果没有找到匹配项
        return None

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        real_bs = len(batch['crops'])
        role_detection = [None] * real_bs
        role_confidence = [0.0] * real_bs
        
        idxs = list(range(len(batch['crops'])))
        images = [batch['images'][idx].cpu().numpy() for idx in idxs]
        crops = [batch['crops'][idx].cpu().numpy() for idx in idxs]
        
        messages = [[{"role": "user", "content":[{"type": "image", "image": Image.fromarray(image)}, {"type": "image", "image": Image.fromarray(crop)}, {"type": "text", "text": self.text_prompt}]}] for image, crop in zip(images, crops)]
        
        inputs = self.processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        
        generated_ids = self.model.generate(**inputs, num_beams=self.NUM_BEAMS, temperature=self.TEMPERATURE, max_new_tokens=self.MAX_NEW_TOKENS, use_cache=self.USE_CACHE, do_sample=self.DO_SAMPLE)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        for idx, output_text in zip(idxs, output_texts):
            role = self.extract_role(output_text)
            role_detection[idx] = role
            role_confidence[idx] = 1.0 if role is not None else 0.0

        detections['role_detection'] = role_detection
        detections['role_confidence'] = role_confidence

        return detections