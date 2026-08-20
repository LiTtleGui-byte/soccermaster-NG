import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)

class QWEN2_5VL_ROLE_CROP_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["role_detection", "role_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"
        self.batch_size = batch_size
        self.device = device
        
        self.text_prompt = """
            Analyze this image and determine the role of the person in this image. 
            Respond ONLY with a single word in ['player', 'referee', 'goalkeeper', 'other']. 
            If there is no person in the image, or the person is not an athlete on the pitch, respond 'other'.
            """
        self.role_list = ['player', 'referee', 'goalkeeper', 'other']

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((28, 28, 3), dtype=np.uint8)
        
        batch = {'imgs': Unbatchable([crop])}
        
        return batch

    def extract_role(self, text):
        text = text.lower()
        
        if text in self.role_list:
            return text  # 返回匹配的小写形式
    
        # 如果没有找到匹配项
        return None

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        real_bs = len(batch['imgs'])
        role_detection = [None] * real_bs
        role_confidence = [0.0] * real_bs
        
        idxs = list(range(len(batch['imgs'])))
        imgs = [batch['imgs'][idx].cpu().numpy() for idx in idxs]
        
        messages = [[{"role": "user", "content":[{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": self.text_prompt}]}] for img in imgs]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
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
    
class QWEN2_5VL_ROLE_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["role_detection", "role_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=(cfg.image_width*cfg.image_height/cfg.downsample_factor/cfg.downsample_factor)//(28*28)*(28*28))
        self.processor.tokenizer.padding_side = "left"
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
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
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