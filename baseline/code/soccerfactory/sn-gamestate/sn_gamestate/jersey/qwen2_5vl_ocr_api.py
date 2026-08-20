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

# class QWEN2_5VL_OCR(DetectionLevelModule):
#     input_columns = ["bbox_ltwh"]
#     output_columns = ["jersey_number_detection", "jersey_number_confidence"]
#     collate_fn = default_collate

#     def __init__(self, cfg, batch_size, device, tracking_dataset=None):
#         super().__init__(batch_size=batch_size)
#         self.cfg = cfg
#         self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
#         if self.save_jersey_number_full_detection:
#             self.output_columns.append("jersey_number_full_detection")
#         self.use_legibility_filter = cfg.use_legibility_filter
#         if self.use_legibility_filter:
#             self.input_columns.append("legibility_score")
#             self.legibility_filter_threshold = cfg.legibility_filter_threshold
        
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2.5-VL-7B-Instruct",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#             device_map=device,
#         )
#         self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#         self.batch_size = batch_size
#         self.device = device

#     def no_jersey_number(self):
#         return None, 0

#     @torch.no_grad()
#     def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
#         l, t, r, b = detection.bbox.ltrb(
#             image_shape=(image.shape[1], image.shape[0]), rounded=True
#         )
#         crop = image[t:b, l:r]
#         if crop.shape[0] == 0 or crop.shape[1] == 0:
#             crop = np.zeros((10, 10, 3), dtype=np.uint8)
#         crop = Unbatchable([crop])
        
#         batch = {
#             "img": crop,
#         }
        
#         if self.use_legibility_filter:
#             batch["legibility_score"] = Unbatchable([detection.legibility_score])
        
#         return batch

#     def extract_numbers(self, text):
#         if text.strip() == "?":
#             return None
#         number = ''
#         for char in text:
#             if char.isdigit():
#                 number += char
#         return number if number != '' else None

#     @torch.no_grad()
#     def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
#         jersey_number_detection = []
#         jersey_number_confidence = []
#         jersey_number_full_detection = []
        
#         for img in batch['img']:
#             img = img.cpu().numpy()
#             img_PIL = Image.fromarray(img)
#             if self.use_legibility_filter:
#                 # legibility_score = batch["legibility_score"][0].cpu().numpy()
#                 legibility_score = batch["legibility_score"][0]
#                 if legibility_score < self.legibility_filter_threshold:
#                     jersey_number_detection.append(None)
#                     jersey_number_confidence.append(0.0)
#                     if self.save_jersey_number_full_detection:
#                         jersey_number_full_detection.append('')
#                     continue
            
#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image",
#                             "image": img_PIL,
#                         },
#                         # {"type": "text", "text": "Extract the jersey number from the back of the player and return only the numeric value. If the number is not visible or cannot be clearly identified, return only a question mark ('?')."},
#                         {"type": "text", "text": "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."},
#                     ],
#                 }
#             ]

#             text = self.processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = self.processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt",
#             )
#             inputs = inputs.to(self.device)

#             generated_ids = self.model.generate(**inputs, max_new_tokens=128)
#             generated_ids_trimmed = [
#                 out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#             ]
#             output_text = self.processor.batch_decode(
#                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )[0]

#             jersey_number = self.extract_numbers(output_text)
#             jersey_number_detection.append(jersey_number)
#             jersey_number_confidence.append(1.0 if jersey_number is not None else 0.0)
#             if self.save_jersey_number_full_detection:
#                 jersey_number_full_detection.append(output_text)

#         detections['jersey_number_detection'] = jersey_number_detection
#         detections['jersey_number_confidence'] = jersey_number_confidence
#         if self.save_jersey_number_full_detection:
#             detections['jersey_number_full_detection'] = jersey_number_full_detection

#         return detections

class QWEN2_5VL_OCR_BATCH(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cfg = cfg
        self.model_path = self.cfg.model_path
        self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
        if self.save_jersey_number_full_detection:
            self.output_columns.append("jersey_number_full_detection")
        self.use_legibility_filter = cfg.use_legibility_filter
        if self.use_legibility_filter:
            self.input_columns.append("legibility_score")
            self.legibility_filter_threshold = cfg.legibility_filter_threshold
        
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
        
        self.text_prompt = "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."

        self.NUM_BEAMS = 1
        self.TEMPERATURE = 0.0
        self.MAX_NEW_TOKENS = 128
        self.USE_CACHE = True
        self.DO_SAMPLE = True if self.TEMPERATURE > 0 else False

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
        
        batch = {'imgs': Unbatchable([crop])}
        
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
            generated_ids = self.model.generate(**inputs, num_beams=self.NUM_BEAMS, temperature=self.TEMPERATURE, max_new_tokens=self.MAX_NEW_TOKENS, use_cache=self.USE_CACHE, do_sample=self.DO_SAMPLE)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            for idx, output_text in zip(idxs, output_texts):
                jersey_number = self.extract_numbers(output_text)
                jersey_number_detection[idx] = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = output_text

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            detections['jersey_number_full_detection'] = jersey_number_full_detection

        return detections