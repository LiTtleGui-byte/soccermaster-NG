import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

from multiprocessing import Pool

log = logging.getLogger(__name__)


class QWEN2VL_OCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.save_jersey_number_full_detection = cfg.save_jersey_number_full_detection
        if self.save_jersey_number_full_detection:
            self.output_columns.append("jersey_number_full_detection")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
        self.batch_size = batch_size
        self.device = device

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
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
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
        jersey_number_detection = []
        jersey_number_confidence = []
        jersey_number_full_detection = []
        
        for img in batch['img']:
            img = img.cpu().numpy()
            img_PIL = Image.fromarray(img)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_PIL,
                        },
                        # {"type": "text", "text": "Extract the jersey number from the back of the player and return only the numeric value. If the number is not visible or cannot be clearly identified, return only a question mark ('?')."},
                        {"type": "text", "text": "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
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
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            jersey_number = self.extract_numbers(output_text)
            jersey_number_detection.append(jersey_number)
            jersey_number_confidence.append(1.0 if jersey_number is not None else 0.0)
            if self.save_jersey_number_full_detection:
                jersey_number_full_detection.append(output_text)

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            detections['jersey_number_full_detection'] = jersey_number_full_detection

        return detections
