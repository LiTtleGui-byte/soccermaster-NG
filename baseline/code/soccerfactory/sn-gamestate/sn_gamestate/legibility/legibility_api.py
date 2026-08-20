import pandas as pd
import torch
import numpy as np
import logging
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torch.backends.cudnn as cudnn

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)

class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x

class Legibility(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["legibility_score"]
    collate_fn = default_collate

    def __init__(self, cfg, batch_size, device, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.device = device
        cudnn.benchmark = True
        
        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = LegibilityClassifier34()
        legibility_model_path = cfg.legibility_model_path
        state_dict = torch.load(legibility_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

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

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        img_crops_PIL = [Image.fromarray(im_crop) for im_crop in im_crops]
        img_crops_PIL_transformed = [self.transforms(img_crop_PIL) for img_crop_PIL in img_crops_PIL]
        img_crops_PIL_transformed = torch.stack(img_crops_PIL_transformed).to(self.device)
        
        # Get legibility scores
        outputs = self.model(img_crops_PIL_transformed)
        legibility_scores = outputs[:,0].float().cpu().detach().numpy()
        
        # legibility_scores = []
        # for img in batch['img']:
        #     img = img.cpu().numpy()
        #     img_PIL = Image.fromarray(img)
            
        #     # Transform image
        #     img_transformed = self.transforms(img_PIL)
        #     img_transformed = img_transformed.unsqueeze(0).to(self.device)
            
        #     # Get legibility score
        #     output = self.model(img_transformed)
        #     score = output[0,0].float().cpu().detach().numpy()
        #     legibility_scores.append(score)

        # detections['legibility_score'] = legibility_scores
        # return detections
        
        ls_df = pd.DataFrame(
            {
                "legibility_score": list(legibility_scores),
            },
            index=detections.index,
        )
        return ls_df
