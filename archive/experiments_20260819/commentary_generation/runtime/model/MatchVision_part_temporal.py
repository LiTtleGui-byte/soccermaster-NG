from PIL import Image
import requests
from transformers import AutoProcessor, SiglipVisionModel
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionConfig

from ..paths import SIGLIP2_ROOT

class ResidualAttentionBlock(nn.Module):
    def __init__(self, res_idx, d_model, n_head, drop_path=0., attn_mask=None, dropout=0., attention_type='divided_space_time', model_name=str(SIGLIP2_ROOT), use_temporal=True):
        super().__init__()
        model = SiglipVisionModel.from_pretrained(model_name)
        vision_model = model.vision_model

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_temporal = use_temporal
        # print(f'Droppath: {drop_path}')

        # Temporal Attention Parameters
        if attention_type == 'divided_space_time' and use_temporal:
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.temporal_fc = nn.Linear(d_model, d_model)
            self.register_parameter('temporal_alpha_attn', nn.Parameter(torch.tensor(0.)))

        self.encoder = vision_model.encoder.layers[res_idx]
        self.attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x)[0]
    
    def temporal_attention(self, x):
        return self.temporal_attn(x, x, x)[0]

    def forward(self, x, B, T):
        # divided_space_time 
        
        if self.use_temporal:
            ## Temporal 
            xt = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            res_temporal = self.drop_path(self.temporal_attention(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x + self.temporal_alpha_attn.tanh() * res_temporal # 180 196 768

            ## Spatial
            xs = xt # always 180 196 768
            res_spatial = self.encoder(xs, self.attn_mask)[0]
        else:
            ## Spatial only (no temporal attention)
            res_spatial = self.encoder(x, self.attn_mask)[0]
        
        return res_spatial
    

class Timesformer(nn.Module):
    def __init__(self, width, layers, heads, model_name, drop_path=0., checkpoint_num=0, dropout=0., temporal_start_layer=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        self.temporal_start_layer = temporal_start_layer
        for idx in range(layers):
            # Only enable temporal attention for layers >= temporal_start_layer
            use_temporal = (idx >= temporal_start_layer)
            self.resblocks.append(ResidualAttentionBlock(res_idx=idx, d_model=width, n_head=heads, drop_path=dpr[idx], dropout=dropout, model_name=model_name, use_temporal=use_temporal))
        self.checkpoint_num = checkpoint_num
        # self.checkpoint_num = 24
            
    def forward(self, x, B, T):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, B, T)
        return x


class VisionTimesformer(nn.Module):
    def __init__(
        self, output_dim=768, num_frames=30, 
        input_resolution = 224, patch_size = 16, width = 768, heads=12,
        encoder_type = "spatial_and_temporal",
        model_name = str(SIGLIP2_ROOT),
        temporal_start_layer = 16,
    ):
        super().__init__()

        self.num_frames = num_frames
        model = SiglipVisionModel.from_pretrained(model_name)
        model_config = SiglipVisionConfig.from_pretrained(model_name)
        
        self.encoder_type = encoder_type
        self.patch_size = patch_size
        self.width = width
        self.temporal_start_layer = temporal_start_layer

        if self.encoder_type == "spatial_only":
            self.vision_model = model

        elif self.encoder_type == "spatial_and_temporal":
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

            vision_model = model.vision_model
            self.vision_model_embedding = vision_model.embeddings
            self.timesformer = Timesformer(width=width, layers=model_config.num_hidden_layers, heads=model_config.num_attention_heads, model_name=model_name, temporal_start_layer=temporal_start_layer)
            self.post_layernorm = vision_model.post_layernorm
            self.head = vision_model.head


    def get_num_layers(self):
        return len(self.timesformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):   
        return {'temporal_positional_embedding'}

    def forward(self, x, return_local_features=False):
        B, _, T, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.vision_model_embedding(x)
        
        # Process spatial-only layers (before temporal_start_layer)
        for idx in range(self.temporal_start_layer):
            x = self.timesformer.resblocks[idx](x, B, T)
            
        local_features = x
        local_features = rearrange(local_features, "(b t) n m -> b t n m", b=B, t=T)
        
        x = rearrange(x, "(b t) n m -> b n t m", b =B, t=T)
        x = x + self.temporal_positional_embedding
        x = rearrange(x, 'b n t m -> (b t) n m')
        
        for idx in range(self.temporal_start_layer, self.get_num_layers()):
            x = self.timesformer.resblocks[idx](x, B, T)
        
        x = self.post_layernorm(x)
        x = self.head(x) # 180 768
        x = rearrange(x, "(b t) m -> b t m", b=B, t=T) # 6 30 768

        if return_local_features:
            return x, local_features
        else:
            return x
    

class TextEncoder(nn.Module):
    def __init__(
        self, model_name = str(SIGLIP2_ROOT)
    ):
        super().__init__()
        self.model_name = model_name
        self.model = SiglipTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences):
        # important: make sure to set padding="max_length" as that's how the model was trained
        # inputs = self.tokenizer(sentences, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        if 'siglip2' in self.model_name:
            inputs = self.tokenizer(sentences, padding="max_length", max_length=64, return_tensors="pt", truncation=True)
        else:
            inputs = self.tokenizer(sentences, padding="max_length", return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return pooled_output, last_hidden_state
