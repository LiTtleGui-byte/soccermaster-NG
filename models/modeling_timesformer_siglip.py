# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TimeSformerV2 built upon PyTorch TimeSformer model."""
import collections
import collections.abc
from dataclasses import dataclass
import types
from typing import Dict, List, Optional, Tuple, Union
import random
import math
import copy
import torch
import torch.nn.functional
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_soccer_backbone import SoccerBackboneConfig
from transformers import CLIPTextModel, CLIPTextConfig, AutoTokenizer, CLIPConfig, SiglipTextModel, SiglipConfig
_CONFIG_FOR_DOC = "TimesformerConfig"
logger = logging.get_logger(__name__)

TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/timesformer-base-finetuned-k400",
    # See all TimeSformer models at https://huggingface.co/models?filter=timesformer
]

VIDEO_TEMPLATES = [
    # "A video of an action where",
    # "A video showing",
    # "A video that demonstrates",
    # "A video clip of",
    # "A video recording of",
    # "A video featuring",
    # "A video capturing",
    # "A video displaying",
    # "A video presenting",
    # "A video illustrating",
    # "Watch this action where",
    # "Look at this scene showing", 
    # "Here is a demonstration of",
    # "This clip captures",
    # "Observe this recording of",
    # "This moment shows",
    # "Witness this scene of",
    'a photo of {}.',
    'a photo of a person {}.',
    'a photo of a person using {}.',
    'a photo of a person doing {}.',
    'a photo of a person during {}.',
    'a photo of a person performing {}.',
    'a photo of a person practicing {}.',
    'a video of {}.',
    'a video of a person {}.',
    'a video of a person using {}.',
    'a video of a person doing {}.',
    'a video of a person during {}.',
    'a video of a person performing {}.',
    'a video of a person practicing {}.',
    'a example of {}.',
    'a example of a person {}.',
    'a example of a person using {}.',
    'a example of a person doing {}.',
    'a example of a person during {}.',
    'a example of a person performing {}.',
    'a example of a person practicing {}.',
    'a demonstration of {}.',
    'a demonstration of a person {}.',
    'a demonstration of a person using {}.',
    'a demonstration of a person doing {}.',
    'a demonstration of a person during {}.',
    'a demonstration of a person performing {}.',
    'a demonstration of a person practicing {}.',
]

SCENE_TEMPLATES = [
    "{}",
]

from torch import distributed as dist 
has_distributed = True 
import torch.nn.functional as F

class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )

        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )  # H // Ph x W // Pw

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Conv2d to get the patch embeddings

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = (
            pixel_values.shape
        )  # (B, T, 3, H, W)
        pixel_values = pixel_values.reshape(
            batch_size * num_frames, num_channels, height, width
        )  # (B*T, 3, H, W)

        embeddings = self.projection(
            pixel_values
        )  # (B*T, D, H', W')
        patch_width = embeddings.size(-1)
        embeddings = embeddings.flatten(2).transpose(
            1, 2
        )  # (B*T, H'*W', D) or (B*T, N, D)

        return embeddings, num_frames, patch_width


class TimesformerEmbeddingsSigLIP(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        self.attention_type = attention_type
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        # Positional Embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != "space_only":
            # Time Embeddings
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.position_embeddings.shape[1]
        if npatch == N and w == h:
            return self.position_embeddings
        pos_embed = self.position_embeddings.float()
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embeddings.patch_size[0]
        h0 = h // self.patch_embeddings.patch_size[1]
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        # if self.interpolate_offset:
        #     # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        #     # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        #     sx = float(w0 + self.interpolate_offset) / M
        #     sy = float(h0 + self.interpolate_offset) / M
        #     kwargs["scale_factor"] = (sx, sy)
        # else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=True,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    def forward(self, pixel_values):
        batch_size, num_frames, _, height, width = pixel_values.shape
        embeddings, _, _ = self.patch_embeddings(pixel_values)  # (B*T, N, D)

        # Resizing the positional embeddings in case they don't match the input at inference
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, width, height)  # (B*T, N, D)
        embeddings = self.pos_drop(embeddings)

        # Time Embeddings
        if self.attention_type != "space_only":
            # embeddings = embeddings[:, 0:]
            _, num_patch, num_dim = embeddings.shape
            embeddings = (
                embeddings.reshape(batch_size, num_frames, num_patch, num_dim)  # (B, T, N, D)
                .permute(0, 2, 1, 3)  # (B, N, T, D)
                .reshape(batch_size * num_patch, num_frames, num_dim)  # (B*N, T, D)
            )
            # Resizing time embeddings in case they don't match
            if num_frames != self.time_embeddings.size(1):
                time_embeddings = self.time_embeddings.transpose(1, 2)  # (1, D, 8)
                new_time_embeddings = nn.functional.interpolate(
                    time_embeddings, size=(num_frames), mode="nearest"
                )  # (1, D, T)
                new_time_embeddings = new_time_embeddings.transpose(1, 2)  # (1, T, D)
                embeddings = embeddings + new_time_embeddings  # (B*N, T, D)
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.reshape(batch_size, num_patch * num_frames, num_dim)  # (B, N*T, D)

        return embeddings  # (B, N*T, D)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TimeSformer
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class TimesformerCausalSelfAttention(nn.Module):
    def __init__(self, config: SoccerBackboneConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)
        self.register_buffer("mask", torch.tril(torch.ones(config.num_frames, config.num_frames)))
        
    def _add_lora(self, lora_rank):
        # freeze the qkv layer
        for param in self.qkv.parameters():
            param.requires_grad = False
            
        self.qkv_lora_a = nn.Linear(self.qkv.in_features, lora_rank, bias=False)
        self.qkv_lora_b = nn.Linear(lora_rank, self.qkv.out_features,  bias=False)
        self.add_module('qkv_lora_a', self.qkv_lora_a)
        self.add_module('qkv_lora_b', self.qkv_lora_b)
        # move to device
        self.qkv_lora_a.to(self.qkv.weight.device)
        self.qkv_lora_b.to(self.qkv.weight.device)
        
        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.qkv_lora_a.weight, std=0.02)
        nn.init.zeros_(self.qkv_lora_b.weight)
        def lora_forward(self, hidden_states, output_attentions: bool = False):
            batch_size, hidden_size, num_channels = hidden_states.shape
            qkv = (
                (self.qkv(hidden_states) + self.qkv_lora_b(self.qkv_lora_a(hidden_states)))
                .reshape(
                    batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads
                )  # B x C x 3 x num_heads x head_dim
                .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
            )
            query, key, value = qkv[0], qkv[1], qkv[2]

            attention_probs = (query @ key.transpose(-2, -1)) * self.scale
            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = self.attn_drop(attention_probs)

            context_layer = (
                (attention_probs @ value)
                .transpose(1, 2)
                .reshape(batch_size, hidden_size, num_channels)
            )

            outputs = (
                (context_layer, attention_probs) if output_attentions else (context_layer,)
            )

            return outputs
        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)
        
    def forward(self, hidden_states, output_attentions: bool = False):
        batch_size, hidden_size, num_channels = hidden_states.shape  # (B*H*W) x T x C
        qkv = (
            self.qkv(hidden_states)
            .reshape(
                batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads
            )  # B x T x 3 x num_heads x head_dim
            .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x T x head_dim
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_scores = (query @ key.transpose(-2, -1)) * self.scale

        # causal mask
        num_frames = hidden_states.shape[1]
        mask = torch.tril(torch.ones(num_frames, num_frames, device=attention_scores.device, dtype=torch.bool))
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf")) # mask out future frames
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (
            (attention_probs @ value)
            .transpose(1, 2)
            .reshape(batch_size, hidden_size, num_channels)
        )

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
class TimesformerSelfAttention(nn.Module):
    def __init__(self, config: SoccerBackboneConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)

    def _add_lora(self, lora_rank):
        # freeze the qkv layer
        for param in self.qkv.parameters():
            param.requires_grad = False
            
        self.qkv_lora_a = nn.Linear(self.qkv.in_features, lora_rank, bias=False)
        self.qkv_lora_b = nn.Linear(lora_rank, self.qkv.out_features,  bias=False)
        self.add_module('qkv_lora_a', self.qkv_lora_a)
        self.add_module('qkv_lora_b', self.qkv_lora_b)
        # move to device
        self.qkv_lora_a.to(self.qkv.weight.device)
        self.qkv_lora_b.to(self.qkv.weight.device)
        
        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.qkv_lora_a.weight, std=0.02)
        nn.init.zeros_(self.qkv_lora_b.weight)
        def lora_forward(self, hidden_states, output_attentions: bool = False):
            batch_size, hidden_size, num_channels = hidden_states.shape
            qkv = (
                (self.qkv(hidden_states) + self.qkv_lora_b(self.qkv_lora_a(hidden_states)))
                .reshape(
                    batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads
                )  # B x C x 3 x num_heads x head_dim
                .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
            )
            query, key, value = qkv[0], qkv[1], qkv[2]

            attention_probs = (query @ key.transpose(-2, -1)) * self.scale
            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = self.attn_drop(attention_probs)

            context_layer = (
                (attention_probs @ value)
                .transpose(1, 2)
                .reshape(batch_size, hidden_size, num_channels)
            )

            outputs = (
                (context_layer, attention_probs) if output_attentions else (context_layer,)
            )

            return outputs
        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)
    
    def forward(self, hidden_states, output_attentions: bool = False):
        batch_size, hidden_size, num_channels = hidden_states.shape  # B x N x C
        qkv = (
            self.qkv(hidden_states)
            .reshape(
                batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads
            )  # B x C x 3 x num_heads x head_dim
            .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (
            (attention_probs @ value)
            .transpose(1, 2)
            .reshape(batch_size, hidden_size, num_channels)
        )

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def _add_lora(self, lora_rank: int = 32):
        # freeze the dense layer
        for param in self.dense.parameters():
            param.requires_grad = False
        self.dense_lora_a = nn.Linear(self.dense.in_features, lora_rank, bias=False)
        self.dense_lora_b = nn.Linear(lora_rank, self.dense.out_features, bias=False)
        self.add_module('dense_lora_a', self.dense_lora_a)
        self.add_module('dense_lora_b', self.dense_lora_b)
        
        # move to device
        self.dense_lora_a.to(self.dense.weight.device)
        self.dense_lora_b.to(self.dense.weight.device)
        
        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.dense_lora_a.weight, std=0.02)
        nn.init.zeros_(self.dense_lora_b.weight)
        def lora_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.dense(hidden_states) + self.dense_lora_b(self.dense_lora_a(hidden_states))
            hidden_states = self.dropout(hidden_states)

            return hidden_states
        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class TimeSformerCausalAttention(nn.Module):
    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.attention = TimesformerCausalSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
    
class TimeSformerAttention(nn.Module):
    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.attention = TimesformerSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
    
    
class TimesformerIntermediate(nn.Module):
    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
    
class TimesformerOutput(nn.Module):
    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
class TimesformerLayerSigLIP(nn.Module):
    def __init__(self, config: SoccerBackboneConfig, layer_index: int) -> None:
        super().__init__()

        attention_type = config.attention_type

        drop_path_rates = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        drop_path_rate = drop_path_rates[layer_index]

        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.config = config
        self.attention_type = attention_type
        if attention_type not in ["divided_space_time", "space_only", "joint_space_time"]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.enable_causal_temporal:
                print('==> Using causal temporal attention at layer {}'.format(layer_index))
                self.temporal_attention = TimeSformerCausalAttention(config)
            elif not config.enable_causal_temporal:
                print('==> Using bidirectional temporal attention at layer {}'.format(layer_index))
                self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.temporal_attention_gating = nn.Parameter(torch.tensor(0.0))
            # self.temporal_attention_gating = torch.tensor(0.0) ### for ablation, train spatial LoRA only
            # self.register_parameter("temporal_attention_gating", nn.Parameter(torch.tensor(0.0)))
            
    def forward(self, hidden_states: torch.Tensor, num_frames: int, output_attentions: bool = False):
        # hidden_states: (B, N*T, D)

        # num_frames = self.config.num_frames
        # num_patch_width = self.config.image_size // self.config.patch_size
        batch_size = hidden_states.shape[0]
        # num_spatial_tokens = (hidden_states.size(1)) // num_frames # siglip has no cls token
        # num_patch_height = num_spatial_tokens // num_patch_width
        
        if self.attention_type in ["space_only", "joint_space_time"]:
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states), output_attentions=output_attentions
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            hidden_states = hidden_states + self.drop_path(attention_output)

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs
        elif self.attention_type == "divided_space_time":
            # Temporal Attention
            
            temporal_embedding = hidden_states # siglip has no need to remove cls token
            temporal_embedding = temporal_embedding.reshape(-1, num_frames, temporal_embedding.shape[2])  # (B*N, T, D)
            # temporal_embedding = temporal_embedding.reshape(
            #     batch_size, num_patch_height, num_patch_width, num_frames, temporal_embedding.shape[2]
            # ).reshape(batch_size * num_patch_height * num_patch_width, num_frames, temporal_embedding.shape[2]) # (B*N, T, D) s.t. temporal attension is applied to the same patch within frames
            
            temporal_attention_outputs = self.temporal_attention(
                self.temporal_layernorm(temporal_embedding),
            )
            attention_output = temporal_attention_outputs[0] # (B*N, T, D) only the attention output wanted
            
            residual_temporal = self.drop_path(attention_output)  # (B*N, T, D)
            # residual_temporal = residual_temporal.reshape(
            #     batch_size, num_patch_height, num_patch_width, num_frames,
            #     residual_temporal.shape[2]
            # ).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_temporal.shape[2])
            residual_temporal = residual_temporal.reshape(batch_size, -1, residual_temporal.shape[2])  # (B, N*T, D)
            residual_temporal = self.temporal_dense(residual_temporal)
            temporal_embedding = hidden_states + self.temporal_attention_gating.tanh() * residual_temporal  # (B, N*T, D)
            
            # Spatial
            # init_cls_token = hidden_states[:, 0, :].unsqueeze(1) # TODO check shape
            # cls_token = init_cls_token.repeat(1, num_frames, 1) 
            # cls_token = cls_token.reshape(batch_size * num_frames, 1, cls_token.shape[2])
            spatial_embedding = temporal_embedding  # (B, N*T, D)
            # spatial_embedding = (
            #     spatial_embedding.reshape(
            #         batch_size, num_patch_height, num_patch_width, num_frames,
            #         spatial_embedding.shape[2]
            #     )
            #     .permute(0, 3, 1, 2, 4) # B x T x H x W x C
            #     .reshape(batch_size * num_frames, num_patch_height * num_patch_width, spatial_embedding.shape[2]) # (B x T) x (H x W) x C
            # )
            spatial_embedding = spatial_embedding.reshape(
                batch_size, -1, num_frames, spatial_embedding.shape[2]  # (B, N, T, D)
                ).permute(0, 2, 1, 3  # (B, T, N, D)
                ).reshape(batch_size * num_frames, -1, spatial_embedding.shape[2])  # (B*T, N, D)
            
            spatial_attention_outputs = self.attention(
                self.layernorm_before(spatial_embedding),
                output_attentions=output_attentions
            )
            attention_output = spatial_attention_outputs[0]  # (B*T, N, D)
            outputs = spatial_attention_outputs[1:] # TODO check shape here, null?
            
            residual_spatial = self.drop_path(attention_output)  # (B*T, N, D)
            
            # CLS token
            # cls_token = residual_spatial[:, 0, :]
            # cls_token = cls_token.reshape(batch_size, num_frames, cls_token.shape[1])
            # cls_token = torch.mean(cls_token, 1, True) # average over frames
            # residual_spatial = residual_spatial # siglip no need to remove cls token
            # residual_spatial = (
            #     residual_spatial.reshape(
            #         batch_size, num_frames, num_patch_height, num_patch_width, residual_spatial.shape[2]
            #     ) # B x T x H x W x C
            #     .permute(0, 2, 3, 1, 4) # B x H x W x T x C
            #     .reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_spatial.shape[2]) # B x (H x W x T) x C
            # )
            residual_spatial = residual_spatial.reshape(batch_size, num_frames, -1, residual_spatial.shape[2]  # (B, T, N, D)
                ).permute(0, 2, 1, 3  # (B, N, T, D)
                ).reshape(batch_size, -1, residual_spatial.shape[2])  # (B, N*T, D)
            residual = residual_spatial
            hidden_states = temporal_embedding
            
            # MLP
            hidden_states = hidden_states + residual
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)
            
            outputs = (layer_output,) + outputs
            
            return outputs  # (B, N*T, D)
            
            
class TimesformerEncoder(nn.Module):
    def __init__(self, config: SoccerBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TimesformerLayerSigLIP(config, ind) for ind in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N*T, D)
        num_frames: int,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, num_frames, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
            

   
class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SoccerBackboneConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddingsSigLIP):
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        else:
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dim() > 1:
                    nn.init.kaiming_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        


TIMESFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TimesformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TIMESFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.preprocess`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare TimeSformer Model transformer outputting raw hidden-states without any specific head on top.",
    TIMESFORMER_START_DOCSTRING,
)

class TimesformerModel(TimesformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TIMESFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import av
        >>> import numpy as np

        >>> from transformers import AutoImageProcessor, TimesformerModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`List[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

        >>> # prepare video for the model
        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1569, 768]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
        

@add_start_docstrings(
    """TimeSformer Model transformer with a video classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.""",
    TIMESFORMER_START_DOCSTRING,
)
# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Siglip
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class TimesformerSiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        # hidden_state: (B*T, N, D)
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]  # (B*T, 1, D)

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]  # (B*T, D)

class TimesformerModelSigLIP(TimesformerPreTrainedModel):
    """A TimeSFormer utilizing SigLIP's pretrained weights."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)
        # self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = TimesformerSiglipMultiheadAttentionPoolingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)
        pre_pool_output = sequence_output.view(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1)) # reshape to (batch_size * num_frames, patch_num, hidden_size)
        pooled_output = torch.mean(self.head(pre_pool_output).view(sequence_output.size(0), self.config.num_frames, sequence_output.size(-1)), 1, True).squeeze(1)
        
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class TimesformerMultiTaskingModelSigLIP(TimesformerPreTrainedModel):
    """A TimeSFormer utilizing SigLIP's pretrained weights."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)
        # self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = TimesformerSiglipMultiheadAttentionPoolingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def add_lora_spatial(self):
        assert self.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    module.attention._add_lora(32)
                    module.output._add_lora(32)
        print("Added LoRA to the following layers: ", name_list) 
    
    def frozen_spatial(self):
        assert self.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    for param in module.attention.qkv.parameters():
                        param.requires_grad = False
                    for param in module.attention.dense.parameters():
                        param.requires_grad = False
        print("Freezing spatial attention to the following layers: ", name_list) 

    def forward(
        self,
        pixel_values: torch.FloatTensor,  # (B, T, 3, H, W)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embeddings(pixel_values)  # (B, N*T, D)
        B, T, _, _, _ = pixel_values.shape

        encoder_outputs = self.encoder(
            embedding_output,
            num_frames=T,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)  # (B, N*T, D)
        # pre_pool_output = sequence_output.view(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1)) # reshape to (batch_size * num_frames, patch_num, hidden_size)
        # FIX BUG HERE with reshape
        # num_patch_width = self.config.image_size // self.config.patch_size
        # num_patch_height = (encoder_outputs[0].size(1)) // self.config.num_frames // num_patch_width 
        # pre_pool_output = sequence_output.reshape(sequence_output.size(0), num_patch_height, num_patch_width,self.config.num_frames, sequence_output.size(-1)).permute(0, 3, 1, 2, 4).reshape(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1))
        pre_pool_output = sequence_output.reshape(B, -1, T, sequence_output.size(-1)  # (B, N, T, D)
            ).permute(0, 2, 1, 3  # (B, T, N, D)
            ).reshape(B * T, -1, sequence_output.size(-1))  # (B*T, N, D)
        # the reduce step for frame dimension should be done in each task head
        pooled_output = self.head(pre_pool_output  # (B*T, D)
            ).reshape(B, T, sequence_output.size(-1))  # (B, T, D)
        # pooled_output = pooled_output.to(torch.float32) # TODO why float 16 is auto-transformed here
        #pooled_output = torch.mean(self.head(pre_pool_output).view(sequence_output.size(0), self.config.num_frames, sequence_output.size(-1)), 1, True).squeeze(1)
        
        sequence_output = sequence_output.reshape(B, -1, T, sequence_output.size(-1)).permute(0, 2, 1, 3).reshape(B, T, -1, sequence_output.size(-1))
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,  # L x (B, N*T, D)
            attentions=encoder_outputs.attentions,
        )

        
class TimesformerForMultiTaskingSigLIP(TimesformerPreTrainedModel):
    def __init__(self, config, multi_task_config):
        super().__init__(config)
        self.config = config
        
        self.timesformer = TimesformerMultiTaskingModelSigLIP(config)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))
        self.text_tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        self.text_encoder = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
        
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.task_heads = nn.ModuleDict()
        if multi_task_config:
            self.task_types = multi_task_config.keys()
        else:
            self.task_types = []
        for task_type in self.task_types:
            if task_type == "SSV2" or task_type == "Kinetics":
                # recognition
                self.task_heads[task_type] = TimesformerVideoClassificationHead(config, multi_task_config[task_type]["label2id"])
                # self.task_heads[task_type] = TimesformerVideoClassificationLinearHead(config, multi_task_config[task_type]["label2id"])
            elif task_type in ['THUMOS14Grounding','ActivityNetGrounding','FineActionGrounding','HACSGrounding','TaskLocalization']:
                self.task_heads[task_type] = TimesformerUniversalLocalizationHead(config, multi_task_config[task_type]["label2id"])
            elif task_type in ["THUMOS14", "ActivityNet","FineAction", "HACS"]:
                # temporal action localization
                self.task_heads[task_type] = TimesformerNaiveLocalizationHead(config, multi_task_config[task_type]["label2id"])
                # DEBUG actionformer config
                # from models.actionformer.config import default_actionformer_config
                # self.task_heads[task_type] = TimesformerTemporalActionLocalizationHead(config, default_actionformer_config)
            elif task_type in ["MSRVTT", "WebVid", "TaskRetrieval"]:
                # video retrieval
                self.task_heads[task_type] = TimesformerVideoRetrievalHead(config)
            elif task_type in ["CharadesSTA", "QVHighlights", "TaCoS", "TVSum", "ActivityNetCaptions", "DiDeMo", "QuerYD", "TaskGrounding"]:
                # self.task_heads[task_type] = TimesformerTemporalGroundingHead(config)
                self.task_heads[task_type] = TimesformerTemporalGroundingContrastiveHead(config)
            elif task_type in ["YoutubeVIS", "LVVIS", "COCOPseudoVIS", "TaskVIS"]:
                # self.task_heads[task_type] = TimesformerVideoInstanceSegmentationHead(config, multi_task_config[task_type]["label2id"], self.timesformer.head)
                self.task_heads[task_type] = TimesformerUniversalVideoInstanceSegmentationHead(config, multi_task_config[task_type]["label2id"], self.timesformer.head)
                # self.task_heads[task_type] = TimesformerUniversalSigmoidVideoInstanceSegmentationHead(config, multi_task_config[task_type]["label2id"], self.timesformer.head)
            elif task_type in ["MEVIS", "ReferYoutubeVOS", "RefCOCOPseudo", "TaskReferVOS"]:
                # self.task_heads[task_type] = TimesformerVideoContrastiveSegmentationHead(config, multi_task_config[task_type]["label2id"], self.timesformer.head)
                self.task_heads[task_type] = TimesformerVideoContrastiveCrossEntropySegmentationHead(config, multi_task_config[task_type]["label2id"], self.timesformer.head)
            elif task_type == "text_encoder": # TODO use text encoder from task_type for now, change in the future
                self.text_encoder = multi_task_config[task_type]
                for name, param in self.text_encoder.named_parameters():
                    param.requires_grad = False
            elif task_type == "text_tokenizer":
                self.text_tokenizer = multi_task_config[task_type]
            # elif task_type == 'video_instance_segmentation':

            else:
                raise NotImplementedError(f"Task type {task_type} not implemented")
            
        self.post_init()
        
    def frozen_backbone(self):
        for name, param in self.timesformer.named_parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def prepare_for_multi_tasks(self):
        for head in self.task_heads.values():
            head.prepare_multi_task(self.text_encoder, self.text_tokenizer, self.logit_scale, self.logit_bias, self.timesformer)
            
    def add_lora_spatial(self):
        assert self.timesformer.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.timesformer.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    module.attention._add_lora(32)
                    module.output._add_lora(32)
        print("Added LoRA to the following layers: ", name_list) 

    def frozen_spatial(self):
        assert self.timesformer.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.timesformer.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    for param in module.attention.qkv.parameters():
                        param.requires_grad = False
                    for param in module.output.dense.parameters():
                        param.requires_grad = False
        print("Freezing spatial attention to the following layers: ", name_list) 
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multi_task_input: Optional[dict] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = pixel_values.reshape(-1, self.config.num_frames, 3, self.config.image_size, self.config.image_size)
        
        backbone_outputs = self.timesformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        losses = {}
        outputs = {}
        task_name = multi_task_input['task_name'] # one task at a time
        if not self.training:
            outputs[task_name] = self.task_heads[task_name](backbone_outputs, multi_task_input['task_input'])
            return outputs
        losses[task_name], outputs[task_name] = self.task_heads[task_name](backbone_outputs, multi_task_input['task_input'])
        if not return_dict: 
            raise ValueError("return_dict must be True for multi-tasking")
            output = (logits,) + backbone_outputs[1:]
            return ((losses,) + output) if losses is not None else output
        return losses, outputs
        return MultiTaskOutput(
            losses=losses,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
    @torch.no_grad()
    def forward_features(self, pixel_values, pooling_method="mean"):
        backbone_outputs = self.timesformer(
            pixel_values,
        ).pooler_output

        if pooling_method == "mean":
            return torch.mean(backbone_outputs, 1, False)
        elif pooling_method == "no_pooling":
            return backbone_outputs
        else:
            return backbone_outputs[:, -1]# use the last frame
    @torch.no_grad()
    def extract_feature(self, pixel_values, multi_task_input, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        task_name = multi_task_input['task_name']
        # make sure the pixel_values[1] can be divisible by self.config.num_frames, if not, pad the pixel_values
        batch_size = pixel_values.shape[0]
        total_frames = pixel_values.shape[1]
        window_size = 384 # TODO use naive window for iteration to extract features to avoid OOM, to be added to the config in the future
        if total_frames % window_size != 0:
            pad_frames = window_size - total_frames % window_size
            # pad the pixel_values
            pixel_values = torch.cat([pixel_values, torch.zeros(batch_size, pad_frames, 3, self.config.image_size, self.config.image_size).to(pixel_values.device)], dim=1)
        # pixel_values = pixel_values.reshape(-1, self.config.num_frames, 3, self.config.image_size, self.config.image_size)
        # extract the feature
        output_feature_list = []
        multi_task_input['task_input']['masks'] = [] # fake masks for feature extraction
        for i in range(0, pixel_values.shape[1], window_size):
            window_pixel_values = pixel_values[:, i:i+window_size, :, :, :]
            if window_pixel_values.shape[1] < self.config.num_frames:
                window_pixel_values = torch.cat([window_pixel_values, torch.zeros(batch_size, self.config.num_frames - window_pixel_values.shape[1], 3, self.config.image_size, self.config.image_size).to(window_pixel_values.device)], dim=1)
            # construct mask
            mask = torch.ones(window_size).bool()
            if i + window_size > total_frames:
                mask = mask[:total_frames - i]
            multi_task_input['task_input']['masks'].append(mask)
            window_pixel_values = window_pixel_values.reshape(-1, self.config.num_frames, 3, self.config.image_size, self.config.image_size)
            backbone_outputs = self.timesformer(
                window_pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).pooler_output
            
            output_feature_list.append(self.task_heads[task_name].extract_feature(backbone_outputs, multi_task_input['task_input']))
        backbone_outputs = torch.cat(output_feature_list, dim=1)
        backbone_outputs = backbone_outputs.reshape(batch_size, -1, backbone_outputs.shape[2])
        # remove the padded frames
        if total_frames % self.config.num_frames != 0:
            backbone_outputs = backbone_outputs[:, :total_frames, :]
        return backbone_outputs
    
class TimesformerVideoClassificationLinearHead(nn.Module):
    def __init__(self, config, label2id):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.classifier = nn.Linear(config.hidden_size, len(label2id))
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.classifier.bias, 0)
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        pass
        
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None):
        image_embeds = task_head_input.pooler_output[:,-1,:].squeeze(1)
        logits = self.classifier(image_embeds)
        labels = task_specific_input['label']
        loss = F.cross_entropy(logits, labels)
        return loss, logits
        
    
class TimesformerVideoClassificationHead(nn.Module):
    """Video Classification Head
    Using CLIP text encoder in zero-shot video classification
    
    Args:
        config: TimesformerConfig
        label2id: dict, label to id mapping
    """
    def __init__(self, config, label2id):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.video_templates = VIDEO_TEMPLATES
        
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        # initialize the weights
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        
        input_ids = self._tokenize_labels()
        self.label_embeddings = []
        for label_input_ids in input_ids:
            label_input_ids = label_input_ids.to(self.text_encoder.device)
            outputs = self.text_encoder(label_input_ids)[1]
            outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
            outputs = outputs.mean(dim=0)
            self.label_embeddings.append(outputs)
        self.label_embeddings = torch.stack(self.label_embeddings, dim=0)
        # for name, param in self.text_encoder.named_parameters():
        #     param.requires_grad = False
        
    def _tokenize_labels(self):
        # labels = self.config.label2id.keys()
        assert self.label2id is not None, "Please provide label2id dict"
        labels = self.label2id.keys()
        # prompt_template = "A photo of a {} person."
        # input_ids = self.text_tokenizer([prompt_template.format(label) for label in labels], return_tensors="pt", padding="max_length")
        # self.input_ids = input_ids # for debug only
        input_ids = []
        for label in labels:
            texts = [template.format(label) for template in self.video_templates]
            encoded_texts = self.text_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64)['input_ids']
            input_ids.append(encoded_texts)
        
        # input_ids = self.text_tokenizer([f"{random.choice(self.video_templates)} {label}" for label in labels], return_tensors="pt", padding="max_length")
        return input_ids
    
        
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        # image_embeds = torch.mean(task_head_input.pooler_output, 1, True).squeeze(1) # mean pooling to reduce the frame dimension [B, D]
        image_embeds = task_head_input.pooler_output[:,-1,:].squeeze(1)
        text_embeds = self.label_embeddings.detach() # [L, D]
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * self.logit_scale.exp()
            + self.logit_bias
        ) # [L, B]
        logits_per_image = logits_per_text.t() # [B, L]
        targets = task_specific_input['label'] # [B]
        target_labels = -torch.ones_like(logits_per_image) # [B, L]
        target_labels[range(targets.size(0)), targets] = 1 # [B, L], set the target label to 1
        loss = - F.logsigmoid(target_labels * logits_per_image).sum() / targets.size(0)
        return loss, logits_per_image
        return logits_per_image # TODO add in the furture

class TimesformerVideoInstanceSegmentationHead(nn.Module):
    """Video Instance Segmentation Head"""
    def __init__(self, config, label2id, head):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.head = head
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        
        input_ids = self._tokenize_labels()
        input_ids = input_ids.to(self.text_encoder.device) 
        outputs = self.text_encoder(**input_ids)
        self.label_embeddings = outputs[1]

        self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
        self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
        self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
        self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
        self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
        self.head_mlp = copy.deepcopy(vision_model.head.mlp)

        self.w_v.requires_grad = False
        self.v_proj.requires_grad = False
        self.head_layernorm.requires_grad = False
        self.head_mlp.requires_grad = False

    def _dense_feature_projection(self, dense_image_embeds):
        dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
        residual = dense_image_embeds
        dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
        return dense_image_embeds
        
        
    def _tokenize_labels(self):
        # labels = self.config.label2id.keys()
        assert self.label2id is not None, "Please provide label2id dict"
        labels = self.label2id.keys()
        # prompt_template = "A photo of a {}."
        prompt_template = '{}'
        input_ids = self.text_tokenizer([prompt_template.format(label) for label in labels], return_tensors="pt", padding="max_length")
        self.input_ids = input_ids # for debug only
        return input_ids
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
        dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
        text_embeds = self.label_embeddings.detach() # [L, D]
        
        image_embeds = dense_image_embeds / dense_image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        similarity_per_frame_per_patch = torch.einsum('btpd,ld->btpl', image_embeds, text_embeds) # [B, T, (HxW), L]
        logits = similarity_per_frame_per_patch * self.logit_scale.exp() + self.logit_bias # [B, T, (HxW), L]
        if not self.training:
            return logits
        # transform the logits patch-wise to pixel-wise by interpolating the logits
        mask_sizes = task_specific_input["mask_size"]
        
        losses = []
        for i in range(logits.shape[0]):
            patch_size = 14  # TODO hardcoded patch size
            target_h, target_w = mask_sizes[i]  # 720, 1280
            
            # Calculate new dimensions maintaining aspect ratio with height=224
            new_h = 224
            new_w = int(target_w * (new_h / target_h))
            
            # Reshape from (T, patch_h*patch_w, L) to (T, patch_h, patch_w, L)
            curr_logits = logits[i].reshape(logits.shape[1], patch_size, patch_size, logits.shape[3])
            # Permute to (T, L, patch_h, patch_w) for interpolate
            curr_logits = curr_logits.permute(0, 3, 1, 2)
            # Interpolate each frame to new dimensions
            curr_logits = F.interpolate(
                curr_logits, 
                size=(new_h, new_w),
                mode='nearest',
            )  # (T, L, H, W)

            # Also resize the target mask
            mask_target = task_specific_input["mask_target"][i].to(curr_logits.device)  # [T, H, W]
            # mask_target = F.interpolate(
            #     mask_target.unsqueeze(1).float(),
            #     size=(new_h, new_w),
            #     mode='nearest'
            # ).squeeze(1).long()
            # mask_target = fast_priority_downsample(mask_target, (new_h, new_w))
            mask_target = mask_target.masked_fill(mask_target == 0, -1)
            # import ipdb; ipdb.set_trace()
            if (mask_target == -1).all():
                loss = torch.tensor(0.0, device=logits.device)
            else:
                loss = F.cross_entropy(curr_logits, mask_target, ignore_index = -1)
            # loss = focal_loss(curr_logits, mask_target, gamma=5.0)
            losses.append(loss)
        # for i in range(logits.shape[0]):
        #     patch_size = 14 # TODO hardcoded patch size
        #     target_h, target_w = mask_sizes[i]  # 720, 1280
        #     # Reshape from (T, patch_h*patch_w, L) to (T, patch_h, patch_w, L)
        #     curr_logits = logits[i].reshape(logits.shape[1], patch_size, patch_size, logits.shape[3])
        #     # Permute to (T, L, patch_h, patch_w) for interpolate
        #     curr_logits = curr_logits.permute(0, 3, 1, 2)
        #     # Interpolate each frame
        #     # import ipdb; ipdb.set_trace()
        #     curr_logits = F.interpolate(
        #         curr_logits, 
        #         size=(target_h, target_w),
        #         mode='bilinear',
        #         align_corners=False
        #     )  # (T, L, H, W)
            
        #     mask_target = task_specific_input["mask_target"][i].to(curr_logits.device) # [T, H, W]
        #     mask_target = mask_target.masked_fill(mask_target == 0, -1)
        #     # loss = F.cross_entropy(curr_logits, mask_target)
        #     # loss = focal_loss(curr_logits, mask_target, gamma=2.0)
        #     loss = F.cross_entropy(curr_logits, mask_target, ignore_index = -1)
        #     losses.append(loss)
        losses = torch.stack(losses)
        return losses.mean(), logits


class TimesformerUniversalVideoInstanceSegmentationHead(nn.Module):
    """Video Instance Segmentation Head"""
    def __init__(self, config, label2id, head):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.head = head
        self.video_templates = SCENE_TEMPLATES
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer

        self.dataset_label_embeddings = {}
        # self.label2id = {'YoutubeVIS': self.label2id}
        for dataset_name, dataset_label2id in self.label2id.items():
            input_ids = self._tokenize_labels(dataset_label2id)
            # input_ids = input_ids.to(self.text_encoder.device)
            label_embeddings = []
            for label_input_ids in input_ids:
                label_input_ids = label_input_ids.to(self.text_encoder.device)
                outputs = self.text_encoder(label_input_ids)[1]
                outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
                outputs = outputs.mean(dim=0)
                label_embeddings.append(outputs)
            self.dataset_label_embeddings[dataset_name] = torch.stack(label_embeddings, dim=0)
        

        self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
        self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
        self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
        self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
        self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
        self.head_mlp = copy.deepcopy(vision_model.head.mlp)

        self.w_v.requires_grad = False
        self.v_proj.requires_grad = False
        self.head_layernorm.requires_grad = False
        self.head_mlp.requires_grad = False

    def _dense_feature_projection(self, dense_image_embeds):
        dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
        residual = dense_image_embeds
        dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
        return dense_image_embeds
        
    def _tokenize_labels(self, label2id):
        labels = label2id.keys()

        input_ids = []
        for label in labels:
            texts = [template.format(label) for template in self.video_templates]
            encoded_texts = self.text_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64)['input_ids']
            input_ids.append(encoded_texts)
        
        return input_ids
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
        dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
        # text_embeds = self.label_embeddings.detach() # [L, D]
        
        NUM_MAX_CLASSES = 100 # TODO: make this a hyper-parameter
        # 对每个样本分别计算
        losses = []

        #print('dataset_name', task_specific_input["dataset"])
        for i in range(dense_image_embeds.shape[0]):
            dataset_name = task_specific_input["dataset"][i]
            # dataset_name = 'YoutubeVIS'

            curr_image_embeds = dense_image_embeds[i] # [T, (HxW), D]
            curr_image_embeds = curr_image_embeds / curr_image_embeds.norm(p=2, dim=-1, keepdim=True)

            if not self.training:
                curr_text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                continue

            curr_text_embeds = self.dataset_label_embeddings[dataset_name].detach()
            # curr_text_embeds = curr_text_embeds / curr_text_embeds.norm(p=2, dim=-1, keepdim=True)

            # print('dataset_name', dataset_name)
            # print('curr_image_embeds.shape', curr_image_embeds.shape)
            # print('curr_text_embeds.shape', curr_text_embeds.shape)
            # breakpoint()

            # 如果当前样本的类别数大于NUM_MAX_CLASSES，则随机选择负样本类别
            mask_target = task_specific_input["mask_target"][i].to(curr_image_embeds.device)  # [T, H, W]
            
            if curr_text_embeds.shape[0] > NUM_MAX_CLASSES:
                # 获取当前样本中出现的类别
                unique_labels = torch.unique(mask_target)
                # print('Before removing background and ignore classes:', unique_labels)
                unique_labels = unique_labels[unique_labels > 0]  # 移除背景类
                # print('After removing background and ignore classes:', unique_labels)
                # breakpoint()
                # 随机选择负样本类别
                num_neg = min(NUM_MAX_CLASSES - len(unique_labels), curr_text_embeds.shape[0] - len(unique_labels))
                all_indices = set(range(curr_text_embeds.shape[0]))
                pos_indices = set(unique_labels.cpu().numpy())
                neg_indices = list(all_indices - pos_indices)
                selected_neg_indices = random.sample(neg_indices, num_neg)

                # 合并正负样本的索引
                selected_indices = torch.cat([unique_labels, torch.tensor(selected_neg_indices, device=unique_labels.device)])
                
                # 构建新的标签映射
                label_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
                # print('pos_indices', pos_indices)
                # print('neg_indices', neg_indices)
                # print('selected_neg_indices', selected_neg_indices)
                # print('selected_indices', selected_indices)
                # print('label_mapping', label_mapping)
                # breakpoint()

                # 选择对应的text embeddings
                selected_text_embeds = curr_text_embeds[selected_indices]
                selected_text_embeds = selected_text_embeds / selected_text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # 计算相似度
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, selected_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                
                # print('selected_text_embeds.shape', selected_text_embeds.shape)
                # print('curr_logits.shape', curr_logits.shape)
                # breakpoint()

                # 调整mask_target的标签
                new_mask_target = torch.ones_like(mask_target).long() * -1
                for old_idx, new_idx in label_mapping.items():
                    new_mask_target[mask_target == old_idx] = new_idx

            else:
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                new_mask_target = mask_target.clone().long()
                new_mask_target = new_mask_target.masked_fill(new_mask_target == 0, -1)

            # print('old mask_target unique', torch.unique(mask_target))
            # print('new_mask_target unique', torch.unique(new_mask_target))
            # print('mask size', task_specific_input["mask_size"][i])
            # breakpoint()

            # 计算loss
            patch_size = 14
            target_h, target_w = task_specific_input["mask_size"][i]
            new_h = 224
            new_w = int(target_w * (new_h / target_h))
            
            curr_logits = curr_logits.reshape(curr_logits.shape[0], patch_size, patch_size, curr_logits.shape[2])
            curr_logits = curr_logits.permute(0, 3, 1, 2)

            # curr_logits = F.interpolate(
            #     curr_logits, 
            #     size=(new_h, new_w),
            #     mode='nearest',
            # )

            curr_logits = F.interpolate(
                curr_logits, 
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

            # new_mask_target = F.interpolate(
            #     new_mask_target.unsqueeze(1).float(),
            #     size=(patch_size, patch_size),
            #     mode='nearest'
            # ).squeeze(1).long()
            
            # new_mask_target = fast_priority_downsample(new_mask_target, (patch_size, patch_size))  # [T, new_h, new_w]

            # new_mask_target = F.interpolate(
            #     new_mask_target.unsqueeze(1).float(),
            #     size=(new_h, new_w),
            #     mode='nearest'
            # ).squeeze(1).long()

            # print('curr_logits.shape', curr_logits.shape)
            # print('new_mask_target.shape', new_mask_target.shape)
            # print('new_mask_target', torch.unique(new_mask_target))
            # # print(torch.where(new_mask_target > 0)[0].sum())
            # breakpoint()

            if (new_mask_target == -1).all():
                loss = torch.tensor(0.0, device=curr_logits.device)
            else:
                loss = F.cross_entropy(curr_logits, new_mask_target, ignore_index=-1)
            losses.append(loss)

            # print('dataset_name', dataset_name)
            # print('curr_logits.shape', curr_logits.shape)
            # print('new_mask_target.shape', new_mask_target.shape, torch.unique(new_mask_target))
            # print('loss', loss)
            # breakpoint()

        if self.training:
            losses = torch.stack(losses)
            return losses.mean(), None
        else:
            return None

            
# class TimesformerUniversalSigmoidVideoInstanceSegmentationHead(nn.Module):
#     """Video Instance Segmentation Head"""
#     def __init__(self, config, label2id, head):
#         super().__init__()
#         self.config = config
#         self.label2id = label2id
#         self.head = head
#         self.video_templates = SCENE_TEMPLATES
        
#     def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
#         self.logit_scale = copy.deepcopy(logit_scale)
#         self.logit_bias = copy.deepcopy(logit_bias)
        
#         self.text_encoder = text_encoder
#         self.text_tokenizer = text_tokenizer

#         self.dataset_label_embeddings = {}
#         for dataset_name, dataset_label2id in self.label2id.items():
#             input_ids = self._tokenize_labels(dataset_label2id)
#             label_embeddings = []
#             for label_input_ids in input_ids:
#                 label_input_ids = label_input_ids.to(self.text_encoder.device)
#                 outputs = self.text_encoder(label_input_ids)[1]
#                 outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
#                 outputs = outputs.mean(dim=0)
#                 label_embeddings.append(outputs)
#             self.dataset_label_embeddings[dataset_name] = torch.stack(label_embeddings, dim=0)
        
#         self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
#         self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
#         self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
#         self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
#         self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
#         self.head_mlp = copy.deepcopy(vision_model.head.mlp)

#         self.w_v.requires_grad = False
#         self.v_proj.requires_grad = False
#         self.head_layernorm.requires_grad = False
#         self.head_mlp.requires_grad = False

#     def _dense_feature_projection(self, dense_image_embeds):
#         dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
#         dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
#         residual = dense_image_embeds
#         dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
#         dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
#         return dense_image_embeds
        
#     def _tokenize_labels(self, label2id):
#         labels = label2id.keys()

#         input_ids = []
#         for label in labels:
#             texts = [template.format(label) for template in self.video_templates]
#             encoded_texts = self.text_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64)['input_ids']
#             input_ids.append(encoded_texts)
        
#         return input_ids
    
#     def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
#         last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
#         dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
#         # text_embeds = self.label_embeddings.detach() # [L, D]
        
#         NUM_MAX_CLASSES = 100 # TODO: make this a hyper-parameter
#         # 对每个样本分别计算
#         losses = []

#         #print('dataset_name', task_specific_input["dataset"])
#         for i in range(dense_image_embeds.shape[0]):
#             dataset_name = task_specific_input["dataset"][i]
#             # dataset_name = 'YoutubeVIS'

#             curr_image_embeds = dense_image_embeds[i] # [T, (HxW), D]
#             curr_image_embeds = curr_image_embeds / curr_image_embeds.norm(p=2, dim=-1, keepdim=True)

#             if not self.training:
#                 curr_text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
#                 curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
#                 curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
#                 continue

#             curr_text_embeds = self.dataset_label_embeddings[dataset_name].detach()
#             # curr_text_embeds = curr_text_embeds / curr_text_embeds.norm(p=2, dim=-1, keepdim=True)

#             # print('dataset_name', dataset_name)
#             # print('curr_image_embeds.shape', curr_image_embeds.shape)
#             # print('curr_text_embeds.shape', curr_text_embeds.shape)
#             # breakpoint()

#             # 如果当前样本的类别数大于NUM_MAX_CLASSES，则随机选择负样本类别
#             mask_target = task_specific_input["mask_target"][i].to(curr_image_embeds.device)  # [T, H, W]
            
#             if curr_text_embeds.shape[0] > NUM_MAX_CLASSES:
#                 # 获取当前样本中出现的类别
#                 unique_labels = torch.unique(mask_target)
#                 # print('Before removing background and ignore classes:', unique_labels)
#                 unique_labels = unique_labels[unique_labels > 0]  # 移除背景类
#                 # print('After removing background and ignore classes:', unique_labels)
#                 # breakpoint()
#                 # 随机选择负样本类别
#                 num_neg = min(NUM_MAX_CLASSES - len(unique_labels), curr_text_embeds.shape[0] - len(unique_labels))
#                 all_indices = set(range(curr_text_embeds.shape[0]))
#                 pos_indices = set(unique_labels.cpu().numpy())
#                 neg_indices = list(all_indices - pos_indices)
#                 selected_neg_indices = random.sample(neg_indices, num_neg)

#                 # 合并正负样本的索引
#                 selected_indices = torch.cat([unique_labels, torch.tensor(selected_neg_indices, device=unique_labels.device)])
                
#                 # 构建新的标签映射
#                 label_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
#                 # print('pos_indices', pos_indices)
#                 # print('neg_indices', neg_indices)
#                 # print('selected_neg_indices', selected_neg_indices)
#                 # print('selected_indices', selected_indices)
#                 # print('label_mapping', label_mapping)
#                 # breakpoint()

#                 # 选择对应的text embeddings
#                 selected_text_embeds = curr_text_embeds[selected_indices]
#                 selected_text_embeds = selected_text_embeds / selected_text_embeds.norm(p=2, dim=-1, keepdim=True)
                
#                 # 计算相似度
#                 curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, selected_text_embeds)
#                 curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                
#                 # print('selected_text_embeds.shape', selected_text_embeds.shape)
#                 # print('curr_logits.shape', curr_logits.shape)
#                 # breakpoint()

#                 # 调整mask_target的标签
#                 new_mask_target = torch.ones_like(mask_target).long() * -1
#                 for old_idx, new_idx in label_mapping.items():
#                     new_mask_target[mask_target == old_idx] = new_idx

#             else:
#                 curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
#                 curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
#                 new_mask_target = mask_target.clone().long()
#                 new_mask_target = new_mask_target.masked_fill(new_mask_target == 0, -1)

#             # print('old mask_target unique', torch.unique(mask_target))
#             # print('new_mask_target unique', torch.unique(new_mask_target))
#             # print('mask size', task_specific_input["mask_size"][i])
#             # breakpoint()
            

#             # 计算loss
#             patch_size = 14
#             target_h, target_w = task_specific_input["mask_size"][i]
#             new_h = 224
#             new_w = int(target_w * (new_h / target_h))
            
#             curr_logits = curr_logits.reshape(curr_logits.shape[0], patch_size, patch_size, curr_logits.shape[2])
#             curr_logits = curr_logits.permute(0, 3, 1, 2)
#             # curr_logits = F.interpolate(
#             #     curr_logits, 
#             #     size=(new_h, new_w),
#             #     mode='nearest',
#             # )
#             curr_logits = F.interpolate(
#                 curr_logits, 
#                 size=(new_h, new_w),
#                 mode='bilinear',
#                 align_corners=False
#             )

#             curr_logits = curr_logits.permute(0, 2, 3, 1) #[t, h, w, class]

#             final_mask_target = -torch.ones_like(curr_logits).to(curr_logits.device)
#             mask_t, mask_h, mask_w = new_mask_target.shape
#             new_mask_target = rearrange(new_mask_target, 't h w -> (t h w)').cpu()
#             new_mask_target_foreground = new_mask_target >= 0
#             final_mask_target = rearrange(final_mask_target, 't h w l -> (t h w) l')
#             final_mask_target[torch.arange(len(new_mask_target_foreground))[new_mask_target_foreground], new_mask_target[new_mask_target_foreground]] = 1
#             final_mask_target = rearrange(final_mask_target, '(t h w) l -> t h w l', t=mask_t, h=mask_h, w=mask_w)
            
#             # print('mask_target unique', torch.unique(mask_target))
#             # print('new_mask_target unique', torch.unique(new_mask_target))
#             # print('final_mask_target unique', torch.unique(final_mask_target))
#             # print('torch.where new_mask_target', torch.where(new_mask_target >= 0))
#             # print('torch.where final_mask_target', torch.where(final_mask_target.view(-1, final_mask_target.shape[-1]) == 1))
#             # breakpoint()

#             # new_mask_target = F.interpolate(
#             #     new_mask_target.unsqueeze(1).float(),
#             #     size=(new_h, new_w),
#             #     mode='nearest'
#             # ).squeeze(1).long()

#             # print('curr_logits.shape', curr_logits.shape)
#             # print('new_mask_target.shape', new_mask_target.shape)
#             # print('new_mask_target', torch.unique(new_mask_target))
#             # breakpoint()

#             if (new_mask_target == -1).all():
#                 loss = torch.tensor(0.0, device=curr_logits.device)
#             else:
#                 loss = -F.logsigmoid(final_mask_target * curr_logits).sum() / (curr_logits.shape[0] * curr_logits.shape[1] * curr_logits.shape[2])
#             losses.append(loss)

#             # print('dataset_name', dataset_name)
#             # print('curr_logits.shape', curr_logits.shape)
#             # print('new_mask_target.shape', new_mask_target.shape, torch.unique(new_mask_target))
#             # print('loss', loss)
#             # breakpoint()

#         if self.training:
#             losses = torch.stack(losses)
#             return losses.mean(), None
#         else:
#             return None

class TimesformerUniversalSigmoidVideoInstanceSegmentationHead(nn.Module):
    """Video Instance Segmentation Head"""
    def __init__(self, config, label2id, head):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.head = head
        self.video_templates = SCENE_TEMPLATES
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer

        self.dataset_label_embeddings = {}
        for dataset_name, dataset_label2id in self.label2id.items():
            input_ids = self._tokenize_labels(dataset_label2id)
            label_embeddings = []
            for label_input_ids in input_ids:
                label_input_ids = label_input_ids.to(self.text_encoder.device)
                outputs = self.text_encoder(label_input_ids)[1]
                outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
                outputs = outputs.mean(dim=0)
                label_embeddings.append(outputs)
            self.dataset_label_embeddings[dataset_name] = torch.stack(label_embeddings, dim=0)
        
        self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
        self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
        self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
        self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
        self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
        self.head_mlp = copy.deepcopy(vision_model.head.mlp)

        self.w_v.requires_grad = False
        self.v_proj.requires_grad = False
        self.head_layernorm.requires_grad = False
        self.head_mlp.requires_grad = False

    def _dense_feature_projection(self, dense_image_embeds):
        dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
        residual = dense_image_embeds
        dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
        return dense_image_embeds
        
    def _tokenize_labels(self, label2id):
        labels = label2id.keys()

        input_ids = []
        for label in labels:
            texts = [template.format(label) for template in self.video_templates]
            encoded_texts = self.text_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64)['input_ids']
            input_ids.append(encoded_texts)
        
        return input_ids
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
        dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
        # text_embeds = self.label_embeddings.detach() # [L, D]
        
        NUM_MAX_CLASSES = 100 # TODO: make this a hyper-parameter
        # 对每个样本分别计算
        losses = []

        #print('dataset_name', task_specific_input["dataset"])
        for i in range(dense_image_embeds.shape[0]):
            dataset_name = task_specific_input["dataset"][i]
            # dataset_name = 'YoutubeVIS'

            curr_image_embeds = dense_image_embeds[i] # [T, (HxW), D]
            curr_image_embeds = curr_image_embeds / curr_image_embeds.norm(p=2, dim=-1, keepdim=True)

            if not self.training:
                curr_text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                continue

            curr_text_embeds = self.dataset_label_embeddings[dataset_name].detach()
            # curr_text_embeds = curr_text_embeds / curr_text_embeds.norm(p=2, dim=-1, keepdim=True)

            # print('dataset_name', dataset_name)
            # print('curr_image_embeds.shape', curr_image_embeds.shape)
            # print('curr_text_embeds.shape', curr_text_embeds.shape)
            # breakpoint()

            # 如果当前样本的类别数大于NUM_MAX_CLASSES，则随机选择负样本类别
            mask_target = task_specific_input["mask_target"][i].to(curr_image_embeds.device)  # [T, H, W]
            
            if curr_text_embeds.shape[0] > NUM_MAX_CLASSES:
                # 获取当前样本中出现的类别
                unique_labels = torch.unique(mask_target)
                # print('Before removing background and ignore classes:', unique_labels)
                unique_labels = unique_labels[unique_labels > 0]  # 移除背景类
                # print('After removing background and ignore classes:', unique_labels)
                # breakpoint()
                # 随机选择负样本类别
                num_neg = min(NUM_MAX_CLASSES - len(unique_labels), curr_text_embeds.shape[0] - len(unique_labels))
                all_indices = set(range(curr_text_embeds.shape[0]))
                pos_indices = set(unique_labels.cpu().numpy())
                neg_indices = list(all_indices - pos_indices)
                selected_neg_indices = random.sample(neg_indices, num_neg)

                # 合并正负样本的索引
                selected_indices = torch.cat([unique_labels, torch.tensor(selected_neg_indices, device=unique_labels.device)])
                
                # 构建新的标签映射
                label_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
                # print('pos_indices', pos_indices)
                # print('neg_indices', neg_indices)
                # print('selected_neg_indices', selected_neg_indices)
                # print('selected_indices', selected_indices)
                # print('label_mapping', label_mapping)
                # breakpoint()

                # 选择对应的text embeddings
                selected_text_embeds = curr_text_embeds[selected_indices]
                selected_text_embeds = selected_text_embeds / selected_text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # 计算相似度
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, selected_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                
                # print('selected_text_embeds.shape', selected_text_embeds.shape)
                # print('curr_logits.shape', curr_logits.shape)
                # breakpoint()

                # 调整mask_target的标签
                new_mask_target = torch.ones_like(mask_target).long() * -1
                for old_idx, new_idx in label_mapping.items():
                    new_mask_target[mask_target == old_idx] = new_idx

            else:
                curr_similarity = torch.einsum('tpd,ld->tpl', curr_image_embeds, curr_text_embeds)
                curr_logits = curr_similarity * self.logit_scale.exp() + self.logit_bias
                new_mask_target = mask_target.clone().long()
                new_mask_target = new_mask_target.masked_fill(new_mask_target == 0, -1)

            # print('old mask_target unique', torch.unique(mask_target))
            # print('new_mask_target unique', torch.unique(new_mask_target))
            # print('mask size', task_specific_input["mask_size"][i])
            # breakpoint()
            

            # 计算loss
            patch_size = 14
            target_h, target_w = task_specific_input["mask_size"][i]
            new_h = 224
            new_w = int(target_w * (new_h / target_h))
            
            curr_logits = curr_logits.reshape(curr_logits.shape[0], patch_size, patch_size, curr_logits.shape[2])
            curr_logits = curr_logits.permute(0, 3, 1, 2)
            # curr_logits = F.interpolate(
            #     curr_logits, 
            #     size=(new_h, new_w),
            #     mode='nearest',
            # )
            curr_logits = F.interpolate(
                curr_logits, 
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

            curr_logits = curr_logits.permute(0, 2, 3, 1) #[t, h, w, class]

            final_mask_target = -torch.ones_like(curr_logits).to(curr_logits.device)
            mask_t, mask_h, mask_w = new_mask_target.shape
            new_mask_target = rearrange(new_mask_target, 't h w -> (t h w)').cpu()
            new_mask_target_foreground = new_mask_target >= 0
            final_mask_target = rearrange(final_mask_target, 't h w l -> (t h w) l')
            final_mask_target[torch.arange(len(new_mask_target_foreground))[new_mask_target_foreground], new_mask_target[new_mask_target_foreground]] = 1
            final_mask_target = rearrange(final_mask_target, '(t h w) l -> t h w l', t=mask_t, h=mask_h, w=mask_w)
            
            # print('mask_target unique', torch.unique(mask_target))
            # print('new_mask_target unique', torch.unique(new_mask_target))
            # print('final_mask_target unique', torch.unique(final_mask_target))
            # print('torch.where new_mask_target', torch.where(new_mask_target >= 0))
            # print('torch.where final_mask_target', torch.where(final_mask_target.view(-1, final_mask_target.shape[-1]) == 1))
            # breakpoint()

            # new_mask_target = F.interpolate(
            #     new_mask_target.unsqueeze(1).float(),
            #     size=(new_h, new_w),
            #     mode='nearest'
            # ).squeeze(1).long()

            # print('curr_logits.shape', curr_logits.shape)
            # print('new_mask_target.shape', new_mask_target.shape)
            # print('new_mask_target', torch.unique(new_mask_target))
            # breakpoint()
            
            final_mask_target = final_mask_target.masked_fill(final_mask_target == -1, 0)
            # 计算前景部分的掩码
            foreground_mask = (final_mask_target != 0)  # 前景部分为 1，背景部分为 

            if (new_mask_target == -1).all():
                loss = torch.tensor(0.0, device=curr_logits.device)
            else:    
                # 计算损失
                loss = -F.logsigmoid(final_mask_target * curr_logits)  # 计算每个像素的损失
                loss = loss * foreground_mask  # 只保留前景部分的损失
                loss = loss.sum() / foreground_mask.sum()

            losses.append(loss)
            # print('dataset_name', dataset_name)
            # print('curr_logits.shape', curr_logits.shape)
            # print('new_mask_target.shape', new_mask_target.shape, torch.unique(new_mask_target))
            # print('loss', loss)
            # breakpoint()

        if self.training:
            losses = torch.stack(losses)
            return losses.mean(), None
        else:
            return None


class TimesformerVideoContrastiveSegmentationHead(nn.Module):
    """Video Instance Segmentation Head implemented with contrastive loss"""
    def __init__(self, config, label2id, head):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.head = head
        self.world_size = dist.get_world_size()
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer

        self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
        self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
        self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
        self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
        self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
        self.head_mlp = copy.deepcopy(vision_model.head.mlp)

        self.w_v.requires_grad = False
        self.v_proj.requires_grad = False
        self.head_layernorm.requires_grad = False
        self.head_mlp.requires_grad = False

    def _dense_feature_projection(self, dense_image_embeds):
        dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
        residual = dense_image_embeds
        dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
        return dense_image_embeds
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
        dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
        
        # Encode text captions
        captions = task_specific_input["caption"]  # List[str] of length B
        text_inputs = self.text_tokenizer(captions, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(dense_image_embeds.device)
        text_outputs = self.text_encoder(**text_inputs)
        text_embeds = text_outputs[1]  # [B, D]
        
        # 收集所有GPU上的text embeddings
        gathered_text = [torch.zeros_like(text_embeds) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_text, text_embeds)
        text_embeds_all = torch.cat(gathered_text, dim=0)  # [world_size*B, D]
        
        # 归一化
        image_embeds = dense_image_embeds / dense_image_embeds.norm(p=2, dim=-1, keepdim=True)  # [B, T, (HxW), D]
        text_embeds_all = text_embeds_all / text_embeds_all.norm(p=2, dim=-1, keepdim=True)  # [world_size*B, D]
        
        # 计算patch级别的相似度
        similarity = torch.einsum('btpd,nd->btpn', image_embeds, text_embeds_all)  # [B, T, (HxW), world_size*B]
        similarity = similarity * self.logit_scale.exp() + self.logit_bias

        # print('image_embeds.shape', image_embeds.shape)
        # print('text_embeds_all.shape', text_embeds_all.shape)
        # print('similarity.shape', similarity.shape)
        # breakpoint()
        
        if not self.training:
            # 对于推理，只返回与当前batch中text的相似度
            return similarity[..., :text_embeds.shape[0]]
            
        # 对每个样本计算contrastive loss
        total_loss = 0
        total_samples = 0
        mask_sizes = task_specific_input["mask_size"]
        
        for i in range(dense_image_embeds.shape[0]):
            patch_size = 14  # TODO hardcoded patch size
            target_h, target_w = mask_sizes[i]  # 720, 1280
            
            # Calculate new dimensions maintaining aspect ratio with height=224
            new_h = 224  # 使用较小的分辨率以节省内存
            new_w = int(target_w * (new_h / target_h))
            
            # Reshape and interpolate similarity scores
            curr_sim = similarity[i]  # [T, (HxW), world_size*B]
            curr_sim = curr_sim.reshape(curr_sim.shape[0], patch_size, patch_size, curr_sim.shape[2])  # [T, patch_h, patch_w, world_size*B]
            curr_sim = curr_sim.permute(0, 3, 1, 2)  # [T, world_size*B, patch_h, patch_w]
            # curr_sim = F.interpolate(
            #     curr_sim, 
            #     size=(new_h, new_w), 
            #     mode='nearest'
            # )  # [T, world_size*B, new_h, new_w]
            curr_sim = F.interpolate(
                curr_sim, 
                size=(new_h, new_w), 
                mode='bilinear',
                align_corners=False
            )  # [T, world_size*B, new_h, new_w]
            
            # Get current rank and index
            curr_rank = dist.get_rank()
            curr_index = curr_rank * text_embeds.shape[0] + i
            
            # Get and resize mask
            mask_target = task_specific_input["mask_target"][i].to(curr_sim.device).long()  # [T, H, W]
            # mask_target = F.interpolate(
            #     mask_target.unsqueeze(1).float(),
            #     size=(new_h, new_w),
            #     mode='nearest'
            # ).squeeze(1).long()
            
            # mask_target = fast_priority_downsample(mask_target, (new_h, new_w))  # [T, new_h, new_w]

            
            # print('curr_sim.shape', curr_sim.shape)
            # print('mask_target.shape', mask_target.shape)
            # breakpoint()

            # Prepare labels
            curr_sim = curr_sim.permute(0, 2, 3, 1)  # [T, new_h, new_w, world_size*B]
            labels = torch.ones_like(curr_sim) * -1  # [T, new_h, new_w, world_size*B]
            
            # print('curr_index', curr_index) # 0
            # print('curr_sim.shape', curr_sim.shape) # [16, 224, 286, 16]
            # print('labels.shape', labels.shape) # [16, 224, 286, 16]
            # breakpoint()

            # # Set positive labels
            # labels[mask_target == 1, curr_index] = 1
            # Set positive labels - 修复维度不匹配的问题
            mask_target_expanded = mask_target.unsqueeze(-1).expand(-1, -1, -1, labels.size(-1))  # [T, new_h, new_w, world_size*B]
            labels = torch.where(
                (mask_target_expanded == 1) & (torch.arange(labels.size(-1), device=labels.device) == curr_index),
                torch.ones_like(labels),
                labels
            )
            
            # Calculate loss
            loss = -F.logsigmoid(labels * curr_sim).sum() / (curr_sim.shape[0] * curr_sim.shape[1] * curr_sim.shape[2])  # 除以总像素数
            
            # print('curr_sim.shape', curr_sim.shape)
            # print('mask_target.shape', mask_target.shape)
            # print('labels.shape', labels.shape)
            # print('loss', loss)
            # breakpoint()

            total_loss += loss
            total_samples += 1
        
        # 如果没有任何样本，返回零损失
        if total_samples == 0:
            return torch.tensor(0.0, device=dense_image_embeds.device), None
            
        # 计算平均loss
        avg_loss = total_loss / total_samples
        return avg_loss, similarity



class TimesformerVideoContrastiveCrossEntropySegmentationHead(nn.Module):
    """Video Instance Segmentation Head implemented with contrastive loss"""
    def __init__(self, config, label2id, head):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.head = head
        self.world_size = dist.get_world_size()
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer

        self.w_v = nn.Linear(vision_model.config.hidden_size, vision_model.config.hidden_size, bias=True)
        self.w_v.weight.data = copy.deepcopy(vision_model.head.attention.in_proj_weight.data[2*vision_model.config.hidden_size:, :])
        self.w_v.bias.data = copy.deepcopy(vision_model.head.attention.in_proj_bias.data[2*vision_model.config.hidden_size:])
        self.v_proj = copy.deepcopy(vision_model.head.attention.out_proj)
        self.head_layernorm = copy.deepcopy(vision_model.head.layernorm)
        self.head_mlp = copy.deepcopy(vision_model.head.mlp)

        self.w_v.requires_grad = False
        self.v_proj.requires_grad = False
        self.head_layernorm.requires_grad = False
        self.head_mlp.requires_grad = False

    def _dense_feature_projection(self, dense_image_embeds):
        dense_image_embeds = self.w_v(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.v_proj(dense_image_embeds) # [B, (HxW), D]
        
        residual = dense_image_embeds
        dense_image_embeds = self.head_layernorm(dense_image_embeds) # [B, (HxW), D]
        dense_image_embeds = self.head_mlp(dense_image_embeds) + residual # [B, (HxW), D]
        return dense_image_embeds
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        last_hidden_state = task_head_input.last_hidden_state # [B, T, (HxW), D]
        dense_image_embeds = self._dense_feature_projection(last_hidden_state.reshape(-1, last_hidden_state.shape[2], last_hidden_state.shape[3])).reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2], last_hidden_state.shape[3]) # [B, T, (HxW), D]
        
        # Encode text captions
        captions = task_specific_input["caption"]  # List[str] of length B
        text_inputs = self.text_tokenizer(captions, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(dense_image_embeds.device)
        text_outputs = self.text_encoder(**text_inputs)
        text_embeds = text_outputs[1]  # [B, D]
        
        # 收集所有GPU上的text embeddings
        gathered_text = [torch.zeros_like(text_embeds) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_text, text_embeds)
        text_embeds_all = torch.cat(gathered_text, dim=0)  # [world_size*B, D]
        
        # 归一化
        image_embeds = dense_image_embeds / dense_image_embeds.norm(p=2, dim=-1, keepdim=True)  # [B, T, (HxW), D]
        text_embeds_all = text_embeds_all / text_embeds_all.norm(p=2, dim=-1, keepdim=True)  # [world_size*B, D]
        
        # 计算patch级别的相似度
        similarity = torch.einsum('btpd,nd->btpn', image_embeds, text_embeds_all)  # [B, T, (HxW), world_size*B]
        similarity = similarity * self.logit_scale.exp() + self.logit_bias

        # print('image_embeds.shape', image_embeds.shape)
        # print('text_embeds_all.shape', text_embeds_all.shape)
        # print('similarity.shape', similarity.shape)
        # breakpoint()
        
        if not self.training:
            # 对于推理，只返回与当前batch中text的相似度
            return similarity[..., :text_embeds.shape[0]]
            
        # 对每个样本计算contrastive loss
        total_loss = 0
        total_samples = 0
        mask_sizes = task_specific_input["mask_size"]
        
        for i in range(dense_image_embeds.shape[0]):
            patch_size = 14  # TODO hardcoded patch size
            target_h, target_w = mask_sizes[i]  # 720, 1280
            
            # Calculate new dimensions maintaining aspect ratio with height=224
            new_h = 224  # 使用较小的分辨率以节省内存
            new_w = int(target_w * (new_h / target_h))
            
            # Reshape and interpolate similarity scores
            curr_sim = similarity[i]  # [T, (HxW), world_size*B]
            curr_sim = curr_sim.reshape(curr_sim.shape[0], patch_size, patch_size, curr_sim.shape[2])  # [T, patch_h, patch_w, world_size*B]
            curr_sim = curr_sim.permute(0, 3, 1, 2)  # [T, world_size*B, patch_h, patch_w]
            # curr_sim = F.interpolate(
            #     curr_sim, 
            #     size=(new_h, new_w), 
            #     mode='nearest'
            # )  # [T, world_size*B, new_h, new_w]
            curr_sim = F.interpolate(
                curr_sim, 
                size=(new_h, new_w), 
                mode='bilinear',
                align_corners=False
            )  # [T, world_size*B, new_h, new_w]
            
            # Get current rank and index
            curr_rank = dist.get_rank()
            curr_index = curr_rank * text_embeds.shape[0] + i
            
            # Get and resize mask
            mask_target = task_specific_input["mask_target"][i].to(curr_sim.device).long()  # [T, H, W]

            # Prepare labels
            labels = -torch.ones((curr_sim.shape[0], new_h, new_w), dtype=torch.long, device=curr_sim.device)                 
            labels[mask_target == 1] = curr_index

            curr_sim = curr_sim.permute(0, 2, 3, 1)  # [T, new_h, new_w, world_size*B]

            logits_flat = curr_sim.view(-1, curr_sim.size(-1))  # [T * new_h * new_w, world_size*B]
            labels_flat = labels.view(-1)  # [T * new_h * new_w * world_size*B]

            # print('Check location')
            # print(torch.where(mask_target == 1))
            # print(torch.where(labels == curr_index))
            # breakpoint()
            # Calculate loss
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-1)
            
            # print('curr_sim.shape', curr_sim.shape)
            # print('mask_target.shape', mask_target.shape)
            # print('labels.shape', labels.shape)
            # print('loss', loss)
            # breakpoint()

            total_loss += loss
            total_samples += 1
        
        # 如果没有任何样本，返回零损失
        if total_samples == 0:
            return torch.tensor(0.0, device=dense_image_embeds.device), None
            
        # 计算平均loss
        avg_loss = total_loss / total_samples
        return avg_loss, similarity


def fast_priority_downsample(mask, new_size):
    # mask: [T, H, W]
    mask = mask.unsqueeze(1).float()  # [T, 1, H, W]
    return F.adaptive_max_pool2d(mask, output_size=new_size).squeeze(1).long()
# def fast_priority_downsample(mask, new_size):
#     # mask: [T, H, W]
#     mask = mask.unsqueeze(1).float()  # [T, 1, H, W]
#     kernel_h = mask.shape[2] // new_size[0]
#     kernel_w = mask.shape[3] // new_size[1]
#     return F.max_pool2d(mask, kernel_size=(kernel_h, kernel_w), 
#                        stride=(kernel_h, kernel_w)).squeeze(1).long()
def area_voting_downsample(mask, new_size):
    device = mask.device
    B, H, W = mask.shape
    new_h, new_w = new_size
    
    h_stride = H // new_h
    w_stride = W // new_w
    
    output = torch.zeros((B, new_h, new_w), device=device)
    
    for i in range(new_h):
        for j in range(new_w):
            h_start = i * h_stride
            w_start = j * w_stride
            h_end = min((i + 1) * h_stride, H)
            w_end = min((j + 1) * w_stride, W)
            
            window = mask[:, h_start:h_end, w_start:w_end]
            unique_labels = torch.unique(window)
            
            if len(unique_labels) == 1:
                output[:, i, j] = unique_labels[0]
            else:
                weights = torch.ones_like(window, dtype=torch.float)
                weights[window != 0] = 4.0  
                
                max_vote = 0
                max_label = 0
                
                for label in unique_labels:
                    votes = (weights * (window == label)).sum()
                    if votes > max_vote:
                        max_vote = votes
                        max_label = label
                
                output[:, i, j] = max_label
    
    return output.long()

# 使用方法
def priority_downsample(mask, new_size):
    device = mask.device
    B, H, W = mask.shape
    new_h, new_w = new_size
    
    # 计算池化窗口大小
    h_stride = H // new_h
    w_stride = W // new_w
    
    output = torch.zeros((B, new_h, new_w), device=device)
    
    for i in range(new_h):
        for j in range(new_w):
            h_start = i * h_stride
            w_start = j * w_stride
            h_end = min((i + 1) * h_stride, H)
            w_end = min((j + 1) * w_stride, W)
            
            # 获取当前窗口
            window = mask[:, h_start:h_end, w_start:w_end]
            
            # 如果窗口中有非0值，则优先选择非0值
            non_zero = window != 0
            if torch.any(non_zero):
                # 获取非0值中出现最多的类别
                non_zero_values = window[non_zero]
                values, counts = torch.unique(non_zero_values, return_counts=True)
                output[:, i, j] = values[counts.argmax()]
            else:
                # 如果没有非0值，则使用众数
                values, counts = torch.unique(window, return_counts=True)
                output[:, i, j] = values[counts.argmax()]
    
    return output.long()

# 使用方法

# def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction='mean'):
#     # logits: [B, C, H, W]，模型的原始输出
#     # targets: [B, H, W]，目标标签，包含每个像素的类别索引

#     # 使用 F.cross_entropy 计算原始交叉熵损失
#     ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]
    
#     # 计算每个像素的预测概率
#     probs = F.softmax(logits, dim=1)  # [B, C, H, W]
#     # 获取每个像素正确类别的概率
#     pt = probs.gather(1, targets.unsqueeze(1))  # [B, 1, H, W]
#     pt = pt.squeeze(1)  # [B, H, W]

#     # 计算 Focal Loss
#     focal_loss = alpha * (1 - pt) ** gamma * ce_loss  # 每个像素的 focal loss

#     # 返回平均损失
#     if reduction == 'mean':
#         return focal_loss.mean()  # 返回整个 batch 的平均损失
#     elif reduction == 'sum':
#         return focal_loss.sum()  # 返回整个 batch 的总损失
#     else:
#         return focal_loss  # 返回逐像素的损失

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    # logits: [B, C, H, W]，模型的原始输出
    # targets: [B, H, W]，目标标签，包含每个像素的类别索引，其中-1表示忽略的像素
    
    # 创建mask来标识需要计算loss的像素
    valid_mask = (targets != -1)
    
    # 只对有效像素计算交叉熵损失
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=-1)  # [B, H, W]
    
    # 计算每个像素的预测概率
    probs = F.softmax(logits, dim=1)  # [B, C, H, W]
    # 获取每个有效像素正确类别的概率
    pt = probs.gather(1, targets.clamp(min=0).unsqueeze(1))  # [B, 1, H, W]
    pt = pt.squeeze(1)  # [B, H, W]
    
    # 计算focal loss，只对有效像素计算
    focal_weight = (1 - pt) ** gamma
    focal_loss = alpha * focal_weight * ce_loss
    
    # 使用valid_mask确保只计算有效像素的loss
    focal_loss = focal_loss * valid_mask.float()
    
    # 返回平均损失
    if reduction == 'mean':
        # 计算有效像素的平均loss
        num_valid_pixels = valid_mask.sum().item()
        return focal_loss.sum() / (num_valid_pixels + 1e-6)  # 添加小值避免除零
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

class TimesformerNaiveLocalizationHead(nn.Module):
    """
    Naive Localization Head, using pooler output to dot-product with label embeddings 
    """
    def __init__(self, config, label2id):
        super().__init__()
        self.config = config
        self.label2id = label2id
        # self.conv = Conv(config.hidden_size, config.hidden_size, config.hidden_size, num_layers=2, kernel_size=3)
        
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        # initialize the necessaryweights
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        
        input_ids = self._tokenize_labels()
        input_ids = input_ids.to(self.text_encoder.device) 
        outputs = self.text_encoder(**input_ids)
        self.label_embeddings = outputs[1]
    
    def _tokenize_labels(self):
        # labels = self.config.label2id.keys()
        assert self.label2id is not None, "Please provide label2id dict"
        labels = self.label2id.keys()
        prompt_template = "A photo of a {} person."
        input_ids = self.text_tokenizer([prompt_template.format(label) for label in labels], return_tensors="pt", padding="max_length")
        self.input_ids = input_ids # for debug only
        return input_ids
    
    def extract_feature(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None):
        window_size = 384
        sequence_output = task_head_input
        sequence_output = sequence_output.reshape(-1, window_size, sequence_output.shape[2]) # [B, TxW, D]
        # sequence_output = self.conv(sequence_output)
        # sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)
        return sequence_output
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        window_size = task_specific_input["masks"][0].shape[0]
        
        # if self.training:
        sequence_output = task_head_input.pooler_output # [BxW, T, D], WxT is the window size
        sequence_output = sequence_output.reshape(-1, window_size, sequence_output.shape[2]) # [B, TxW, D]
        # sequence_output = self.conv(sequence_output)
        # else:
        #     sequence_output = task_head_input
        #     sequence_output = self.conv(sequence_output)
        
        text_embeds = self.label_embeddings.detach()  # [L, D]
        
        image_embeds = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        similarity_per_frame = torch.einsum('btd,ld->btl', image_embeds, text_embeds) # [B, TxW, L]
        logits = similarity_per_frame * self.logit_scale.exp() + self.logit_bias # [B, TxW, L]
        if not self.training:
            return logits
        target_list = []
        for i in range(logits.shape[0]):
            gt_labels = task_specific_input["gt_labels"][i].to(logits.device, non_blocking=True) #[N], each element is the label index
            gt_segments = task_specific_input["gt_segments"][i].to(logits.device, non_blocking=True) #[N, 2]
            target_labels = - torch.ones_like(logits[i]) # [TxW, L]
            # TODO check masks here
            if task_specific_input["masks"] is not None: 
                gt_masks = task_specific_input["masks"][i].to(logits.device) #[TxW] 
                target_labels[gt_masks != True] = 0 # set the non-action frames to 0 
            for label_idx, label in enumerate(gt_labels):
                # get the start and end frame index in the window
                start_idx = int(gt_segments[label_idx][0]) if gt_segments[label_idx][0] == int(gt_segments[label_idx][0]) else int(gt_segments[label_idx][0]) + 1
                end_idx = int(gt_segments[label_idx][1]) + 1
                target_labels[start_idx:end_idx, label] = 1
            target_list.append(target_labels)
        target_labels = torch.stack(target_list) # [B, TxW, L]
        
        # ### debug ###
        # print(target_labels.shape, logits.shape)
        # labels = target_labels[0].argmax(-1)
        # preds = logits[0].argmax(-1)
        # print('label:', labels)
        # print('pred:', preds)
        # inds = torch.where(labels != 0)
        # print('ind    s:', inds)
        # print('inds_logits:', preds[inds])
        # print('inds_target:', labels[inds])
        # breakpoint()

        loss = - F.logsigmoid(target_labels * logits).sum() / target_labels.shape[0] / target_labels.shape[1]
        # make labels and logits to 2D
        # target_labels = target_labels.view(-1, target_labels.shape[2]) # [BxTxW, L]
        # logits = logits.view(-1, logits.shape[2]) # [BxTxW, L]
        
        # loss = - F.logsigmoid(target_labels * logits).sum() / image_embeds.shape[0]
        return loss, logits

class TimesformerUniversalLocalizationHead(nn.Module):
    """
    Naive Localization Head, using pooler output to dot-product with label embeddings 
    """
    def __init__(self, config, label2id):
        super().__init__()
        self.config = config
        self.label2id = label2id
        self.video_templates = VIDEO_TEMPLATES
        # self.conv = Conv(config.hidden_size, config.hidden_size, config.hidden_size, num_layers=2, kernel_size=3)
        
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        # initialize the necessaryweights
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        
        # 为每个数据集准备label embeddings
        self.dataset_label_embeddings = {}
        for dataset_name, dataset_label2id in self.label2id.items():
            input_ids = self._tokenize_labels(dataset_label2id)
            # input_ids = input_ids.to(self.text_encoder.device)
            label_embeddings = []
            for label_input_ids in input_ids:
                label_input_ids = label_input_ids.to(self.text_encoder.device)
                outputs = self.text_encoder(label_input_ids)[1]
                outputs /= outputs.norm(p=2, dim=-1, keepdim=True)
                outputs = outputs.mean(dim=0)
                label_embeddings.append(outputs)
            self.dataset_label_embeddings[dataset_name] = torch.stack(label_embeddings, dim=0)
        
    
    def _tokenize_labels(self, label2id):
        labels = label2id.keys()
        
        input_ids = []
        for label in labels:
            texts = [template.format(label) for template in self.video_templates]
            encoded_texts = self.text_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=64)['input_ids']
            input_ids.append(encoded_texts)
        
        return input_ids
    
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:        
        sequence_output = task_head_input.pooler_output # [B, T, D]
        image_embeds = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)
        
        # 获取每个batch对应的数据集
        datasets = task_specific_input['dataset']  # [B]
        gt_labels = task_specific_input['label'].long()  # [B, T]
        
        total_loss = 0
        all_logits = []
        
        # print('Datasets:', datasets)
        # print('gtlabels:', gt_labels)
        # breakpoint()

        # 对每个batch分别处理
        for i in range(len(datasets)):
            dataset_name = datasets[i]
            text_embeds = self.dataset_label_embeddings[dataset_name].detach()  # [L, D]
            # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 计算当前batch的logits
            similarity = torch.einsum('td,ld->tl', image_embeds[i], text_embeds)  # [T, L]
            logits = similarity * self.logit_scale.exp() + self.logit_bias  # [T, L]
            all_logits.append(logits)
            
            if not self.training:
                continue
                
            # 准备target labels
            num_classes = len(self.label2id[dataset_name])
            target_labels = -torch.ones(logits.shape[0], num_classes).to(logits.device)  # [T, L]
            
            # breakpoint()
            # 设置前景标签为1
            frame_labels = gt_labels[i] # [T]
            foreground_mask = frame_labels >= 0  # [T]
            target_labels[torch.arange(len(frame_labels))[foreground_mask], frame_labels[foreground_mask]] = 1
            
            # breakpoint()
            # 计算loss
            batch_loss = -F.logsigmoid(target_labels * logits).sum() / logits.shape[0]
            total_loss += batch_loss
            
        if not self.training:
            return all_logits
            
        avg_loss = total_loss / len(datasets)
        return avg_loss, all_logits
            
                
class TimesformerGroundingLikeLocalizationHead(nn.Module):
    """
    Naive Localization Head, using pooler output to dot-product with label embeddings 
    """
    def __init__(self, config, label2id):
        super().__init__()
        self.config = config
        self.label2id = label2id
        # self.conv = Conv(config.hidden_size, config.hidden_size, config.hidden_size, num_layers=2, kernel_size=3)
        
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        # initialize the necessaryweights
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        
        input_ids = self._tokenize_labels()
        input_ids = input_ids.to(self.text_encoder.device) 
        outputs = self.text_encoder(**input_ids)
        self.label_embeddings = outputs[1]
    
    def _tokenize_labels(self):
        # labels = self.config.label2id.keys()
        assert self.label2id is not None, "Please provide label2id dict"
        prompt_template = "A video of action {}."
        input_ids = self.text_tokenizer([prompt_template.format(label) if label_value > 0 else "No action in the video" for label, label_value in self.label2id.items()], return_tensors="pt", padding="max_length")
        self.input_ids = input_ids # for debug only
        return input_ids
    
    def extract_feature(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None):
        window_size = 16
        sequence_output = task_head_input
        sequence_output = sequence_output.reshape(-1, window_size, sequence_output.shape[2]) # [B, TxW, D]
        # sequence_output = self.conv(sequence_output)
        # sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)
        return sequence_output
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        window_size = task_specific_input["masks"][0].shape[0]
        
        # if self.training:
        sequence_output = task_head_input.pooler_output # [BxW, T, D], WxT is the window size
        sequence_output = sequence_output.reshape(-1, window_size, sequence_output.shape[2]) # [B, TxW, D]
        # sequence_output = self.conv(sequence_output)
        # else:
        #     sequence_output = task_head_input
        #     sequence_output = self.conv(sequence_output)
        
        text_embeds = self.label_embeddings.detach()  # [L, D]
        
        image_embeds = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        similarity_per_frame = torch.einsum('btd,ld->btl', image_embeds, text_embeds) # [B, TxW, L]
        logits = similarity_per_frame * self.logit_scale.exp() + self.logit_bias # [B, TxW, L]
        
        labels = task_specific_input["gt_labels"].to(logits.device)
        
        
        
        ################ try softmax loss for now ###############
        # reshaped_logits = rearrange(logits, 'b t d -> (b t) d')
        # reshaped_labels = rearrange(labels, 'b t -> (b t)')
        ### debug ###
        # print(reshaped_logits.shape, reshaped_labels.shape)
        # preds = torch.argmax(reshaped_logits, dim=-1)
        # preds = rearrange(preds, '(b t) -> b t', b=logits.shape[0])
        # for i in range(preds.shape[0]):
        #     print('video:', i)
        #     print('pred:', preds[i])
        #     print('label:', labels[i])
        #     print()
        # breakpoint()
        # if not self.training:
        #     return logits
        # loss = F.cross_entropy(reshaped_logits, reshaped_labels, ignore_index=0)
        # return loss, logits
        ############################    
        
        if not self.training:
            return logits
        target_list = []
        for i in range(logits.shape[0]):
            gt_labels = labels[i]
            target_labels = - torch.ones_like(logits[i]) # [TxW, L]
            # TODO check masks here
            # if task_specific_input["masks"] is not None: 
            #     gt_masks = task_specific_input["masks"][i].to(logits.device) #[TxW] 
            #     target_labels[gt_masks != True] = 0 # set the non-action frames to 0 
            unique_labels = torch.unique(gt_labels)
            for label in unique_labels:
                if label == 0: ## this is background
                    continue
                
                indices = torch.where(gt_labels == label)[0]
                target_labels[indices, label] = 1

            target_list.append(target_labels)

        #     print(i)
        #     print('Target:', target_labels)
        #     print('Logits:', logits[i])
        # breakpoint()
        target_labels = torch.stack(target_list) # [B, TxW, L]
        
        
        loss = - F.logsigmoid(target_labels * logits).sum() / target_labels.shape[0] / target_labels.shape[1]
        return loss, logits     

# from .actionformer.meta_archs import PtTransformer
class TimesformerTemporalActionLocalizationHead(nn.Module):
    """Temporal Action Localization Head, currently using ActionFormer as the head
    """
    def __init__(self, config, actionformer_config):
        super().__init__()
        self.config = config
        self.actionformer = PtTransformer(
            **actionformer_config
        )
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        pass
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict]) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        if self.training:
            windows_size = task_specific_input["masks"][0].shape[0]
            sequence_output = task_head_input.pooler_output # (batch_size, num_frames, hidden_size)
            sequence_output = sequence_output.reshape(-1, windows_size, sequence_output.shape[2]) # (batch_size, num_frames, hidden_size)
            sequence_output = sequence_output.permute(0, 2, 1) # (batch_size, hidden_size, num_frames)
            video_list = []
            for i in range(sequence_output.shape[0]):
                video_list.append({
                    "video_id": task_specific_input["video_id"][i],
                    "feats": sequence_output[i],
                    "masks":  task_specific_input["masks"][i],
                    "segments": torch.tensor(task_specific_input["gt_segments"][i]),
                    "labels": torch.tensor(task_specific_input["gt_labels"][i], dtype=torch.long),
                    "fps": task_specific_input["fps"][i],
                    "duration": task_specific_input["duration"][i],
                    "feat_stride": task_specific_input["feat_stride"][i],
                    "feat_num_frames" :task_specific_input["feat_num_frames"][i],
                })
        else:
            # sequence_output = task_head_input.
            windows_size = task_specific_input["masks"][0].shape[0]
            sequence_output = task_head_input.pooler_output
            sequence_output = sequence_output.reshape(-1, windows_size, sequence_output.shape[2])
            sequence_output = sequence_output.permute(0, 2, 1)
            
            video_list = []
            for i in range(sequence_output.shape[0]):
                video_list.append({
                    "video_id": task_specific_input["video_id"][i],
                    "feats": sequence_output[i],
                    "segments": torch.tensor(task_specific_input["gt_segments"][i]),
                    "masks":  task_specific_input["masks"][i],
                    "labels": torch.tensor(task_specific_input["gt_labels"][i], dtype=torch.long),
                    "fps": task_specific_input["fps"][i],
                    "duration": task_specific_input["duration"][i],
                    "feat_stride": task_specific_input["feat_stride"][i],
                    "feat_num_frames" :task_specific_input["feat_num_frames"][i],
                })
        # prepare video_list, with listed feats, segments, labels
        
        results = self.actionformer(video_list)
        if self.training:
            return results['final_loss'], results
        else:
            return results

class TimesformerVideoRetrievalHead(nn.Module):
    """Video Retrieval Head
    Using SigLIP text encoder for video-text retrieval
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_function = SigLipLoss(rank=dist.get_rank(), world_size=dist.get_world_size())

    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        # self.text_encoder = copy.deepcopy(text_encoder)
        # self.text_tokenizer = copy.deepcopy(text_tokenizer)

    def encode_captions(self, captions, device='cuda'):
        inputs = self.text_tokenizer(captions, return_tensors='pt', padding='max_length', max_length=64, truncation=True).to(device)
        text_features = self.text_encoder(**inputs)[1]
        return text_features

    def get_logits(self, image_features, text_features):
        logits = self.logit_scale.exp() * image_features @ text_features.T + self.logit_bias
        return logits

    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        # image_features = torch.mean(task_head_input.pooler_output, 1, True).squeeze(1)  # mean pooling to reduce the frame dimension
        image_features = task_head_input.pooler_output[:, -1,:].squeeze(1) # use last frame feature 
        
        # Encode captions
        text_features = self.encode_captions(task_specific_input['caption'], device=image_features.device)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        if not self.training:
            return image_features, text_features # eval mode
        # Compute loss
        loss = self.loss_function(image_features, text_features, self.logit_scale.exp(), self.logit_bias)
        
        # Compute logits for debugging or additional processing if needed
        logits = torch.matmul(image_features, text_features.t()) * self.logit_scale.exp()
        return loss, logits

class TimesformerTemporalGroundingContrastiveHead(nn.Module):
    """Temporal Grounding Head
    MultiTaskingSigLIP + This Head = TimesformerForTemporalGroundingSigLIP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.conv = Conv(config.hidden_size, config.hidden_size, config.hidden_size, num_layers=2, kernel_size=3)
        self.conv = nn.Identity()        
        # self.loss_function = SigLipLoss(rank=dist.get_rank(), world_size=dist.get_world_size())
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        image_embeds = task_head_input.pooler_output  # [b,t,d]
        image_embeds = self.conv(image_embeds)
        
        input_ids = self.text_tokenizer(task_specific_input['caption'], return_tensors="pt", padding="max_length", truncation=True).to(image_embeds.device)
        outputs = self.text_encoder(**input_ids)
        text_embeds = outputs[1] # (batch_size, hidden_size)
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)  # [b,t,d]
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)     # [b,d]

        # Reshape image_embeds to [bt,d]
        b, t, d = image_embeds.shape
        device = image_embeds.device
        image_embeds = image_embeds.reshape(-1, d)  # [bt,d]
        
        # 使用all_gather收集所有GPU上的特征
        all_image_embeds = [torch.zeros_like(image_embeds) for _ in range(self.world_size)]
        all_text_embeds = [torch.zeros_like(text_embeds) for _ in range(self.world_size)]
        targets = task_specific_input['label'].to(device)
        targets[targets == 0] = -1 ### to match the label format of SigLIP
        all_targets = [torch.zeros_like(targets) for _ in range(self.world_size)]
        
        dist.all_gather(all_image_embeds, image_embeds)
        dist.all_gather(all_text_embeds, text_embeds)
        dist.all_gather(all_targets, targets)
        
        # 合并所有GPU的特征
        all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [world_size*bt,d]
        all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [world_size*b,d]
        all_targets = torch.cat(all_targets, dim=0)            # [world_size*b,t]
        # 计算所有GPU上的样本之间的相似度
        similarity = torch.matmul(all_image_embeds, all_text_embeds.t())  # [world_size*bt,world_size*b]
        logits = similarity * self.logit_scale.exp() + self.logit_bias
        
        # print('logits shape:', logits.shape)
        # print('all_image_embeds shape:', all_image_embeds.shape)
        # print('all_text_embeds shape:', all_text_embeds.shape)
        # print('all_targets shape:', all_targets.shape)
        # breakpoint()
        # 创建标签矩阵
        total_b = b * self.world_size
        all_targets = all_targets.reshape(-1, 1)  # [world_size*bt,1]
        labels = torch.ones_like(logits) * -1     # [world_size*bt,world_size*b]
        
        for i in range(total_b):
            start_idx = i * t
            end_idx = (i + 1) * t
            labels[start_idx:end_idx, i] = all_targets[start_idx:end_idx].squeeze()
            
        # print('labels shape:', labels.shape)
        # breakpoint()
        labels = labels.to(logits.device)
        loss = - F.logsigmoid(labels * logits).sum() / (total_b * t)
        return loss, logits



class TimesformerTemporalGroundingHead(nn.Module):
    """Temporal Grounding Head
    MultiTaskingSigLIP + This Head = TimesformerForTemporalGroundingSigLIP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.conv = Conv(config.hidden_size, config.hidden_size, config.hidden_size, num_layers=2, kernel_size=3)
        self.conv = nn.Identity()        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        image_embeds = task_head_input.pooler_output
        image_embeds = self.conv(image_embeds)
        
        #input_ids = self.text_tokenizer(task_specific_input['caption'], return_tensors="pt", padding="max_length").to(image_embeds.device)
        input_ids = self.text_tokenizer(task_specific_input['caption'], return_tensors="pt", padding="max_length", truncation=True).to(image_embeds.device)
        outputs = self.text_encoder(**input_ids)
        text_embeds = outputs[1] # (batch_size, hidden_size)
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.einsum('btd,bd->bt', image_embeds, text_embeds) 
        logits = similarity * self.logit_scale.exp() + self.logit_bias
        
        targets = task_specific_input['label']
        labels = targets.masked_fill(targets == 0, -1).to(logits.device)
        loss = - F.logsigmoid(labels * logits).sum() / logits.shape[0]
        return loss, logits

# from .actionformer_head import NLQHead, PositionalEncoding
class TimesformerActionFormerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_encoder = PositionalEncoding(d_model=config.hidden_size)
        self.fusion_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=16, batch_first=True, norm_first=True)
        self.nlq_head = NLQHead(config.hidden_size, config.num_frames)
        self.fusion_layer.apply(self.init_weights)
        self.nlq_head.apply(self.init_weights)
        
    def prepare_multi_task(self, text_encoder, text_tokenizer, logit_scale, logit_bias, vision_model):
        self.logit_scale = copy.deepcopy(logit_scale)
        self.logit_bias = copy.deepcopy(logit_bias)
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder

    def init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, val=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, val=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, task_head_input: ModelOutput, task_specific_input: Optional[dict] = None) -> torch.Tensor:
        image_embeds = task_head_input.pooler_output
        
        input_ids = self.text_tokenizer(task_specific_input['caption'], return_tensors="pt", padding="max_length").to(image_embeds.device)
        outputs = self.text_encoder(**input_ids)
        text_embeds = outputs[1] # (batch_size, hidden_size)

        tokens = torch.cat([text_embeds.unsqueeze(1), image_embeds], dim=1)
        tokens = self.pos_encoder(tokens)
        tokens = self.fusion_layer(tokens)
        image_embeds = tokens[:, 1:, :]

        nlq_results = self.nlq_head(
            feat=image_embeds.permute(0, 2, 1),  # (B, D, T)
            mask=torch.ones(image_embeds.shape[0], self.config.num_frames).unsqueeze(1).bool().to(image_embeds.device),  # (B, 1, T)
            gt_segments=task_specific_input['gt_segments'].to(image_embeds.device),
            gt_labels=torch.ones(image_embeds.shape[0], 1, 1, dtype=torch.int64).to(image_embeds.device),
            training=task_specific_input['training'],
            v_lens=[self.config.num_frames] * image_embeds.shape[0]
        )

        loss = None
        logits = None

        if task_specific_input['training']:
            loss = nlq_results['final_loss'] * 1.0
        else:
            logits = nlq_results


        return loss, logits


    
def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv

def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left

class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)

class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)

class SigLipLoss(nn.Module):

    def __init__(self, cache_labels=False, rank=0, world_size=1, bidir=True, use_horovod=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank 
        self.world_size = world_size 
        assert not use_horovod
        self.use_horovod = use_horovod
        self.bidir = bidir 

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False):
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels 
        return labels 

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T 
        if logit_bias is not None:
            logits += logit_bias 
        return logits 

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(image_features.device, image_features.dtype, image_features.shape[0], negative_only=negative_only)
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss  

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world size - 1 times
            right_rank = (self.rank + 1) % self.world_size 
            left_rank = (self.rank - 1 + self.world_size) % self.world_size 
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features 
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank, 
                        right_rank,
                        text_features_to_left,
                        text_features_to_right
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv
                
                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return loss 
if __name__ == '__main__':
    import numpy as np
    import json
    import argparse
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, frame_idx, use_cache=True):
            super().__init__()
            self.model = model
            self.frame_idx = frame_idx
            self.use_cache = use_cache
            
        def forward(self, x):
            return self.model(x, use_cache=self.use_cache, frame_idx=self.frame_idx)
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prefill_frames', type=int, default=16, help='Number of frames to prefill in KV cache')
    # parser.add_argument('--total_frames', type=int, default=16, help='Total number of frames')
    # args = parser.parse_args()
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 16
    img_size = 224
    
    # debug for now
    model = TimesformerMultiTaskingModelSigLIP.from_pretrained("/raid/yibinyan/LLaVA-NeXT/checkpoints/timesformer-temporal-video").to("cuda:0")
    model.eval()
    import time
    
    # Calculate FLOPs for different scenarios
    if False:
        print("\n=== FLOPs Analysis ===")
        
        # 1. Measure FLOPs for first frame (no kv-cache)
        with torch.no_grad():
            first_frame = torch.rand(1, 1, 3, img_size, img_size).to("cuda:0")
            wrapped_model = ModelWrapper(model, frame_idx=0, use_cache=False)
            flops_first = FlopCountAnalysis(wrapped_model, first_frame)
            
            # print(flop_count_table(flops_first))
        
        # 2. Prefill KV cache with specified number of frames
        prefill_frames = torch.rand(1, args.prefill_frames, 3, img_size, img_size).to("cuda:0")
        model.reset_cache()
        for i in range(args.prefill_frames):
            with torch.no_grad():
                model(prefill_frames[:, i:i+1, :, :, :], use_cache=True, frame_idx=i)
        
        # 3. Measure FLOPs for next frame with kv-cache
        with torch.no_grad():
            next_frame = torch.rand(1, 1, 3, img_size, img_size).to("cuda:0")
            wrapped_model = ModelWrapper(model, frame_idx=args.prefill_frames, use_cache=True)
            flops_cached = FlopCountAnalysis(wrapped_model, next_frame)
            
            # print(flop_count_table(flops_cached))
            time.sleep(1)
            print("\nFirst frame (no kv-cache):")
            print(f"Total FLOPs: {flops_first.total()/1e9:.2f}B")
            print(f"\nNext frame with {args.prefill_frames} frames in kv-cache:")
            print(f"Total FLOPs: {flops_cached.total()/1e9:.2f}B")
        print("\n=====================")
        
        # 4. Measure FLOPS with no KV-cache
        with torch.no_grad():
            all_frames = torch.rand(1, args.prefill_frames, 3, img_size, img_size).to("cuda:0")
            flops_all = FlopCountAnalysis(model, all_frames)
            print("\nAll frames (no kv-cache):")
            print(f"Total FLOPs: {flops_all.total()/1e9:.2f}B")
            print(flop_count_table(flops_all))
    total_time = 0
    total_latency = 0
    num_tests = 2
    fps = 1000
    frame_interval = 1.0 / fps  # 0.25s
    max_memory = 0
    num_frames = 16
    
    import pynvml
    pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    
    
    import threading
    class GPUMemoryMonitor:
        def __init__(self, gpu_id=0, delay=0.0001):
            self.gpu_id = gpu_id
            self.delay = delay
            self.running = False
            self.max_memory = 0
            pynvml.nvmlInit()
            
        def monitor(self):
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            while self.running:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024*1024)  # MB
                self.max_memory = max(self.max_memory, memory)
                time.sleep(self.delay)
                
        def start(self):
            self.running = True
            self.thread = threading.Thread(target=self.monitor)
            self.thread.start()
            
        def stop(self):
            self.running = False
            self.thread.join()
            print(f"Peak GPU memory usage: {self.max_memory:.2f} MB")

    # 在每一个要查看的地方都要重新定义一个meminfo 
    # memory_before = torch.cuda.memory_allocated("cuda:0") / (1024 * 1024)
    for test_num in range(1):
        start_time = time.time()
        # model.reset_cache()
        fake_frames = torch.rand(1, num_frames, 3, img_size, img_size)
        
        next_frame_time = start_time
        monitor = GPUMemoryMonitor(gpu_id=0)
        monitor.start()
        
        for i in range(num_frames):
            # 等待到下一帧的时间点
            current_time = time.time()
            if current_time < next_frame_time:
                time.sleep(next_frame_time - current_time)
                
            frame_start = time.time()
            torch.cuda.synchronize()
              # 转换为MB
            # input_frames = fake_frames[:, i:i+1, :, :, :].to("cuda:0")  # causal temporal attention
            input_frames = fake_frames[:, :i+1, :, :, :].to("cuda:0")  # bi-directional
            with torch.no_grad():
                # outputs = model(input_frames, use_cache=True, frame_idx=i)
                outputs = model(input_frames)
                torch.cuda.synchronize()  # 确保GPU计算完成
                # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # memory_after = meminfo.used / (1024 * 1024)  # 转换为MB
                # memory_usage = memory_after
                del outputs, input_frames
                torch.cuda.empty_cache()
                # max_memory = max(max_memory, memory_usage)
                
                latency = time.time() - frame_start
                total_latency += latency
                print(f"Frame {i} latency: {latency:.4f} seconds")
            
            next_frame_time += frame_interval  # 更新下一帧的目标时间
        monitor.stop()
        end_time = time.time()
        time_per_test = end_time - start_time
        total_time += time_per_test
        print(f"Test {test_num + 1} total time: {time_per_test:.4f} seconds")
        print(f"Max memory usage: {max_memory:.2f} MB")
        print(f"Average time per test: {total_time / num_tests:.4f} seconds")
        print(f"Average per-frame latency: {total_latency / (num_tests * num_frames):.4f} seconds")
        print("--------------------------------")
