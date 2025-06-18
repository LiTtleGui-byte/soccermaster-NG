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