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

class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = StreamformerConfig
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