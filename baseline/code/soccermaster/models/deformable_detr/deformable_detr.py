# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from typing import Optional

from utils import box_ops
from utils.nested_tensor import NestedTensor, nested_tensor_from_tensor_list, nested_tensor_from_tensor_list_during_training
from models.utils.misc import inverse_sigmoid, accuracy, interpolate
from utils.misc import is_distributed, distributed_world_size
from typing import Any, Dict, List, Tuple, Union, Generator
from collections import defaultdict
import copy

from models.deformable_detr.position_encoding import build_position_encoding
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from data.soccernet_gsr_reid import role_mapping, jn_mapping, digit_head_mapping, digit_tail_mapping
from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDetrHead(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, position_encoding, transformer, num_classes, num_queries, num_feature_levels, backbone_strides, backbone_num_channels,
                 aux_loss=True, with_box_refine=False, two_stage=False, detection_data_type = "image", backbone_type='image',
                 enable_role_classification=True, enable_jn_classification=True, local_features_type='none'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        # TODO: find a way to handle positional encoding, strides, channels, etc.
        super().__init__()
        self.position_encoding = position_encoding
        self.num_queries = num_queries
        self.transformer = transformer
        self.enable_role_classification = enable_role_classification
        self.enable_jn_classification = enable_jn_classification
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Create role classification head only if enabled
        if self.enable_role_classification:
            num_role_classes = len(role_mapping)
            self.role_embed = nn.Linear(hidden_dim, num_role_classes)
        else:
            self.role_embed = None
            
        # Create jersey number classification heads only if enabled
        if self.enable_jn_classification:
            num_jn_classes = len(jn_mapping)
            num_digit_head_classes = len(digit_head_mapping)
            num_digit_tail_classes = len(digit_tail_mapping)
            self.jn_holistic_embed = nn.Linear(hidden_dim, num_jn_classes)
            self.digit_head_embed = nn.Linear(hidden_dim, num_digit_head_classes)
            self.digit_tail_embed = nn.Linear(hidden_dim, num_digit_tail_classes)
        else:
            self.jn_holistic_embed = None
            self.digit_head_embed = None
            self.digit_tail_embed = None
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.detection_data_type = detection_data_type
        self.backbone_type = backbone_type
        self.local_features_type = local_features_type

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        if self.enable_role_classification:
            num_role_classes = len(role_mapping)
            self.role_embed.bias.data = torch.ones(num_role_classes) * bias_value
            
        if self.enable_jn_classification:
            num_jn_classes = len(jn_mapping)
            num_digit_head_classes = len(digit_head_mapping)
            num_digit_tail_classes = len(digit_tail_mapping)
            self.jn_holistic_embed.bias.data = torch.ones(num_jn_classes) * bias_value
            self.digit_head_embed.bias.data = torch.ones(num_digit_head_classes) * bias_value
            self.digit_tail_embed.bias.data = torch.ones(num_digit_tail_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if self.enable_role_classification:
                self.role_embed = _get_clones(self.role_embed, num_pred)
            if self.enable_jn_classification:
                self.jn_holistic_embed = _get_clones(self.jn_holistic_embed, num_pred)
                self.digit_head_embed = _get_clones(self.digit_head_embed, num_pred)
                self.digit_tail_embed = _get_clones(self.digit_tail_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.enable_role_classification:
                self.role_embed = nn.ModuleList([self.role_embed for _ in range(num_pred)])
            if self.enable_jn_classification:
                self.jn_holistic_embed = nn.ModuleList([self.jn_holistic_embed for _ in range(num_pred)])
                self.digit_head_embed = nn.ModuleList([self.digit_head_embed for _ in range(num_pred)])
                self.digit_tail_embed = nn.ModuleList([self.digit_tail_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        global_features = backbone_outputs['global_features']
        
        # Select local features based on configuration
        if self.local_features_type == 'none':
            local_features = backbone_outputs['local_features']
        elif self.local_features_type == 'early':
            local_features = backbone_outputs['local_features_early']
        elif self.local_features_type == 'late':
            local_features = backbone_outputs['local_features_late']
        elif self.local_features_type == 'both':
            # Concatenate early and late features along the channel dimension
            local_features_early = backbone_outputs['local_features_early']
            local_features_late = backbone_outputs['local_features_late']
            if self.backbone_type == 'video':
                # [B, T, N, D] -> concatenate on D dimension
                local_features = torch.cat([local_features_early, local_features_late], dim=-1)
            else:
                # [B, N, D] -> concatenate on D dimension
                local_features = torch.cat([local_features_early, local_features_late], dim=-1)
        else:
            # Fallback to default behavior for backward compatibility
            local_features = backbone_outputs.get('local_features', backbone_outputs['local_features_late'])
        # if self.backbone_type == 'video' and self.detection_data_type == 'image':
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = local_features.shape
            local_features = local_features.reshape(bs * num_frames, *local_features.shape[2:])
        else:
            bs, _, _ = local_features.shape
        
        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        features = reshaped_local_features
        
        features = [nested_tensor_from_tensor_list_during_training(features)]

        pos = []
        for x in features:
            pos.append(self.position_encoding(x).to(x.tensors.dtype))
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = features[0].mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_encoding(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_roles = [] if self.enable_role_classification else None
        outputs_jn_holistic = [] if self.enable_jn_classification else None
        outputs_digit_head = [] if self.enable_jn_classification else None
        outputs_digit_tail = [] if self.enable_jn_classification else None
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            
            # Only compute role outputs if enabled
            if self.enable_role_classification:
                outputs_role = self.role_embed[lvl](hs[lvl])
            
            # Only compute jersey number outputs if enabled
            if self.enable_jn_classification:
                outputs_jn = self.jn_holistic_embed[lvl](hs[lvl])
                outputs_digit_h = self.digit_head_embed[lvl](hs[lvl])
                outputs_digit_t = self.digit_tail_embed[lvl](hs[lvl])
            
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            
            if self.backbone_type == 'video':
                outputs_class = outputs_class.reshape(bs, num_frames, *outputs_class.shape[1:])
                outputs_coord = outputs_coord.reshape(bs, num_frames, *outputs_coord.shape[1:])
                if self.enable_role_classification:
                    outputs_role = outputs_role.reshape(bs, num_frames, *outputs_role.shape[1:])
                if self.enable_jn_classification:
                    outputs_jn = outputs_jn.reshape(bs, num_frames, *outputs_jn.shape[1:])
                    outputs_digit_h = outputs_digit_h.reshape(bs, num_frames, *outputs_digit_h.shape[1:])
                    outputs_digit_t = outputs_digit_t.reshape(bs, num_frames, *outputs_digit_t.shape[1:])
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.enable_role_classification:
                outputs_roles.append(outputs_role)
            if self.enable_jn_classification:
                outputs_jn_holistic.append(outputs_jn)
                outputs_digit_head.append(outputs_digit_h)
                outputs_digit_tail.append(outputs_digit_t)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        # Only stack outputs if the features are enabled
        if self.enable_role_classification:
            outputs_role = torch.stack(outputs_roles)
        if self.enable_jn_classification:
            outputs_jn_holistic = torch.stack(outputs_jn_holistic)
            outputs_digit_head = torch.stack(outputs_digit_head)
            outputs_digit_tail = torch.stack(outputs_digit_tail)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.enable_role_classification:
            out['pred_roles'] = outputs_role[-1]
        if self.enable_jn_classification:
            out['pred_jn_holistic'] = outputs_jn_holistic[-1]
            out['pred_digit_head'] = outputs_digit_head[-1]
            out['pred_digit_tail'] = outputs_digit_tail[-1]
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, 
                outputs_role if self.enable_role_classification else None, 
                outputs_jn_holistic if self.enable_jn_classification else None, 
                outputs_digit_head if self.enable_jn_classification else None, 
                outputs_digit_tail if self.enable_jn_classification else None
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Output the outputs of last decoder layer.
        # We need these outputs to generate the embeddings for objects.
        if self.backbone_type == 'video':
            out["outputs"] = hs[-1].reshape(bs, num_frames, *hs[-1].shape[1:])
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_role, outputs_jn_holistic, outputs_digit_head, outputs_digit_tail):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_outputs = []
        for i in range(len(outputs_class) - 1):  # exclude the last layer
            aux_out = {
                'pred_logits': outputs_class[i], 
                'pred_boxes': outputs_coord[i]
            }
            if outputs_role is not None:
                aux_out['pred_roles'] = outputs_role[i]
            if outputs_jn_holistic is not None:
                aux_out['pred_jn_holistic'] = outputs_jn_holistic[i]
            if outputs_digit_head is not None:
                aux_out['pred_digit_head'] = outputs_digit_head[i]
            if outputs_digit_tail is not None:
                aux_out['pred_digit_tail'] = outputs_digit_tail[i]
            aux_outputs.append(aux_out)
        return aux_outputs

def softmax_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
    """
    Softmax focal loss for multi-class classification.
    
    Args:
        inputs: A float tensor of shape [N, C] where N is the number of samples and C is the number of classes
        targets: A long tensor of shape [N] containing class indices
        num_boxes: Number of boxes for normalization
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        Loss tensor
    """
    
    # Apply log softmax to get log probabilities
    log_probs = F.log_softmax(inputs, dim=-1)
    
    # Get the log probabilities for the correct classes
    log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # Get the probabilities for the correct classes
    p = torch.exp(log_p)
    
    # Compute focal weight: (1 - p)^gamma
    focal_weight = (1 - p) ** gamma
    
    # For softmax focal loss, alpha weight is simply alpha for all correct predictions
    alpha_weight = alpha
    
    # Compute focal loss
    focal_loss = -alpha_weight * focal_weight * log_p
    
    return focal_loss.sum() / num_boxes

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, detr_loss_batch_len=10, detection_data_type='image', backbone_type='image', enable_softmax_focal_loss=False, detect_ball=False, detect_ball_only=False, enable_role_classification=True, enable_jn_classification=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            enable_softmax_focal_loss: whether to use softmax focal loss instead of sigmoid focal loss for attributes
            detect_ball: whether ball detection is enabled
            detect_ball_only: whether to only detect ball (exclude person detection)
        """
        super().__init__()
        self.num_classes = num_classes
        self.enable_role_classification = enable_role_classification
        self.enable_jn_classification = enable_jn_classification
        
        if self.enable_role_classification:
            self.num_role_classes = len(role_mapping)
        if self.enable_jn_classification:
            self.num_jn_classes = len(jn_mapping)
            self.num_digit_head_classes = len(digit_head_mapping)
            self.num_digit_tail_classes = len(digit_tail_mapping)
            
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.detr_loss_batch_len = detr_loss_batch_len
        self.detection_data_type = detection_data_type
        self.backbone_type = backbone_type
        self.enable_softmax_focal_loss = enable_softmax_focal_loss
        self.detect_ball = detect_ball
        self.detect_ball_only = detect_ball_only
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_roles(self, outputs, targets, indices, num_boxes, log=True):
        """Role classification loss (NLL)
        targets dicts must contain the key "roles" containing a tensor of dim [nb_target_boxes]
        """
        # Skip if role classification is disabled
        if not self.enable_role_classification:
            device = next(iter(outputs.values())).device
            loss_role = torch.tensor(0.0, device=device, requires_grad=True)
            losses = {'loss_role': loss_role}
            if log:
                losses['role_error'] = torch.tensor(0.0, device=device)
            return losses
            
        assert 'pred_roles' in outputs
        src_logits = outputs['pred_roles']

        idx = self._get_src_permutation_idx(indices)
        target_roles_o = torch.cat([t["roles"][J] for t, (_, J) in zip(targets, indices)])
        
        # Role supervision logic based on detection mode
        if self.detect_ball_only:
            # In ball-only mode, there are no person objects, so skip role supervision entirely
            loss_role = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
            losses = {'loss_role': loss_role}
            if log:
                losses['role_error'] = torch.tensor(0.0, device=src_logits.device)
            return losses
        elif self.detect_ball:
            # In mixed mode, only supervise roles for person objects (category 0)
            target_labels_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            person_mask = (target_labels_o == 0).cpu()  # Only person objects
            if person_mask.sum() == 0:
                # No person objects to supervise
                loss_role = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
                losses = {'loss_role': loss_role}
                if log:
                    losses['role_error'] = torch.tensor(0.0, device=src_logits.device)
                return losses
            
            # Filter to only person objects
            idx_filtered = (idx[0][person_mask], idx[1][person_mask])
            target_roles_o = target_roles_o[person_mask]
        else:
            # In person-only mode, supervise all objects (they are all persons)
            idx_filtered = idx
        
        if self.enable_softmax_focal_loss:
            # Use softmax focal loss
            if len(target_roles_o) > 0:
                loss_role = softmax_focal_loss(src_logits[idx_filtered], target_roles_o, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            else:
                loss_role = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
        else:
            # Use sigmoid focal loss (original implementation)
            target_roles_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_roles_onehot[idx_filtered[0], idx_filtered[1], target_roles_o] = 1
            loss_role = sigmoid_focal_loss(src_logits, target_roles_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_role': loss_role}

        if log:
            losses['role_error'] = 100 - accuracy(src_logits[idx_filtered], target_roles_o)[0]
        return losses

    def loss_jn_holistic(self, outputs, targets, indices, num_boxes, log=True):
        """Jersey number holistic classification loss (NLL)
        targets dicts must contain the key "jn_holistic" containing a tensor of dim [nb_target_boxes]
        """
        # Skip if jersey number classification is disabled
        if not self.enable_jn_classification:
            device = next(iter(outputs.values())).device
            loss_jn_holistic = torch.tensor(0.0, device=device, requires_grad=True)
            losses = {'loss_jn_holistic': loss_jn_holistic}
            if log:
                losses['jn_holistic_error'] = torch.tensor(0.0, device=device)
            return losses
            
        assert 'pred_jn_holistic' in outputs
        src_logits = outputs['pred_jn_holistic']

        idx = self._get_src_permutation_idx(indices)
        target_jn_holistic_o = torch.cat([t["jersey"][J] for t, (_, J) in zip(targets, indices)])
        
        # Jersey number supervision logic based on detection mode
        if self.detect_ball_only:
            # In ball-only mode, there are no person objects, so skip jersey number supervision entirely
            loss_jn_holistic = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
            losses = {'loss_jn_holistic': loss_jn_holistic}
            if log:
                losses['jn_holistic_error'] = torch.tensor(0.0, device=src_logits.device)
            return losses
        elif self.detect_ball:
            # In mixed mode, only supervise jersey numbers for person objects (category 0)
            target_labels_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            person_mask = (target_labels_o == 0).cpu()  # Only person objects
            if person_mask.sum() == 0:
                # No person objects to supervise
                loss_jn_holistic = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
                losses = {'loss_jn_holistic': loss_jn_holistic}
                if log:
                    losses['jn_holistic_error'] = torch.tensor(0.0, device=src_logits.device)
                return losses
            
            # Filter to only person objects
            idx_filtered = (idx[0][person_mask], idx[1][person_mask])
            target_jn_holistic_o = target_jn_holistic_o[person_mask]
        else:
            # In person-only mode, supervise all objects (they are all persons)
            idx_filtered = idx
        
        if self.enable_softmax_focal_loss:
            # Use softmax focal loss
            if len(target_jn_holistic_o) > 0:
                loss_jn_holistic = softmax_focal_loss(src_logits[idx_filtered], target_jn_holistic_o, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            else:
                loss_jn_holistic = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
        else:
            # Use sigmoid focal loss (original implementation)
            target_jn_holistic_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_jn_holistic_onehot[idx_filtered[0], idx_filtered[1], target_jn_holistic_o] = 1
            loss_jn_holistic = sigmoid_focal_loss(src_logits, target_jn_holistic_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_jn_holistic': loss_jn_holistic}

        if log:
            losses['jn_holistic_error'] = 100 - accuracy(src_logits[idx_filtered], target_jn_holistic_o)[0]
        return losses

    def loss_digit_head(self, outputs, targets, indices, num_boxes, log=True):
        """Digit head classification loss (NLL)
        targets dicts must contain the key "digit_head" containing a tensor of dim [nb_target_boxes]
        """
        # Skip if jersey number classification is disabled
        if not self.enable_jn_classification:
            device = next(iter(outputs.values())).device
            loss_digit_head = torch.tensor(0.0, device=device, requires_grad=True)
            losses = {'loss_digit_head': loss_digit_head}
            if log:
                losses['digit_head_error'] = torch.tensor(0.0, device=device)
            return losses
            
        assert 'pred_digit_head' in outputs
        src_logits = outputs['pred_digit_head']

        idx = self._get_src_permutation_idx(indices)
        target_digit_head_o = torch.cat([t["digit_head"][J] for t, (_, J) in zip(targets, indices)])
        
        # Digit head supervision logic based on detection mode
        if self.detect_ball_only:
            # In ball-only mode, there are no person objects, so skip digit head supervision entirely
            loss_digit_head = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
            losses = {'loss_digit_head': loss_digit_head}
            if log:
                losses['digit_head_error'] = torch.tensor(0.0, device=src_logits.device)
            return losses
        elif self.detect_ball:
            # In mixed mode, only supervise digit head for person objects (category 0)
            target_labels_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            person_mask = (target_labels_o == 0).cpu()  # Only person objects
            if person_mask.sum() == 0:
                # No person objects to supervise
                loss_digit_head = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
                losses = {'loss_digit_head': loss_digit_head}
                if log:
                    losses['digit_head_error'] = torch.tensor(0.0, device=src_logits.device)
                return losses
            
            # Filter to only person objects
            idx_filtered = (idx[0][person_mask], idx[1][person_mask])
            target_digit_head_o = target_digit_head_o[person_mask]
        else:
            # In person-only mode, supervise all objects (they are all persons)
            idx_filtered = idx
        
        if self.enable_softmax_focal_loss:
            # Use softmax focal loss
            if len(target_digit_head_o) > 0:
                loss_digit_head = softmax_focal_loss(src_logits[idx_filtered], target_digit_head_o, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            else:
                loss_digit_head = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
        else:
            # Use sigmoid focal loss (original implementation)
            target_digit_head_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_digit_head_onehot[idx_filtered[0], idx_filtered[1], target_digit_head_o] = 1
            loss_digit_head = sigmoid_focal_loss(src_logits, target_digit_head_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_digit_head': loss_digit_head}

        if log:
            losses['digit_head_error'] = 100 - accuracy(src_logits[idx_filtered], target_digit_head_o)[0]
        return losses

    def loss_digit_tail(self, outputs, targets, indices, num_boxes, log=True):
        """Digit tail classification loss (NLL)
        targets dicts must contain the key "digit_tail" containing a tensor of dim [nb_target_boxes]
        """
        # Skip if jersey number classification is disabled
        if not self.enable_jn_classification:
            device = next(iter(outputs.values())).device
            loss_digit_tail = torch.tensor(0.0, device=device, requires_grad=True)
            losses = {'loss_digit_tail': loss_digit_tail}
            if log:
                losses['digit_tail_error'] = torch.tensor(0.0, device=device)
            return losses
            
        assert 'pred_digit_tail' in outputs
        src_logits = outputs['pred_digit_tail']

        idx = self._get_src_permutation_idx(indices)
        target_digit_tail_o = torch.cat([t["digit_tail"][J] for t, (_, J) in zip(targets, indices)])
        
        # Digit tail supervision logic based on detection mode
        if self.detect_ball_only:
            # In ball-only mode, there are no person objects, so skip digit tail supervision entirely
            loss_digit_tail = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
            losses = {'loss_digit_tail': loss_digit_tail}
            if log:
                losses['digit_tail_error'] = torch.tensor(0.0, device=src_logits.device)
            return losses
        elif self.detect_ball:
            # In mixed mode, only supervise digit tail for person objects (category 0)
            target_labels_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            person_mask = (target_labels_o == 0).cpu()  # Only person objects
            if person_mask.sum() == 0:
                # No person objects to supervise
                loss_digit_tail = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
                losses = {'loss_digit_tail': loss_digit_tail}
                if log:
                    losses['digit_tail_error'] = torch.tensor(0.0, device=src_logits.device)
                return losses
            
            # Filter to only person objects
            idx_filtered = (idx[0][person_mask], idx[1][person_mask])
            target_digit_tail_o = target_digit_tail_o[person_mask]
        else:
            # In person-only mode, supervise all objects (they are all persons)
            idx_filtered = idx
        
        if self.enable_softmax_focal_loss:
            # Use softmax focal loss
            if len(target_digit_tail_o) > 0:
                loss_digit_tail = softmax_focal_loss(src_logits[idx_filtered], target_digit_tail_o, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            else:
                loss_digit_tail = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
        else:
            # Use sigmoid focal loss (original implementation)
            target_digit_tail_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_digit_tail_onehot[idx_filtered[0], idx_filtered[1], target_digit_tail_o] = 1
            loss_digit_tail = sigmoid_focal_loss(src_logits, target_digit_tail_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        losses = {'loss_digit_tail': loss_digit_tail}

        if log:
            losses['digit_tail_error'] = 100 - accuracy(src_logits[idx_filtered], target_digit_tail_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # assert "batch_len" in kwargs, f"batch_len is not in kwargs"
        # batch_len = kwargs["batch_len"]
        batch_len = self.detr_loss_batch_len
        kwargs = {}     # to default setting

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'roles': self.loss_roles,
            'jn_holistic': self.loss_jn_holistic,
            'digit_head': self.loss_digit_head,
            'digit_tail': self.loss_digit_tail
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        # Organize the batch data:
        loss_dict = {}
        iter_idxs = torch.tensor(list(range(0, len(targets))), dtype=torch.int64, device=outputs['pred_logits'].device)
        for batch_iter_idxs, batch_targets, batch_indices in batch_iterator(
            batch_len, iter_idxs, targets, indices
        ):
            batch_outputs = tensor_dict_index_select(outputs, batch_iter_idxs, dim=0)
            batch_loss_dict = loss_map[loss](batch_outputs, batch_targets, batch_indices, 1, **kwargs)  # num_boxes=1
            for k, v in batch_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = v
                else:
                    loss_dict[k] = loss_dict[k] + v
        # Average the loss:
        if loss == "labels" or loss == "boxes" or loss == "masks" or loss == "roles" or loss == "jn_holistic" or loss == "digit_head" or loss == "digit_tail":
            for k in loss_dict.keys():
                loss_dict[k] /= num_boxes
        return loss_dict
        # return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      For video mode: list of lists, where each inner list contains annotations for each frame
        """
        # Handle video mode: reshape outputs and flatten targets
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = outputs['pred_logits'].shape
            
            # Reshape outputs from [bs, num_frames, ...] to [bs*num_frames, ...]
            reshaped_outputs = {}
            for key, value in outputs.items():
                if key in ['aux_outputs', 'enc_outputs']:
                    continue
                if isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                    # Reshape [bs, num_frames, ...] to [bs*num_frames, ...]
                    reshaped_outputs[key] = value.reshape(bs * num_frames, *value.shape[2:])
                else:
                    reshaped_outputs[key] = value
            
            # Flatten targets: convert list of list of dicts to list of dicts
            flattened_targets = []
            for batch_targets in targets:
                if isinstance(batch_targets, list):
                    # This is a list of annotations for each frame
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                else:
                    # Single frame target
                    flattened_targets.append(batch_targets)
            
            # Use the reshaped data for loss computation
            outputs_for_loss = reshaped_outputs
            targets_for_loss = flattened_targets
        else:
            # Image mode: use original format
            outputs_for_loss = outputs
            targets_for_loss = targets

        outputs_without_aux = {k: v for k, v in outputs_for_loss.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.detr_loss_batch_len is None:
            indices = self.matcher(outputs_without_aux, targets_for_loss)
        else:
            indices = []
            iter_idxs = torch.tensor(
                list(range(0, len(targets_for_loss))), dtype=torch.int64, device=outputs_without_aux['pred_logits'].device
            )
            for batch_iter_idxs, batch_targets in batch_iterator(
                    self.detr_loss_batch_len, iter_idxs, targets_for_loss
            ):
                batch_outputs_without_aux = tensor_dict_index_select(outputs_without_aux, batch_iter_idxs, dim=0)
                _ = self.matcher(batch_outputs_without_aux, batch_targets)
                indices += _
                pass

        # batch_len = kwargs["batch_len"]         # HELLORPG Added
        batch_len = self.detr_loss_batch_len
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets_for_loss)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs_for_loss.values())).device)
        if is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"batch_len": batch_len}         # HELLORPG Added
            losses.update(self.get_loss(loss, outputs_for_loss, targets_for_loss, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # Handle video mode for auxiliary outputs
                if self.detection_data_type == "video" and self.backbone_type == "video":
                    if len(aux_outputs['pred_logits'].shape) == 4:  # [bs, num_frames, num_queries, num_classes]
                        bs, num_frames = aux_outputs['pred_logits'].shape[:2]
                        # Reshape auxiliary outputs
                        reshaped_aux_outputs = {}
                        for key, value in aux_outputs.items():
                            if isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                                reshaped_aux_outputs[key] = value.reshape(bs * num_frames, *value.shape[2:])
                            else:
                                reshaped_aux_outputs[key] = value
                        aux_outputs = reshaped_aux_outputs
                
                indices = self.matcher(aux_outputs, targets_for_loss)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs["batch_len"] = batch_len     # HELLORPG Added
                    l_dict = self.get_loss(loss, aux_outputs, targets_for_loss, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            # Handle video mode for encoder outputs
            if self.detection_data_type == "video" and self.backbone_type == "video":
                if len(enc_outputs['pred_logits'].shape) == 4:  # [bs, num_frames, num_queries, num_classes]
                    bs, num_frames = enc_outputs['pred_logits'].shape[:2]
                    # Reshape encoder outputs
                    reshaped_enc_outputs = {}
                    for key, value in enc_outputs.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                            reshaped_enc_outputs[key] = value.reshape(bs * num_frames, *value.shape[2:])
                        else:
                            reshaped_enc_outputs[key] = value
                    enc_outputs = reshaped_enc_outputs
            
            bin_targets = copy.deepcopy(targets_for_loss)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # losses = {k: (v * self.weight_dict[k] if k in self.weight_dict else v) for k, v in losses.items()}

        return losses, self.weight_dict, indices



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 处理attributes（如果存在）
        results = []
        for batch_idx, (s, l, b) in enumerate(zip(scores, labels, boxes)):
            result = {'scores': s, 'labels': l, 'boxes': b, 'topk_boxes': topk_boxes[batch_idx]}
            
            # 添加attributes（只有在相应功能启用时才处理）
            if 'pred_roles' in outputs:
                pred_roles = outputs['pred_roles'][batch_idx]  # [num_queries, num_role_classes]
                roles = torch.argmax(pred_roles, dim=-1)  # [num_queries]
                result['roles'] = torch.gather(roles, 0, topk_boxes[batch_idx])
            
            if 'pred_jn_holistic' in outputs:
                pred_jersey = outputs['pred_jn_holistic'][batch_idx]  # [num_queries, num_jersey_classes]
                jersey = torch.argmax(pred_jersey, dim=-1)  # [num_queries]
                result['jersey'] = torch.gather(jersey, 0, topk_boxes[batch_idx])
            
            if 'pred_digit_head' in outputs:
                pred_digit_head = outputs['pred_digit_head'][batch_idx]  # [num_queries, num_digit_head_classes]
                digit_head = torch.argmax(pred_digit_head, dim=-1)  # [num_queries]
                result['digit_head'] = torch.gather(digit_head, 0, topk_boxes[batch_idx])
            
            if 'pred_digit_tail' in outputs:
                pred_digit_tail = outputs['pred_digit_tail'][batch_idx]  # [num_queries, num_digit_tail_classes]
                digit_tail = torch.argmax(pred_digit_tail, dim=-1)  # [num_queries]
                result['digit_tail'] = torch.gather(digit_tail, 0, topk_boxes[batch_idx])
            
            results.append(result)

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Args:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: fields to add to this `Instances`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        # with warnings.catch_warnings(record=True):
        #     data_len = len(value)
        # if len(self._fields):
        #     assert (
        #         len(self) == data_len
        #     ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

def cvt_config_to_args(config: dict):
    # Generate DETR args:
    detr_args = Args()
    
    # Validate configuration options
    if config["DETECT_BALL_ONLY"] and config["DETR_DETECT_BALL"]:
        print("Warning: Both DETECT_BALL_ONLY and DETR_DETECT_BALL are set to True. DETECT_BALL_ONLY takes precedence.")
    
    # 1. transformer:
    # Automatically set num_classes based on DETR_DETECT_BALL and DETECT_BALL_ONLY
    if config["DETECT_BALL_ONLY"]:
        detr_args.num_classes = 1  # only ball (0 in this case, since we remap ball to index 0)
    elif config["DETR_DETECT_BALL"]:
        detr_args.num_classes = 2  # person (0) and ball (1)
    else:
        detr_args.num_classes = config["NUM_CLASSES"]  # only person (0)
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.role_loss_coef = config["DETR_ROLE_LOSS_COEF"]
    detr_args.jn_loss_coef = config["DETR_JN_LOSS_COEF"]
    detr_args.digit_head_loss_coef = config["DETR_DIGIT_HEAD_LOSS_COEF"]
    detr_args.digit_tail_loss_coef = config["DETR_DIGIT_TAIL_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    detr_args.backbone_strides = [16]
    detr_args.backbone_num_channels = [config["BACKBONE_HIDDEN_DIM"]]
    detr_args.enable_softmax_focal_loss = config["ENABLE_SOFTMAX_FOCAL_LOSS"]
    
    return detr_args
    
    
def build_deformable_detr_head(config: dict):
    args = cvt_config_to_args(config)
    device = torch.device(args.device)
    
    # Adjust backbone_num_channels based on local_features_type
    local_features_type = config["DETR_LOCAL_FEATURES_TYPE"]
    base_channels = config["BACKBONE_HIDDEN_DIM"]
    
    if local_features_type == 'both':
        # When concatenating early and late features, channels are doubled
        backbone_num_channels = [base_channels * 2]
    else:
        backbone_num_channels = [base_channels]
    
    head = DeformableDetrHead(
        position_encoding = build_position_encoding(args),
        transformer = build_deforamble_transformer(args),
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        backbone_strides=args.backbone_strides,
        backbone_num_channels=backbone_num_channels,  # Use adjusted channels
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        detection_data_type=config["DETECTION_DATA_TYPE"],
        backbone_type=config["BACKBONE_TYPE"],
        enable_role_classification=config["DETR_ENABLE_ROLE_CLASSIFICATION"],
        enable_jn_classification=config["DETR_ENABLE_JN_CLASSIFICATION"],
        local_features_type=local_features_type,
    )
    return head

def build_deformable_detr_criterion(config: dict):
    args = cvt_config_to_args(config)
    
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    
    # Only add role and jersey number loss weights if the corresponding features are enabled
    if config["DETR_ENABLE_ROLE_CLASSIFICATION"]:
        weight_dict['loss_role'] = args.role_loss_coef
    if config["DETR_ENABLE_JN_CLASSIFICATION"]:
        weight_dict['loss_jn_holistic'] = args.jn_loss_coef
        weight_dict['loss_digit_head'] = args.digit_head_loss_coef
        weight_dict['loss_digit_tail'] = args.digit_tail_loss_coef
    
    assert args.masks is False, "MASKS is not supported yet."
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    if config["MODEL_ARCH"] == "multitask":
        losses = ['labels', 'boxes', 'cardinality']
        # Only add role and jersey number losses if the corresponding features are enabled
        if config["DETR_ENABLE_ROLE_CLASSIFICATION"]:
            losses.append('roles')
        if config["DETR_ENABLE_JN_CLASSIFICATION"]:
            losses.extend(['jn_holistic', 'digit_head', 'digit_tail'])
    elif config["MODEL_ARCH"] == "yolo":
        losses = ['labels', 'boxes', 'cardinality']

    detr_criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=build_matcher(args),
        weight_dict=weight_dict,
        losses = losses,
        focal_alpha=args.focal_alpha,
        detr_loss_batch_len=config["DETR_CRITERION_BATCH_LEN"],
        detection_data_type=config["DETECTION_DATA_TYPE"],
        backbone_type=config["BACKBONE_TYPE"],
        enable_softmax_focal_loss=config["ENABLE_SOFTMAX_FOCAL_LOSS"],
        detect_ball=config["DETR_DETECT_BALL"],
        detect_ball_only=config["DETECT_BALL_ONLY"],
        enable_role_classification=config["DETR_ENABLE_ROLE_CLASSIFICATION"],
        enable_jn_classification=config["DETR_ENABLE_JN_CLASSIFICATION"],
    )
    return detr_criterion

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]
        
def tensor_dict_index_select(tensor_dict, index, dim=0):
    res_tensor_dict = defaultdict()
    for k in tensor_dict.keys():
        if isinstance(tensor_dict[k], torch.Tensor):
            res_tensor_dict[k] = torch.index_select(tensor_dict[k], index=index, dim=dim).contiguous()
        elif isinstance(tensor_dict[k], dict):
            res_tensor_dict[k] = tensor_dict_index_select(tensor_dict[k], index=index, dim=dim)
        elif isinstance(tensor_dict[k], list):
            res_tensor_dict[k] = [
                tensor_dict_index_select(tensor_dict[k][_], index=index, dim=dim)
                for _ in range(len(tensor_dict[k]))
            ]
        else:
            raise ValueError(f"Unsupported type {type(tensor_dict[k])} in the tensor dict index select.")
    return dict(res_tensor_dict)

class DetectionMetrics(nn.Module):
    """
    计算detection常见的metrics，包括mAP、IoU、precision、recall等指标
    支持多进程聚合和整个数据集上的AP计算
    支持分类别计算metrics（人和球分别计算）
    """
    def __init__(self, num_classes, iou_thresholds=None, score_threshold=0.5, backbone_type='image', class_names=None, enable_role_classification=True, enable_jn_classification=True):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.score_threshold = score_threshold
        self.backbone_type = backbone_type
        self.enable_role_classification = enable_role_classification
        self.enable_jn_classification = enable_jn_classification
        self.postprocess = PostProcess()
        
        # 设置类别名称，用于分类别metrics
        if class_names is None:
            if num_classes == 1:
                self.class_names = ['person']
            elif num_classes == 2:
                self.class_names = ['person', 'ball']
            else:
                self.class_names = [f'class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
        
        self.reset()
        
    def reset(self):
        """重置收集的数据"""
        # 总体的metrics数据
        self.tp_fp_scores_per_thresh = {thresh: {'tp': [], 'fp': [], 'scores': []} for thresh in self.iou_thresholds}
        self.total_gt_count = 0
        
        # 分类别的metrics数据
        self.tp_fp_scores_per_class_thresh = {}
        self.total_gt_count_per_class = {}
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            self.tp_fp_scores_per_class_thresh[class_name] = {
                thresh: {'tp': [], 'fp': [], 'scores': []} for thresh in self.iou_thresholds
            }
            self.total_gt_count_per_class[class_name] = 0
        
        # Only initialize attribute tracking if the corresponding features are enabled
        self.attribute_matches = {}
        if self.enable_role_classification:
            self.attribute_matches['role'] = {'correct': [], 'total': []}
        if self.enable_jn_classification:
            self.attribute_matches['jersey'] = {'correct': [], 'total': []}
            self.attribute_matches['digit_head'] = {'correct': [], 'total': []}
            self.attribute_matches['digit_tail'] = {'correct': [], 'total': []}
        
    def box_iou(self, boxes1, boxes2):
        """
        计算两组box之间的IoU
        boxes: [N, 4] format: x1, y1, x2, y2
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # 计算交集
        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = area1[:, None] + area2[None, :] - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        return iou

    def compute_ap(self, precision, recall):
        """
        计算Average Precision (AP)
        """
        # 添加起始和结束点
        mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

        # 计算precision的包络线
        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

        # 计算面积
        i = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
        
    def update(self, outputs, targets, target_sizes):
        """
        在当前batch上计算TP/FP并收集结果
        
        Args:
            outputs: 模型输出，可能包含video模式的数据 
            targets: 真实标注，可能是list of lists for video mode
            target_sizes: 图像尺寸
        """
        device = outputs['pred_logits'].device
        
        # Handle video mode: reshape outputs and flatten targets
        if self.backbone_type == 'video':
            # Check if outputs contain video-shaped data
            if len(outputs['pred_logits'].shape) == 4:  # [bs, num_frames, num_queries, num_classes]
                bs, num_frames = outputs['pred_logits'].shape[:2]
                
                # Reshape outputs from [bs, num_frames, ...] to [bs*num_frames, ...]
                reshaped_outputs = {}
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) >= 3:
                        reshaped_outputs[key] = value.reshape(bs * num_frames, *value.shape[2:])
                    else:
                        reshaped_outputs[key] = value
                outputs_for_metrics = reshaped_outputs
                
                # Flatten target_sizes if needed
                if target_sizes.ndim == 3:  # [bs, num_frames, 2]
                    target_sizes = target_sizes.reshape(bs * num_frames, target_sizes.shape[-1])
                elif target_sizes.ndim == 2 and target_sizes.shape[0] == bs:  # [bs, 2] - repeat for all frames
                    target_sizes = target_sizes.unsqueeze(1).repeat(1, num_frames, 1).reshape(bs * num_frames, target_sizes.shape[-1])
            else:
                outputs_for_metrics = outputs
            
            # Check if targets is list of lists (video mode)
            if targets and isinstance(targets[0], list):
                # Flatten targets: convert list of list of dicts to list of dicts
                flattened_targets = []
                for batch_targets in targets:
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                targets_for_metrics = flattened_targets
            else:
                # Already flattened
                targets_for_metrics = targets
        else:
            # Image mode: use original format
            outputs_for_metrics = outputs
            targets_for_metrics = targets
        
        # 使用PostProcess获取预测结果（已包含attributes）
        predictions = self.postprocess(outputs_for_metrics, target_sizes)
        
        # 为每个IoU阈值计算TP/FP
        for iou_thresh in self.iou_thresholds:
            tp_list = []
            fp_list = []
            scores_list = []
            
            # 分类别的TP/FP/scores列表
            tp_list_per_class = {class_name: [] for class_name in self.class_names}
            fp_list_per_class = {class_name: [] for class_name in self.class_names}
            scores_list_per_class = {class_name: [] for class_name in self.class_names}
            
            # 处理当前batch中的每个sample
            for sample_idx, (pred, target, target_size) in enumerate(zip(predictions, targets_for_metrics, target_sizes)):
                pred_boxes = pred['boxes']  # [N, 4]
                pred_scores = pred['scores']  # [N]
                pred_labels = pred['labels']  # [N]
                
                gt_boxes = target['boxes']  # [M, 4] 
                gt_labels = target['labels']  # [M]
                
                # 转换gt_boxes到绝对坐标（如果需要）
                if len(gt_boxes) > 0:
                    if gt_boxes.max() <= 1.0:  # 如果是相对坐标
                        if isinstance(target_size, torch.Tensor):
                            h, w = target_size[0], target_size[1]
                        else:
                            h, w = target_size[0], target_size[1]
                        gt_boxes = gt_boxes * torch.tensor([w, h, w, h], device=gt_boxes.device)
                    
                    # 转换为x1,y1,x2,y2格式（如果是cxcywh格式）
                    gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                
                # 过滤低分预测
                if len(pred_boxes) > 0:
                    valid_mask = pred_scores > self.score_threshold
                    pred_boxes = pred_boxes[valid_mask]
                    pred_scores = pred_scores[valid_mask]
                    pred_labels = pred_labels[valid_mask]
                
                if len(pred_boxes) == 0:
                    continue
                
                # 按分数排序
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                
                # 计算IoU矩阵并匹配
                if len(gt_boxes) > 0:
                    ious = self.box_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]
                    
                    # 为每个预测找到最佳匹配的GT
                    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
                    
                    for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                        pred_class_name = self.class_names[pred_label.item()]
                        
                        # 找到与当前预测同类别的GT
                        same_class_mask = (gt_labels == pred_label)
                        if not same_class_mask.any():
                            fp_list.append(1)
                            tp_list.append(0)
                            fp_list_per_class[pred_class_name].append(1)
                            tp_list_per_class[pred_class_name].append(0)
                        else:
                            # 在同类别GT中找到IoU最大的
                            class_ious = ious[i] * same_class_mask.float()
                            max_iou, max_idx = torch.max(class_ious, dim=0)
                            
                            if max_iou >= iou_thresh and not gt_matched[max_idx]:
                                tp_list.append(1)
                                fp_list.append(0)
                                tp_list_per_class[pred_class_name].append(1)
                                fp_list_per_class[pred_class_name].append(0)
                                gt_matched[max_idx] = True
                                
                                # 只在IoU@0.5时计算attributes准确度
                                if iou_thresh == 0.5:
                                    self._compute_attribute_accuracy(pred, target, i, max_idx.item())
                            else:
                                tp_list.append(0)
                                fp_list.append(1)
                                tp_list_per_class[pred_class_name].append(0)
                                fp_list_per_class[pred_class_name].append(1)
                        
                        scores_list.append(pred_score.cpu().item())  # 转到CPU
                        scores_list_per_class[pred_class_name].append(pred_score.cpu().item())
                else:
                    # 没有GT，所有预测都是FP
                    fp_list.extend([1] * len(pred_boxes))
                    tp_list.extend([0] * len(pred_boxes))
                    scores_list.extend(pred_scores.cpu().tolist())  # 转到CPU
                    
                    # 分类别处理
                    for pred_label, pred_score in zip(pred_labels, pred_scores):
                        pred_class_name = self.class_names[pred_label.item()]
                        fp_list_per_class[pred_class_name].append(1)
                        tp_list_per_class[pred_class_name].append(0)
                        scores_list_per_class[pred_class_name].append(pred_score.cpu().item())
            
            # 将当前batch的结果添加到对应IoU阈值的收集器中
            self.tp_fp_scores_per_thresh[iou_thresh]['tp'].extend(tp_list)
            self.tp_fp_scores_per_thresh[iou_thresh]['fp'].extend(fp_list)
            self.tp_fp_scores_per_thresh[iou_thresh]['scores'].extend(scores_list)
            
            # 分类别收集结果
            for class_name in self.class_names:
                self.tp_fp_scores_per_class_thresh[class_name][iou_thresh]['tp'].extend(tp_list_per_class[class_name])
                self.tp_fp_scores_per_class_thresh[class_name][iou_thresh]['fp'].extend(fp_list_per_class[class_name])
                self.tp_fp_scores_per_class_thresh[class_name][iou_thresh]['scores'].extend(scores_list_per_class[class_name])
        
        # 统计GT数量
        batch_gt_count = sum(len(target['labels']) for target in targets_for_metrics)
        self.total_gt_count += batch_gt_count
        
        # 分类别统计GT数量
        for target in targets_for_metrics:
            for gt_label in target['labels']:
                class_name = self.class_names[gt_label.item()]
                self.total_gt_count_per_class[class_name] += 1
            
    def _compute_attribute_accuracy(self, pred, target, pred_idx, gt_idx):
        """
        计算匹配成功的预测的attribute准确度
        只对person类别计算role、jersey、digit相关的属性准确度
        
        Args:
            pred: 单个样本的预测结果（来自PostProcess，已包含attributes）
            target: 单个样本的真实标注
            pred_idx: 预测框的索引
            gt_idx: 匹配的GT框的索引
        """
        # 检查当前预测是否为person类别
        # 在ball-only模式下，球被映射到类别0，但在这种情况下不应该计算person属性
        # 在mixed模式下，person类别为0，ball类别为1
        # 在person-only模式下，所有对象都是person（类别0）
        
        pred_label = pred['labels'][pred_idx].item()
        gt_label = target['labels'][gt_idx].item()
        
        # 只有当预测和GT都是person时才计算person属性
        # person的判断逻辑：
        # - 在person-only模式（class_names=['person']）下：类别0是person
        # - 在mixed模式（class_names=['person', 'ball']）下：类别0是person，类别1是ball
        # - 在ball-only模式（class_names=['ball']）下：类别0是ball，不计算person属性
        
        is_person_pred = False
        is_person_gt = False
        
        if len(self.class_names) == 1:
            if self.class_names[0] == 'person':
                # person-only模式
                is_person_pred = (pred_label == 0)
                is_person_gt = (gt_label == 0)
            else:
                # ball-only模式，不应该计算person属性
                is_person_pred = False
                is_person_gt = False
        elif len(self.class_names) == 2 and 'person' in self.class_names and 'ball' in self.class_names:
            # mixed模式：person=0, ball=1
            is_person_pred = (pred_label == 0)
            is_person_gt = (gt_label == 0)
        
        # 只有当预测和GT都是person时才计算person相关的属性
        if not (is_person_pred and is_person_gt):
            return
        
        # 获取GT的attributes
        gt_roles = target.get('roles', None)
        gt_jersey = target.get('jersey', None)
        gt_digit_head = target.get('digit_head', None) 
        gt_digit_tail = target.get('digit_tail', None)
        
        # 计算role准确度（只有在启用时才计算）
        if (self.enable_role_classification and 'roles' in pred and 
            gt_roles is not None and gt_idx < len(gt_roles)):
            pred_role = pred['roles'][pred_idx].item()
            gt_role = gt_roles[gt_idx].item() if isinstance(gt_roles[gt_idx], torch.Tensor) else gt_roles[gt_idx]
            self.attribute_matches['role']['correct'].append(1 if pred_role == gt_role else 0)
            self.attribute_matches['role']['total'].append(1)
        
        # 计算jersey准确度（只有在启用时才计算）
        if (self.enable_jn_classification and 'jersey' in pred and 
            gt_jersey is not None and gt_idx < len(gt_jersey)):
            pred_jn = pred['jersey'][pred_idx].item()
            gt_jn = gt_jersey[gt_idx].item() if isinstance(gt_jersey[gt_idx], torch.Tensor) else gt_jersey[gt_idx]
            self.attribute_matches['jersey']['correct'].append(1 if pred_jn == gt_jn else 0)
            self.attribute_matches['jersey']['total'].append(1)
        
        # 计算digit_head准确度（只有在启用时才计算）
        if (self.enable_jn_classification and 'digit_head' in pred and 
            gt_digit_head is not None and gt_idx < len(gt_digit_head)):
            pred_dh = pred['digit_head'][pred_idx].item()
            gt_dh = gt_digit_head[gt_idx].item() if isinstance(gt_digit_head[gt_idx], torch.Tensor) else gt_digit_head[gt_idx]
            self.attribute_matches['digit_head']['correct'].append(1 if pred_dh == gt_dh else 0)
            self.attribute_matches['digit_head']['total'].append(1)
        
        # 计算digit_tail准确度（只有在启用时才计算）
        if (self.enable_jn_classification and 'digit_tail' in pred and 
            gt_digit_tail is not None and gt_idx < len(gt_digit_tail)):
            pred_dt = pred['digit_tail'][pred_idx].item()
            gt_dt = gt_digit_tail[gt_idx].item() if isinstance(gt_digit_tail[gt_idx], torch.Tensor) else gt_digit_tail[gt_idx]
            self.attribute_matches['digit_tail']['correct'].append(1 if pred_dt == gt_dt else 0)
            self.attribute_matches['digit_tail']['total'].append(1)

    def gather_tp_fp_scores(self, accelerator):
        """
        在所有进程间聚合TP/FP/scores结果和attribute匹配结果
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            gathered_tp_fp_scores_per_thresh, gathered_total_gt_count, gathered_attribute_matches, 
            gathered_tp_fp_scores_per_class_thresh, gathered_total_gt_count_per_class
        """
        # 聚合每个IoU阈值的TP/FP/scores
        gathered_tp_fp_scores = {}
        key_list = ['tp', 'fp', 'scores']
        for thresh in self.iou_thresholds:
            gathered_tp_fp_scores[thresh] = {}
            for key in key_list:
                gathered_tp_fp_scores[thresh][key] = gather_object(self.tp_fp_scores_per_thresh[thresh][key])
        
        # 聚合分类别的TP/FP/scores
        gathered_tp_fp_scores_per_class = {}
        for class_name in self.class_names:
            gathered_tp_fp_scores_per_class[class_name] = {}
            for thresh in self.iou_thresholds:
                gathered_tp_fp_scores_per_class[class_name][thresh] = {}
                for key in key_list:
                    gathered_tp_fp_scores_per_class[class_name][thresh][key] = gather_object(
                        self.tp_fp_scores_per_class_thresh[class_name][thresh][key]
                    )
        
        # 聚合GT总数（需要包装成列表）
        gathered_gt_count = gather_object([self.total_gt_count])
        
        # 聚合分类别GT总数
        gathered_gt_count_per_class = {}
        for class_name in self.class_names:
            gathered_gt_count_per_class[class_name] = gather_object([self.total_gt_count_per_class[class_name]])
        
        # 聚合attribute匹配结果（只聚合启用的attributes）
        key_list_attr = ['correct', 'total']
        gathered_attribute_matches = {}
        for attr_name in self.attribute_matches.keys():
            gathered_attribute_matches[attr_name] = {}
            for key in key_list_attr:
                gathered_attribute_matches[attr_name][key] = gather_object(self.attribute_matches[attr_name][key])
        
        return (gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, 
                gathered_tp_fp_scores_per_class, gathered_gt_count_per_class)

    def compute_metrics_from_gathered_tp_fp(self, gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, 
                                           gathered_tp_fp_scores_per_class=None, gathered_gt_count_per_class=None):
        """
        从聚合的TP/FP/scores数据计算metrics
        
        Args:
            gathered_tp_fp_scores: 聚合的TP/FP/scores数据
            gathered_gt_count: 聚合的GT总数
            gathered_attribute_matches: 聚合的attribute匹配结果
            gathered_tp_fp_scores_per_class: 聚合的分类别TP/FP/scores数据
            gathered_gt_count_per_class: 聚合的分类别GT总数
            
        Returns:
            dict: 包含各种metrics的字典
        """
        # 初始化metrics
        metrics = {}
        
        # 为每个IoU阈值计算metrics
        for iou_thresh in self.iou_thresholds:
            thresh_data = gathered_tp_fp_scores[iou_thresh]
            
            # 展平所有进程的数据
            all_tp = flatten_data(thresh_data['tp'])
            all_fp = flatten_data(thresh_data['fp'])
            all_scores = flatten_data(thresh_data['scores'])
            
            if len(all_tp) > 0:
                # 转换为tensor
                tp = torch.tensor(all_tp, dtype=torch.float32)
                fp = torch.tensor(all_fp, dtype=torch.float32)
                scores = torch.tensor(all_scores, dtype=torch.float32)
                
                # 按分数排序
                sorted_indices = torch.argsort(scores, descending=True)
                tp = tp[sorted_indices]
                fp = fp[sorted_indices]
                
                # 计算累积TP和FP
                tp_cumsum = torch.cumsum(tp, dim=0)
                fp_cumsum = torch.cumsum(fp, dim=0)
                
                # 计算precision和recall
                # gathered_gt_count是列表的列表，需要求和
                total_gt_count = sum(gathered_gt_count)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                recall = tp_cumsum / (total_gt_count + 1e-8)
                
                # 计算AP
                ap = self.compute_ap(precision, recall)
                metrics[f'AP@{iou_thresh:.2f}'] = ap.item()
                
                # 保存最终的precision和recall用于计算整体指标
                if iou_thresh == 0.5:
                    final_precision = precision[-1].item() if len(precision) > 0 else 0.0
                    final_recall = recall[-1].item() if len(recall) > 0 else 0.0
                    
                    metrics['precision'] = final_precision
                    metrics['recall'] = final_recall
                    if final_precision + final_recall > 0:
                        metrics['f1'] = 2 * final_precision * final_recall / (final_precision + final_recall)
                    else:
                        metrics['f1'] = 0.0
            else:
                metrics[f'AP@{iou_thresh:.2f}'] = 0.0
                if iou_thresh == 0.5:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
        
        # 计算mAP (所有IoU阈值的平均)
        ap_values = [metrics[f'AP@{thresh:.2f}'] for thresh in self.iou_thresholds]
        metrics['mAP'] = sum(ap_values) / len(ap_values)
        metrics['mAP@0.5'] = metrics.get('AP@0.50', 0.0)
        metrics['mAP@0.75'] = metrics.get('AP@0.75', 0.0)
        
        # 计算attribute准确度（只计算启用的attributes）
        for attr_name in gathered_attribute_matches.keys():
            attr_data = gathered_attribute_matches[attr_name]
            
            # 展平所有进程的数据
            all_correct = flatten_data(attr_data['correct'])
            all_total = flatten_data(attr_data['total'])
            
            # 计算准确度
            if len(all_total) > 0:
                accuracy = sum(all_correct) / len(all_total)
                metrics[f'{attr_name}_accuracy'] = accuracy
                metrics[f'{attr_name}_matched_count'] = len(all_total)
            else:
                metrics[f'{attr_name}_accuracy'] = 0.0
                metrics[f'{attr_name}_matched_count'] = 0
        
        # 计算分类别metrics
        if gathered_tp_fp_scores_per_class is not None and gathered_gt_count_per_class is not None:
            for class_name in self.class_names:
                class_tp_fp_scores = gathered_tp_fp_scores_per_class[class_name]
                class_gt_count = sum(gathered_gt_count_per_class[class_name])
                
                # 为每个IoU阈值计算分类别metrics
                for iou_thresh in self.iou_thresholds:
                    thresh_data = class_tp_fp_scores[iou_thresh]
                    
                    # 展平所有进程的数据
                    all_tp = flatten_data(thresh_data['tp'])
                    all_fp = flatten_data(thresh_data['fp'])
                    all_scores = flatten_data(thresh_data['scores'])
                    
                    if len(all_tp) > 0 and class_gt_count > 0:
                        # 转换为tensor
                        tp = torch.tensor(all_tp, dtype=torch.float32)
                        fp = torch.tensor(all_fp, dtype=torch.float32)
                        scores = torch.tensor(all_scores, dtype=torch.float32)
                        
                        # 按分数排序
                        sorted_indices = torch.argsort(scores, descending=True)
                        tp = tp[sorted_indices]
                        fp = fp[sorted_indices]
                        
                        # 计算累积TP和FP
                        tp_cumsum = torch.cumsum(tp, dim=0)
                        fp_cumsum = torch.cumsum(fp, dim=0)
                        
                        # 计算precision和recall
                        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                        recall = tp_cumsum / (class_gt_count + 1e-8)
                        
                        # 计算AP
                        ap = self.compute_ap(precision, recall)
                        metrics[f'{class_name}_AP@{iou_thresh:.2f}'] = ap.item()
                        
                        # 保存最终的precision和recall用于计算整体指标
                        if iou_thresh == 0.5:
                            final_precision = precision[-1].item() if len(precision) > 0 else 0.0
                            final_recall = recall[-1].item() if len(recall) > 0 else 0.0
                            
                            metrics[f'{class_name}_precision'] = final_precision
                            metrics[f'{class_name}_recall'] = final_recall
                            if final_precision + final_recall > 0:
                                metrics[f'{class_name}_f1'] = 2 * final_precision * final_recall / (final_precision + final_recall)
                            else:
                                metrics[f'{class_name}_f1'] = 0.0
                    else:
                        metrics[f'{class_name}_AP@{iou_thresh:.2f}'] = 0.0
                        if iou_thresh == 0.5:
                            metrics[f'{class_name}_precision'] = 0.0
                            metrics[f'{class_name}_recall'] = 0.0
                            metrics[f'{class_name}_f1'] = 0.0
                
                # 计算分类别mAP
                class_ap_values = [metrics[f'{class_name}_AP@{thresh:.2f}'] for thresh in self.iou_thresholds]
                metrics[f'{class_name}_mAP'] = sum(class_ap_values) / len(class_ap_values)
                metrics[f'{class_name}_mAP@0.5'] = metrics.get(f'{class_name}_AP@0.50', 0.0)
                metrics[f'{class_name}_mAP@0.75'] = metrics.get(f'{class_name}_AP@0.75', 0.0)
                metrics[f'{class_name}_gt_count'] = class_gt_count

        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets, target_sizes):
        """
        计算detection metrics (保持向后兼容)
        这个方法现在只是调用update来收集数据
        """
        self.update(outputs, targets, target_sizes)
        # 返回空字典，实际的metrics计算在compute_final_metrics中进行
        return {}
        
    def compute_final_metrics(self, accelerator):
        """
        计算最终的metrics（在所有数据收集完成后调用）
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            dict: 包含各种metrics的字典
        """
        # 聚合所有进程的TP/FP/scores结果和attribute匹配结果
        (gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, 
         gathered_tp_fp_scores_per_class, gathered_gt_count_per_class) = self.gather_tp_fp_scores(accelerator)
        
        # 只在主进程计算metrics
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_tp_fp(
                gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches,
                gathered_tp_fp_scores_per_class, gathered_gt_count_per_class
            )
        else:
            return {}


def build_detection_metrics(config: dict):
    """
    构建detection metrics计算器
    """
    # Automatically set num_classes based on DETR_DETECT_BALL and DETECT_BALL_ONLY
    if config["DETECT_BALL_ONLY"]:
        num_classes = 1  # only ball (0)
        class_names = ['ball']
    elif config["DETR_DETECT_BALL"]:
        num_classes = 2  # person (0) and ball (1)
        class_names = ['person', 'ball']
    else:
        num_classes = config["NUM_CLASSES"]  # only person (0)
        class_names = ['person']
    
    metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        score_threshold=config["EVAL_SCORE_THRESHOLD"],
        backbone_type=config["BACKBONE_TYPE"],
        class_names=class_names,
        enable_role_classification=config["DETR_ENABLE_ROLE_CLASSIFICATION"],
        enable_jn_classification=config["DETR_ENABLE_JN_CLASSIFICATION"],
    )
    
    return metrics


