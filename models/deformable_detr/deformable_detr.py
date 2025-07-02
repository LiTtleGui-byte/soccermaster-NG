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
from models.deformable_detr.vggt.head_act import activate_pose


from models.deformable_detr.position_encoding import build_position_encoding
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from data.soccernet_gsr_reid import role_mapping, jn_mapping, digit_head_mapping, digit_tail_mapping

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDetrHead(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, position_encoding, transformer, num_classes, num_queries, num_feature_levels, backbone_strides, backbone_num_channels, num_keypoints,
                 aux_loss=True, with_box_refine=False, two_stage=False, backbone_type='image'):
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
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        num_role_classes = len(role_mapping)
        num_jn_classes = len(jn_mapping)
        num_digit_head_classes = len(digit_head_mapping)
        num_digit_tail_classes = len(digit_tail_mapping)
        self.role_embed = nn.Linear(hidden_dim, num_role_classes)
        self.jn_holistic_embed = nn.Linear(hidden_dim, num_jn_classes)
        self.digit_head_embed = nn.Linear(hidden_dim, num_digit_head_classes)
        self.digit_tail_embed = nn.Linear(hidden_dim, num_digit_tail_classes)
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
        self.backbone_type = backbone_type
        self.camera_head = ConvCameraHead(input_channels=backbone_num_channels[0])
        self.keypoints_head = KeypointsHead(dim_in=backbone_num_channels[0], num_keypoints=num_keypoints)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.role_embed.bias.data = torch.ones(num_role_classes) * bias_value
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
            self.role_embed = _get_clones(self.role_embed, num_pred)
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
            self.role_embed = nn.ModuleList([self.role_embed for _ in range(num_pred)])
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
        global_features, local_features = backbone_outputs['global_features'], backbone_outputs['local_features']
        if self.backbone_type == 'video':
            global_features = global_features[:, 0]
            local_features = local_features[:, 0]

        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        features = reshaped_local_features
        
        # TODO: this will lead to a error, and can not backward gradient
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
                m = features.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_roles = []
        outputs_jn_holistic = []
        outputs_digit_head = []
        outputs_digit_tail = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_role = self.role_embed[lvl](hs[lvl])
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
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_roles.append(outputs_role)
            outputs_jn_holistic.append(outputs_jn)
            outputs_digit_head.append(outputs_digit_h)
            outputs_digit_tail.append(outputs_digit_t)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_role = torch.stack(outputs_roles)
        outputs_jn_holistic = torch.stack(outputs_jn_holistic)
        outputs_digit_head = torch.stack(outputs_digit_head)
        outputs_digit_tail = torch.stack(outputs_digit_tail)

        # Use ConvCameraHead with reshaped features
        quaternion, translation, fov = self.camera_head(reshaped_local_features)
        # Use KeypointsHead with reshaped features
        keypoints_heatmap = self.keypoints_head(reshaped_local_features)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_roles': outputs_role[-1], 'pred_jn_holistic': outputs_jn_holistic[-1], 'pred_digit_head': outputs_digit_head[-1], 'pred_digit_tail': outputs_digit_tail[-1]}
        # Add camera predictions to output
        out['pred_camera'] = {
            'quaternion': quaternion,
            'translation': translation,
            'fov': fov
        }
        out['pred_keypoints_heatmap'] = keypoints_heatmap
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_role, outputs_jn_holistic, outputs_digit_head, outputs_digit_tail)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Output the outputs of last decoder layer.
        # We need these outputs to generate the embeddings for objects.
        out["outputs"] = hs[-1]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_role, outputs_jn_holistic, outputs_digit_head, outputs_digit_tail):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_roles': c, 'pred_jn_holistic': d, 'pred_digit_head': e, 'pred_digit_tail': f}
                for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_coord[:-1], outputs_role[:-1], outputs_jn_holistic[:-1], outputs_digit_head[:-1], outputs_digit_tail[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, detr_loss_batch_len=10):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_role_classes = len(role_mapping)
        self.num_jn_classes = len(jn_mapping)
        self.num_digit_head_classes = len(digit_head_mapping)
        self.num_digit_tail_classes = len(digit_tail_mapping)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.detr_loss_batch_len = detr_loss_batch_len
        
        
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
        assert 'pred_roles' in outputs
        src_logits = outputs['pred_roles']

        idx = self._get_src_permutation_idx(indices)
        target_roles_o = torch.cat([t["roles"][J] for t, (_, J) in zip(targets, indices)])
        
        target_roles_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_roles_onehot[idx[0], idx[1], target_roles_o] = 1
        
        loss_role = sigmoid_focal_loss(src_logits, target_roles_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_role': loss_role}

        if log:
            losses['role_error'] = 100 - accuracy(src_logits[idx], target_roles_o)[0]
        return losses

    def loss_jn_holistic(self, outputs, targets, indices, num_boxes, log=True):
        """Jersey number holistic classification loss (NLL)
        targets dicts must contain the key "jn_holistic" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_jn_holistic' in outputs
        src_logits = outputs['pred_jn_holistic']

        idx = self._get_src_permutation_idx(indices)
        target_jn_holistic_o = torch.cat([t["jersey"][J] for t, (_, J) in zip(targets, indices)])
        
        target_jn_holistic_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_jn_holistic_onehot[idx[0], idx[1], target_jn_holistic_o] = 1
        
        loss_jn_holistic = sigmoid_focal_loss(src_logits, target_jn_holistic_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_jn_holistic': loss_jn_holistic}

        if log:
            losses['jn_holistic_error'] = 100 - accuracy(src_logits[idx], target_jn_holistic_o)[0]
        return losses

    def loss_digit_head(self, outputs, targets, indices, num_boxes, log=True):
        """Digit head classification loss (NLL)
        targets dicts must contain the key "digit_head" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_digit_head' in outputs
        src_logits = outputs['pred_digit_head']

        idx = self._get_src_permutation_idx(indices)
        target_digit_head_o = torch.cat([t["digit_head"][J] for t, (_, J) in zip(targets, indices)])
        
        target_digit_head_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_digit_head_onehot[idx[0], idx[1], target_digit_head_o] = 1
        
        loss_digit_head = sigmoid_focal_loss(src_logits, target_digit_head_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_digit_head': loss_digit_head}

        if log:
            losses['digit_head_error'] = 100 - accuracy(src_logits[idx], target_digit_head_o)[0]
        return losses

    def loss_digit_tail(self, outputs, targets, indices, num_boxes, log=True):
        """Digit tail classification loss (NLL)
        targets dicts must contain the key "digit_tail" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_digit_tail' in outputs
        src_logits = outputs['pred_digit_tail']

        idx = self._get_src_permutation_idx(indices)
        target_digit_tail_o = torch.cat([t["digit_tail"][J] for t, (_, J) in zip(targets, indices)])
        
        target_digit_tail_onehot = torch.zeros_like(src_logits, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_digit_tail_onehot[idx[0], idx[1], target_digit_tail_o] = 1
        
        loss_digit_tail = sigmoid_focal_loss(src_logits, target_digit_tail_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_digit_tail': loss_digit_tail}

        if log:
            losses['digit_tail_error'] = 100 - accuracy(src_logits[idx], target_digit_tail_o)[0]
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
                    loss_dict[k] += v
        # Average the loss:
        if loss == "labels" or loss == "boxes" or loss == "masks" or loss == "roles" or loss == "jn_holistic" or loss == "digit_head" or loss == "digit_tail":
            for k in loss_dict.keys():
                loss_dict[k] /= num_boxes
        pass
        return loss_dict
        # return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.detr_loss_batch_len is None:
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = []
            iter_idxs = torch.tensor(
                list(range(0, len(targets))), dtype=torch.int64, device=outputs_without_aux['pred_logits'].device
            )
            for batch_iter_idxs, batch_targets in batch_iterator(
                    self.detr_loss_batch_len, iter_idxs, targets
            ):
                batch_outputs_without_aux = tensor_dict_index_select(outputs_without_aux, batch_iter_idxs, dim=0)
                _ = self.matcher(batch_outputs_without_aux, batch_targets)
                indices += _
                pass

        # batch_len = kwargs["batch_len"]         # HELLORPG Added
        batch_len = self.detr_loss_batch_len
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_distributed():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / distributed_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {"batch_len": batch_len}         # HELLORPG Added
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    kwargs["batch_len"] = batch_len     # HELLORPG Added
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
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

        # Camera loss:
        valid_camera_mask = torch.stack([t["valid_camera"] for t in targets], dim=0)
        
        if valid_camera_mask.any():
            quaternion_gt = torch.stack([t["quaternion"] for t in targets], dim=0)[valid_camera_mask]
            translation_gt = torch.stack([t["translation"] for t in targets], dim=0)[valid_camera_mask]
            fov_hw_gt = torch.stack([t["fov_hw"] for t in targets], dim=0)[valid_camera_mask]
            
            quaternion_pred = outputs["pred_camera"]["quaternion"][valid_camera_mask]
            translation_pred = outputs["pred_camera"]["translation"][valid_camera_mask]
            fov_hw_pred = outputs["pred_camera"]["fov"][valid_camera_mask]
            
            # print('fov_hw_gt', fov_hw_gt)
            # print('fov_hw_pred', fov_hw_pred)
            
            cur_pred_pose_enc = torch.cat([translation_pred, quaternion_pred, fov_hw_pred], dim=-1)
            gt_pose_encoding = torch.cat([translation_gt, quaternion_gt, fov_hw_gt], dim=-1)
            
            loss_T, loss_R, loss_fl = camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="huber")
            losses["loss_T"] = loss_T
            losses["loss_R"] = loss_R
            losses["loss_fl"] = loss_fl
        else:
            losses["loss_T"] = torch.tensor(0.0, device=next(iter(outputs.values())).device)
            losses["loss_R"] = torch.tensor(0.0, device=next(iter(outputs.values())).device)
            losses["loss_fl"] = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        if torch.isnan(losses["loss_T"]):
            print(f"Error: loss_T is nan!")
            # print(f"translation_pred: {translation_pred}")
            # print(f"translation_gt: {translation_gt}")
            exit(0)
        
        if torch.isnan(losses["loss_R"]):
            print(f"Error: loss_R is nan!")
            # print(f"quaternion_pred: {quaternion_pred}")
            # print(f"quaternion_gt: {quaternion_gt}")
            exit(0)
        
        if torch.isnan(losses["loss_fl"]):
            print(f"Error: loss_fl is nan!")
            # print(f"fov_hw_pred: {fov_hw_pred}")
            # print(f"fov_hw_gt: {fov_hw_gt}")
            exit(0)
        
        # Keypoints loss:
        keypoints_gt = torch.stack([t["keypoints_target"] for t in targets], dim=0)
        keypoints_mask = torch.stack([t["keypoints_mask"] for t in targets], dim=0)
        keypoints_pred = outputs["pred_keypoints_heatmap"]
        
        loss_keypoints = F.mse_loss(keypoints_pred, keypoints_gt, reduction='none')
        loss_keypoints = (loss_keypoints * keypoints_mask.unsqueeze(-1).unsqueeze(-1)).sum() / (keypoints_mask.sum() + 1e-6)
        losses["loss_keypoints"] = loss_keypoints
        
        # losses = {k: (v * self.weight_dict[k] if k in self.weight_dict else v) for k, v in losses.items()}

        return losses, self.weight_dict, indices

def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = F.smooth_l1_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3], reduction='none')
        loss_R = F.smooth_l1_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7], reduction='none')
        loss_fl = F.smooth_l1_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:], reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100) # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl

def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """
        
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor

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
            
            # 添加attributes
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
    # 1. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
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
    detr_args.gsr_camera_t_loss_weight = config["GSR_CAMERA_T_LOSS_WEIGHT"]
    detr_args.gsr_camera_r_loss_weight = config["GSR_CAMERA_R_LOSS_WEIGHT"]
    detr_args.gsr_camera_fl_loss_weight = config["GSR_CAMERA_FL_LOSS_WEIGHT"]
    detr_args.num_keypoints = config["NUM_KEYPOINTS"]
    detr_args.gsr_keypoints_loss_weight = config["GSR_KEYPOINTS_LOSS_WEIGHT"]
    detr_args.backbone_strides = [16]
    detr_args.backbone_num_channels = [768]
    
    return detr_args
    
    
def build_deformable_detr_head(config: dict):
    args = cvt_config_to_args(config)
    device = torch.device(args.device)
    
    head = DeformableDetrHead(
        position_encoding = build_position_encoding(args),
        transformer = build_deforamble_transformer(args),
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        backbone_strides=args.backbone_strides,
        backbone_num_channels=args.backbone_num_channels,
        num_keypoints=args.num_keypoints,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        backbone_type=config["BACKBONE_TYPE"],
    )
    return head

def build_deformable_detr_criterion(config: dict):
    args = cvt_config_to_args(config)
    
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    weight_dict['loss_role'] = args.role_loss_coef
    weight_dict['loss_jn_holistic'] = args.jn_loss_coef
    weight_dict['loss_digit_head'] = args.digit_head_loss_coef
    weight_dict['loss_digit_tail'] = args.digit_tail_loss_coef
    weight_dict["loss_T"] = args.gsr_camera_t_loss_weight
    weight_dict["loss_R"] = args.gsr_camera_r_loss_weight
    weight_dict["loss_fl"] = args.gsr_camera_fl_loss_weight
    weight_dict["loss_keypoints"] = args.gsr_keypoints_loss_weight
    
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
    
    detr_criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=build_matcher(args),
        weight_dict=weight_dict,
        losses = ['labels', 'boxes', 'cardinality', 'roles', 'jn_holistic', 'digit_head', 'digit_tail'],
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

class ConvCameraHead(nn.Module):
    def __init__(
        self, 
        input_channels=768,
        trans_act: str = "linear",
        quat_act: str = "linear",
        # fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        fl_act: str = "linear",  # Field of view activations: ensures FOV values are positive.
        ):
        super(ConvCameraHead, self).__init__()
        
        self.input_channels = input_channels
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        
        # Define convolutional layers similar to PoseCNN
        self.convs = {}
        self.convs[0] = nn.Conv2d(input_channels, 256, 7, 2, 3)
        self.convs[1] = nn.Conv2d(256, 256, 5, 2, 2)
        self.convs[2] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[3] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[4] = nn.Conv2d(256, 256, 3, 2, 1)
        
        # Final prediction layer: 4 (quaternion) + 3 (translation) + 2 (fov) = 9
        self.camera_conv = nn.Conv2d(256, 9, 1)
        
        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        
        # Convert to ModuleList for proper parameter registration
        self.net = nn.ModuleList(list(self.convs.values()))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the camera head"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for camera head
        Args:
            x: input features of shape (N, C, H, W)
        Returns:
            quaternion: (N, 4) - camera rotation as quaternion
            translation: (N, 3) - camera translation
            fov: (N, 2) - field of view parameters
        """
        # Apply convolutional layers with ReLU activation
        for i in range(self.num_convs):
            x = self.convs[i](x)
            x = self.relu(x)
        
        # Final prediction layer
        x = self.camera_conv(x)
        
        # Global average pooling to get a single prediction per image
        x = x.mean(3).mean(2)  # Shape: (N, 9)
        
        x = activate_pose(x, self.trans_act, self.quat_act, self.fl_act)
        
        # Split the output into quaternion, translation, and fov
        quaternion = x[:, :4]  # First 4 values
        translation = x[:, 4:7]  # Next 3 values  
        fov = x[:, 7:9]  # Last 2 values
        
        # Normalize quaternion to unit length
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        return quaternion, translation, fov

# class CameraHead(nn.Module):
#     """Simple camera head that works on flattened features"""
#     def __init__(self, dim_in=768):
#         super(CameraHead, self).__init__()
        
#         self.dim_in = dim_in
        
#         # Simple MLP for camera prediction
#         self.camera_mlp = nn.Sequential(
#             nn.Linear(dim_in, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 9)  # 4 quaternion + 3 translation + 2 fov
#         )
        
#         self._init_weights()
    
#     def _init_weights(self):
#         """Initialize weights for the camera head"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, local_features):
#         """
#         Forward pass for camera head
#         Args:
#             local_features: input features of shape (N, L, D) where L is sequence length, D is feature dim
#         Returns:
#             camera_tokens: (N, 9) - concatenated camera parameters
#         """
#         # Global average pooling over the spatial dimension
#         global_features = local_features.mean(dim=1)  # Shape: (N, D)
        
#         # Predict camera parameters
#         camera_params = self.camera_mlp(global_features)  # Shape: (N, 9)
        
#         # Split into components
#         quaternion = camera_params[:, :4]
#         translation = camera_params[:, 4:7] 
#         fov = camera_params[:, 7:9]
        
#         # Normalize quaternion
#         quaternion = F.normalize(quaternion, p=2, dim=1)
        
#         # Scale translation
#         translation = 0.01 * translation
        
#         # Apply sigmoid to fov
#         fov = torch.sigmoid(fov)
        
#         # Concatenate all camera parameters
#         camera_tokens = torch.cat([quaternion, translation, fov], dim=1)
        
#         return camera_tokens
class KeypointsHead(nn.Module):
    def __init__(self, dim_in=768, num_keypoints=58):
        super(KeypointsHead, self).__init__()
        self.dim_in = dim_in
        # Using sub-pixel convolution (pixel shuffle) for learnable upsampling
        # This is more parameter-efficient and often works better than transposed convolution
        
        # Stage 1: (768, 32, 32) -> (192, 64, 64) using 2x upsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),  # 4x channels for 2x upsampling
            nn.PixelShuffle(2),  # (192*4, 32, 32) -> (192, 64, 64)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: (192, 64, 64) -> (96, 128, 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 96 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (96*4, 64, 64) -> (96, 128, 128)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: (96, 128, 128) -> (48, 256, 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 48 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (48*4, 128, 128) -> (48, 256, 256)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # nn.Conv2d(48, 48, kernel_size=3, padding=1),
            # nn.BatchNorm2d(48),
            nn.Conv2d(48, num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_keypoints),
            nn.ReLU(inplace=True)
        )
        
        # # Stage 4: (48, 256, 256) -> (24, 512, 512)
        # self.stage4 = nn.Sequential(
        #     nn.Conv2d(48, 24 * 4, kernel_size=3, padding=1),
        #     nn.PixelShuffle(2),  # (24*4, 256, 256) -> (24, 512, 512)
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(24, 24, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(inplace=True)
        # )
        
        # Final stage: (24, 512, 512) -> (output_channels, 512, 512)
        self.final_conv = nn.Sequential(
            # nn.Conv2d(24, num_keypoints, kernel_size=3, padding=1),
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass using learnable upsampling
        Args:
            x: Input features of shape (N, 768, 32, 32)
        Returns:
            output: Reconstructed features of shape (N, output_channels, 512, 512)
        """
        x = self.stage1(x)      # (N, 192, 64, 64)
        x = self.stage2(x)      # (N, 96, 128, 128)
        x = self.stage3(x)      # (N, 48, 256, 256)
        # x = self.stage4(x)      # (N, 24, 512, 512)
        x = self.final_conv(x)  # (N, output_channels, 512, 512)
        
        return x

class DetectionMetrics(nn.Module):
    """
    计算detection常见的metrics，包括mAP、IoU、precision、recall等指标
    支持多进程聚合和整个数据集上的AP计算
    """
    def __init__(self, num_classes, iou_thresholds=None, score_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds if iou_thresholds is not None else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.score_threshold = score_threshold
        self.postprocess = PostProcess()
        
        # 为每个IoU阈值收集TP/FP/scores和GT数量
        self.tp_fp_scores_per_thresh = {thresh: {'tp': [], 'fp': [], 'scores': []} for thresh in self.iou_thresholds}
        self.total_gt_count = 0
        
        # 为attributes收集匹配结果（只在IoU@0.5时收集）
        self.attribute_matches = {
            'role': {'correct': [], 'total': []},
            'jersey': {'correct': [], 'total': []}, 
            'digit_head': {'correct': [], 'total': []},
            'digit_tail': {'correct': [], 'total': []}
        }
        
        # 为相机参数收集数据
        self.camera_metrics_data = {
            'translation_errors': [],  # 欧氏距离误差
            'rotation_errors': [],     # 角度误差(degrees)
            'fov_errors': [],         # FOV误差
            'valid_count': 0          # 有效样本数量
        }
        
        # 为keypoints收集数据
        self.keypoints_metrics_data = {
            'accuracies': [],     # 准确度
            'precisions': [],     # 精确度
            'recalls': [],        # 召回率
            'f1_scores': [],      # F1分数
            'valid_count': 0      # 有效样本数量
        }
        
    def reset(self):
        """重置收集的数据"""
        self.tp_fp_scores_per_thresh = {thresh: {'tp': [], 'fp': [], 'scores': []} for thresh in self.iou_thresholds}
        self.total_gt_count = 0
        self.attribute_matches = {
            'role': {'correct': [], 'total': []},
            'jersey': {'correct': [], 'total': []}, 
            'digit_head': {'correct': [], 'total': []},
            'digit_tail': {'correct': [], 'total': []}
        }
        self.camera_metrics_data = {
            'translation_errors': [],
            'rotation_errors': [],
            'fov_errors': [],
            'valid_count': 0
        }
        self.keypoints_metrics_data = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'valid_count': 0
        }
        
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

    def quaternion_angular_difference(self, q1, q2):
        """
        计算两个四元数之间的角度差异（以度为单位）
        
        Args:
            q1, q2: 四元数 [N, 4]，格式为 [x, y, z, w]
            
        Returns:
            角度差异（度）[N]
        """
        # 确保四元数是单位四元数
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # 计算四元数点积的绝对值
        dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
        
        # 限制在有效范围内以避免数值误差
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        
        # 计算角度差异（弧度）
        angle_rad = 2 * torch.acos(dot_product)
        
        # 转换为度
        angle_deg = angle_rad * 180.0 / math.pi
        
        return angle_deg

    def compute_camera_metrics(self, pred_camera, targets):
        """
        计算相机参数的metrics
        
        Args:
            pred_camera: 预测的相机参数字典，包含 'quaternion', 'translation', 'fov'
            targets: 目标数据列表
        """
        # 获取有效相机mask
        valid_camera_mask = torch.stack([t.get("valid_camera", torch.tensor(False)) for t in targets], dim=0)
        
        if not valid_camera_mask.any():
            return  # 没有有效的相机数据
        
        # 获取GT相机参数
        quaternion_gt = torch.stack([t["quaternion"] for t in targets], dim=0)[valid_camera_mask]
        translation_gt = torch.stack([t["translation"] for t in targets], dim=0)[valid_camera_mask]
        fov_hw_gt = torch.stack([t["fov_hw"] for t in targets], dim=0)[valid_camera_mask]
        
        # 获取预测相机参数
        quaternion_pred = pred_camera["quaternion"][valid_camera_mask]
        translation_pred = pred_camera["translation"][valid_camera_mask]
        fov_hw_pred = pred_camera["fov"][valid_camera_mask]
        
        # 计算平移误差（欧氏距离）
        translation_errors = torch.norm(translation_pred - translation_gt, dim=-1)  # [N]
        
        # 计算旋转误差（角度差异）
        rotation_errors = self.quaternion_angular_difference(quaternion_pred, quaternion_gt)  # [N]
        
        # 计算FOV误差（L2距离）
        fov_errors = torch.norm(fov_hw_pred - fov_hw_gt, dim=-1)  # [N]
        
        # 转移到CPU并添加到收集器
        self.camera_metrics_data['translation_errors'].extend(translation_errors.cpu().tolist())
        self.camera_metrics_data['rotation_errors'].extend(rotation_errors.cpu().tolist())
        self.camera_metrics_data['fov_errors'].extend(fov_errors.cpu().tolist())
        self.camera_metrics_data['valid_count'] += len(translation_errors)

    def get_keypoints_from_heatmap_batch_maxpool(
            self, 
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 1,
            min_keypoint_pixel_distance: int = 15,
            return_scores: bool = True,
    ):
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling."""
        batch_size, n_channels, height, width = heatmap.shape

        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = min_keypoint_pixel_distance
        
        # exclude border keypoints by padding with highest possible value
        padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
        max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
        
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)

        # moving to CPU
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        
        filtered_indices = []
        for batch_idx in range(batch_size):
            batch_keypoints = []
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                batch_keypoints.append(locs)
            filtered_indices.append(batch_keypoints)

        return torch.tensor(filtered_indices)

    def get_keypoints_from_heatmap_batch_maxpool_l(
            self,
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 2,
            min_keypoint_pixel_distance: int = 10,
            return_scores: bool = True,
    ) -> List[List[List[Tuple[int, int]]]]:
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

        Inspired by mmdetection and CenterNet:
        https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

        Args:
            heatmap (torch.Tensor): NxCxHxW heatmap batch
            max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
            min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

            Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
            abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
            rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

        Returns:
            The extracted keypoints for each batch, channel and heatmap; and their scores
        """
        batch_size, n_channels, _, width = heatmap.shape
        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = int((kernel-1)/2)

        max_pooled_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap

        # all values to zero that are not local maxima
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
        # at this point either score > 0.0, in which case the index is a local maximum
        # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

        #  remove top-k that are not local maxima and threshold (if required)
        # thresholding shouldn't be done during training

        #  moving them to CPU now to avoid multiple GPU-mem accesses!
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
        filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

        # have to do this manually as the number of maxima for each channel can be different
        for batch_idx in range(batch_size):
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                filtered_indices[batch_idx][channel_idx] = locs

        return torch.tensor(filtered_indices)

    def calculate_keypoints_metrics(self, gt, pred, mask, conf_th=0.1, dist_th=5):
        """计算keypoints的metrics"""
        # Convert mask to geometry mask (excluding last channel if needed)
        geometry_mask = (mask > 0).cpu()
            
        # Ensure gt and pred are on CPU for computation
        gt = gt.cpu()
        pred = pred.cpu()
        
        batch_size = gt.shape[0]
        batch_metrics = []
        
        for batch_idx in range(batch_size):
            if not geometry_mask[batch_idx].any():
                # No valid keypoints in this sample
                batch_metrics.append((0.0, 0.0, 0.0, 0.0))
                continue
                
            # Get valid keypoints for this batch
            valid_mask = geometry_mask[batch_idx]
            
            # Extract positions and confidence scores
            gt_batch = gt[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            pred_batch = pred[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            
            # Check confidence thresholds
            gt_conf_mask = gt_batch[:, -1] > conf_th  # GT confidence > threshold
            pred_conf_mask = pred_batch[:, -1] > conf_th  # Pred confidence > threshold
            
            # Calculate distances between predicted and GT positions
            gt_pos = gt_batch[:, :2]  # [valid_kp, 2] (x, y)
            pred_pos = pred_batch[:, :2]  # [valid_kp, 2] (x, y)
            distances = torch.norm(pred_pos - gt_pos, dim=1)  # [valid_kp]
            
            # Count true positives, false positives, and false negatives
            true_positives = ((distances < dist_th) & pred_conf_mask & gt_conf_mask).sum().item()
            true_negatives = (~pred_conf_mask & ~gt_conf_mask).sum().item()
            false_positives = ((pred_conf_mask & ~gt_conf_mask) | ((distances >= dist_th) & pred_conf_mask & gt_conf_mask)).sum().item()
            false_negatives = (~pred_conf_mask & gt_conf_mask).sum().item()
            
            # Calculate metrics
            total_valid = valid_mask.sum().item()
            if total_valid > 0:
                accuracy = (true_positives + true_negatives) / total_valid
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                accuracy = precision = recall = f1 = 0.0
                
            batch_metrics.append((accuracy, precision, recall, f1))
        
        return batch_metrics

    def compute_keypoints_metrics(self, pred_keypoints_heatmap, targets):
        """
        计算keypoints的metrics
        
        Args:
            pred_keypoints_heatmap: 预测的keypoints heatmap [B, num_keypoints, H, W]
            targets: 目标数据列表
        """
        # 检查是否有keypoints数据
        if pred_keypoints_heatmap is None:
            return
            
        # 获取GT keypoints heatmap和mask
        try:
            keypoints_gt_list = [t.get("keypoints_target", None) for t in targets]
            keypoints_mask_list = [t.get("keypoints_mask", None) for t in targets]
            
            # 过滤掉None值
            valid_indices = [i for i, (kp_gt, kp_mask) in enumerate(zip(keypoints_gt_list, keypoints_mask_list)) 
                           if kp_gt is not None and kp_mask is not None]
            
            if not valid_indices:
                return  # 没有有效的keypoints数据
            
            # 只处理有效的数据
            keypoints_gt = torch.stack([keypoints_gt_list[i] for i in valid_indices])
            keypoints_mask = torch.stack([keypoints_mask_list[i] for i in valid_indices])
            pred_keypoints_valid = pred_keypoints_heatmap[valid_indices]
            
            # 从heatmap中提取keypoints
            kp_gt = self.get_keypoints_from_heatmap_batch_maxpool(keypoints_gt[:,:-1,:,:], return_scores=True, max_keypoints=1)
            kp_pred = self.get_keypoints_from_heatmap_batch_maxpool(pred_keypoints_valid[:,:-1,:,:], return_scores=True, max_keypoints=1)
            
            # 计算metrics
            batch_metrics = self.calculate_keypoints_metrics(kp_gt, kp_pred, keypoints_mask[:, :-1])
            
            # 收集metrics
            for accuracy, precision, recall, f1 in batch_metrics:
                self.keypoints_metrics_data['accuracies'].append(accuracy)
                self.keypoints_metrics_data['precisions'].append(precision)
                self.keypoints_metrics_data['recalls'].append(recall)
                self.keypoints_metrics_data['f1_scores'].append(f1)
            
            self.keypoints_metrics_data['valid_count'] += len(batch_metrics)
            
        except Exception as e:
            # 如果keypoints计算失败，静默跳过
            # print(f"Warning: Keypoints metrics calculation failed: {e}")
            pass
        
    def update(self, outputs, targets, target_sizes):
        """
        在当前batch上计算TP/FP并收集结果
        
        Args:
            outputs: 模型输出 
            targets: 真实标注
            target_sizes: 图像尺寸
        """
        device = outputs['pred_logits'].device
        
        # 使用PostProcess获取预测结果（已包含attributes）
        predictions = self.postprocess(outputs, target_sizes)
        
        # 为每个IoU阈值计算TP/FP
        for iou_thresh in self.iou_thresholds:
            tp_list = []
            fp_list = []
            scores_list = []
            
            # 处理当前batch中的每个sample
            for sample_idx, (pred, target, target_size) in enumerate(zip(predictions, targets, target_sizes)):
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
                        # 找到与当前预测同类别的GT
                        same_class_mask = (gt_labels == pred_label)
                        if not same_class_mask.any():
                            fp_list.append(1)
                            tp_list.append(0)
                        else:
                            # 在同类别GT中找到IoU最大的
                            class_ious = ious[i] * same_class_mask.float()
                            max_iou, max_idx = torch.max(class_ious, dim=0)
                            
                            if max_iou >= iou_thresh and not gt_matched[max_idx]:
                                tp_list.append(1)
                                fp_list.append(0)
                                gt_matched[max_idx] = True
                                
                                # 只在IoU@0.5时计算attributes准确度
                                if iou_thresh == 0.5:
                                    self._compute_attribute_accuracy(pred, target, i, max_idx.item())
                            else:
                                tp_list.append(0)
                                fp_list.append(1)
                        
                        scores_list.append(pred_score.cpu().item())  # 转到CPU
                else:
                    # 没有GT，所有预测都是FP
                    fp_list.extend([1] * len(pred_boxes))
                    tp_list.extend([0] * len(pred_boxes))
                    scores_list.extend(pred_scores.cpu().tolist())  # 转到CPU
            
            # 将当前batch的结果添加到对应IoU阈值的收集器中
            self.tp_fp_scores_per_thresh[iou_thresh]['tp'].extend(tp_list)
            self.tp_fp_scores_per_thresh[iou_thresh]['fp'].extend(fp_list)
            self.tp_fp_scores_per_thresh[iou_thresh]['scores'].extend(scores_list)
        
        # 统计GT数量
        batch_gt_count = sum(len(target['labels']) for target in targets)
        self.total_gt_count += batch_gt_count
        
        # 计算相机metrics（如果有相机预测）
        if 'pred_camera' in outputs:
            self.compute_camera_metrics(outputs['pred_camera'], targets)
        
        # 计算keypoints metrics（如果有keypoints预测）
        if 'pred_keypoints_heatmap' in outputs:
            self.compute_keypoints_metrics(outputs['pred_keypoints_heatmap'], targets)
            
    def _compute_attribute_accuracy(self, pred, target, pred_idx, gt_idx):
        """
        计算匹配成功的预测的attribute准确度
        
        Args:
            pred: 单个样本的预测结果（来自PostProcess，已包含attributes）
            target: 单个样本的真实标注
            pred_idx: 预测框的索引
            gt_idx: 匹配的GT框的索引
        """
        # 获取GT的attributes
        gt_roles = target.get('roles', None)
        gt_jersey = target.get('jersey', None)
        gt_digit_head = target.get('digit_head', None) 
        gt_digit_tail = target.get('digit_tail', None)
        
        # 计算role准确度
        if 'roles' in pred and gt_roles is not None and gt_idx < len(gt_roles):
            pred_role = pred['roles'][pred_idx].item()
            gt_role = gt_roles[gt_idx].item() if isinstance(gt_roles[gt_idx], torch.Tensor) else gt_roles[gt_idx]
            self.attribute_matches['role']['correct'].append(1 if pred_role == gt_role else 0)
            self.attribute_matches['role']['total'].append(1)
        
        # 计算jersey准确度
        if 'jersey' in pred and gt_jersey is not None and gt_idx < len(gt_jersey):
            pred_jn = pred['jersey'][pred_idx].item()
            gt_jn = gt_jersey[gt_idx].item() if isinstance(gt_jersey[gt_idx], torch.Tensor) else gt_jersey[gt_idx]
            self.attribute_matches['jersey']['correct'].append(1 if pred_jn == gt_jn else 0)
            self.attribute_matches['jersey']['total'].append(1)
        
        # 计算digit_head准确度
        if 'digit_head' in pred and gt_digit_head is not None and gt_idx < len(gt_digit_head):
            pred_dh = pred['digit_head'][pred_idx].item()
            gt_dh = gt_digit_head[gt_idx].item() if isinstance(gt_digit_head[gt_idx], torch.Tensor) else gt_digit_head[gt_idx]
            self.attribute_matches['digit_head']['correct'].append(1 if pred_dh == gt_dh else 0)
            self.attribute_matches['digit_head']['total'].append(1)
        
        # 计算digit_tail准确度
        if 'digit_tail' in pred and gt_digit_tail is not None and gt_idx < len(gt_digit_tail):
            pred_dt = pred['digit_tail'][pred_idx].item()
            gt_dt = gt_digit_tail[gt_idx].item() if isinstance(gt_digit_tail[gt_idx], torch.Tensor) else gt_digit_tail[gt_idx]
            self.attribute_matches['digit_tail']['correct'].append(1 if pred_dt == gt_dt else 0)
            self.attribute_matches['digit_tail']['total'].append(1)

    def gather_tp_fp_scores(self, accelerator):
        """
        在所有进程间聚合TP/FP/scores结果、attribute匹配结果、相机metrics数据和keypoints metrics数据
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            gathered_tp_fp_scores_per_thresh, gathered_total_gt_count, gathered_attribute_matches, gathered_camera_metrics, gathered_keypoints_metrics
        """
        # 聚合每个IoU阈值的TP/FP/scores
        gathered_tp_fp_scores = {}
        key_list = ['tp', 'fp', 'scores']
        for thresh in self.iou_thresholds:
            gathered_tp_fp_scores[thresh] = {}
            for key in key_list:
                gathered_tp_fp_scores[thresh][key] = accelerator.gather_for_metrics(self.tp_fp_scores_per_thresh[thresh][key])
        
        # 聚合GT总数（需要包装成列表）
        gathered_gt_count = accelerator.gather_for_metrics([self.total_gt_count])
        
        # 聚合attribute匹配结果
        attr_name_list = ['role', 'jersey', 'digit_head', 'digit_tail']
        key_list_attr = ['correct', 'total']
        gathered_attribute_matches = {}
        for attr_name in attr_name_list:
            gathered_attribute_matches[attr_name] = {}
            for key in key_list_attr:
                gathered_attribute_matches[attr_name][key] = accelerator.gather_for_metrics(self.attribute_matches[attr_name][key])
        
        # 聚合相机metrics数据
        camera_key_list = ['translation_errors', 'rotation_errors', 'fov_errors']
        gathered_camera_metrics = {}
        for key in camera_key_list:
            gathered_camera_metrics[key] = accelerator.gather_for_metrics(self.camera_metrics_data[key])
        gathered_camera_metrics['valid_count'] = accelerator.gather_for_metrics([self.camera_metrics_data['valid_count']])
        
        # 聚合keypoints metrics数据
        keypoints_key_list = ['accuracies', 'precisions', 'recalls', 'f1_scores']
        gathered_keypoints_metrics = {}
        for key in keypoints_key_list:
            gathered_keypoints_metrics[key] = accelerator.gather_for_metrics(self.keypoints_metrics_data[key])
        gathered_keypoints_metrics['valid_count'] = accelerator.gather_for_metrics([self.keypoints_metrics_data['valid_count']])
        
        return gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, gathered_camera_metrics, gathered_keypoints_metrics

    def compute_metrics_from_gathered_tp_fp(self, gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches=None, gathered_camera_metrics=None, gathered_keypoints_metrics=None):
        """
        从聚合的TP/FP/scores数据计算metrics
        
        Args:
            gathered_tp_fp_scores: 聚合的TP/FP/scores数据
            gathered_gt_count: 聚合的GT总数
            gathered_attribute_matches: 聚合的attribute匹配结果
            gathered_camera_metrics: 聚合的相机metrics数据
            gathered_keypoints_metrics: 聚合的keypoints metrics数据
            
        Returns:
            dict: 包含各种metrics的字典
        """
        # 处理不同的数据结构
        def flatten_data(data):
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result
            else:
                return data if isinstance(data, list) else [data]
        
        # 初始化metrics
        metrics = {}
        
        # 为每个IoU阈值计算metrics
        for iou_thresh in self.iou_thresholds:
            thresh_data = gathered_tp_fp_scores[iou_thresh]
            
            # 展平所有进程的数据
            all_tp = []
            all_fp = []
            all_scores = []
            
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
        
        # 计算attribute准确度
        if gathered_attribute_matches is not None:
            for attr_name in ['role', 'jersey', 'digit_head', 'digit_tail']:
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
        
        # 计算相机metrics
        if gathered_camera_metrics is not None:
            # 展平所有进程的相机数据
            all_translation_errors = flatten_data(gathered_camera_metrics['translation_errors'])
            all_rotation_errors = flatten_data(gathered_camera_metrics['rotation_errors'])
            all_fov_errors = flatten_data(gathered_camera_metrics['fov_errors'])
            
            # 计算总的有效样本数
            total_valid_count = sum(gathered_camera_metrics['valid_count'])
            
            if total_valid_count > 0:
                # 计算平移误差统计
                translation_errors = torch.tensor(all_translation_errors, dtype=torch.float32)
                metrics['camera_translation_mae'] = translation_errors.mean().item()  # 平均绝对误差
                metrics['camera_translation_rmse'] = torch.sqrt(translation_errors.pow(2).mean()).item()  # 均方根误差
                metrics['camera_translation_median'] = translation_errors.median().item()  # 中位数误差
                
                # 计算旋转误差统计
                rotation_errors = torch.tensor(all_rotation_errors, dtype=torch.float32)
                metrics['camera_rotation_mae'] = rotation_errors.mean().item()  # 平均绝对角度误差(度)
                metrics['camera_rotation_rmse'] = torch.sqrt(rotation_errors.pow(2).mean()).item()  # 均方根角度误差
                metrics['camera_rotation_median'] = rotation_errors.median().item()  # 中位数角度误差
                
                # 计算FOV误差统计
                fov_errors = torch.tensor(all_fov_errors, dtype=torch.float32)
                metrics['camera_fov_mae'] = fov_errors.mean().item()  # 平均绝对FOV误差
                metrics['camera_fov_rmse'] = torch.sqrt(fov_errors.pow(2).mean()).item()  # 均方根FOV误差
                metrics['camera_fov_median'] = fov_errors.median().item()  # 中位数FOV误差
                
                # 记录样本数量
                metrics['camera_valid_samples'] = total_valid_count
                
                # 计算精度阈值内的准确度
                # 平移误差 < 1.0 的比例
                translation_acc_1 = (translation_errors < 1.0).float().mean().item()
                metrics['camera_translation_acc@1.0'] = translation_acc_1
                
                # 旋转误差 < 5度的比例
                rotation_acc_5 = (rotation_errors < 5.0).float().mean().item()
                metrics['camera_rotation_acc@5deg'] = rotation_acc_5
                
                # 旋转误差 < 10度的比例
                rotation_acc_10 = (rotation_errors < 10.0).float().mean().item()
                metrics['camera_rotation_acc@10deg'] = rotation_acc_10
            else:
                # 没有有效的相机数据
                for metric_name in ['camera_translation_mae', 'camera_translation_rmse', 'camera_translation_median',
                                  'camera_rotation_mae', 'camera_rotation_rmse', 'camera_rotation_median', 
                                  'camera_fov_mae', 'camera_fov_rmse', 'camera_fov_median',
                                  'camera_translation_acc@1.0', 'camera_rotation_acc@5deg', 'camera_rotation_acc@10deg']:
                    metrics[metric_name] = 0.0
                metrics['camera_valid_samples'] = 0
        
        # 计算keypoints metrics
        if gathered_keypoints_metrics is not None:
            # 展平所有进程的keypoints数据
            all_accuracies = flatten_data(gathered_keypoints_metrics['accuracies'])
            all_precisions = flatten_data(gathered_keypoints_metrics['precisions'])
            all_recalls = flatten_data(gathered_keypoints_metrics['recalls'])
            all_f1_scores = flatten_data(gathered_keypoints_metrics['f1_scores'])
            
            # 计算总的有效样本数
            total_valid_count = sum(gathered_keypoints_metrics['valid_count'])
            
            if total_valid_count > 0 and len(all_accuracies) > 0:
                # 计算keypoints metrics的平均值
                accuracies = torch.tensor(all_accuracies, dtype=torch.float32)
                precisions = torch.tensor(all_precisions, dtype=torch.float32)
                recalls = torch.tensor(all_recalls, dtype=torch.float32)
                f1_scores = torch.tensor(all_f1_scores, dtype=torch.float32)
                
                metrics['keypoints_accuracy'] = accuracies.mean().item()
                metrics['keypoints_precision'] = precisions.mean().item()
                metrics['keypoints_recall'] = recalls.mean().item()
                metrics['keypoints_f1'] = f1_scores.mean().item()
                
                # 计算中位数
                metrics['keypoints_accuracy_median'] = accuracies.median().item()
                metrics['keypoints_precision_median'] = precisions.median().item()
                metrics['keypoints_recall_median'] = recalls.median().item()
                metrics['keypoints_f1_median'] = f1_scores.median().item()
                
                # 计算高精度阈值下的性能
                # 精度 > 0.8 的比例
                high_acc_ratio = (accuracies > 0.8).float().mean().item()
                metrics['keypoints_high_accuracy_ratio'] = high_acc_ratio
                
                # F1 > 0.7 的比例
                high_f1_ratio = (f1_scores > 0.7).float().mean().item()
                metrics['keypoints_high_f1_ratio'] = high_f1_ratio
                
                # 记录样本数量
                metrics['keypoints_valid_samples'] = total_valid_count
            else:
                # 没有有效的keypoints数据
                for metric_name in ['keypoints_accuracy', 'keypoints_precision', 'keypoints_recall', 'keypoints_f1',
                                  'keypoints_accuracy_median', 'keypoints_precision_median', 'keypoints_recall_median', 'keypoints_f1_median',
                                  'keypoints_high_accuracy_ratio', 'keypoints_high_f1_ratio']:
                    metrics[metric_name] = 0.0
                metrics['keypoints_valid_samples'] = 0
        
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
        # 聚合所有进程的TP/FP/scores结果、attribute匹配结果、相机metrics数据和keypoints metrics数据
        gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, gathered_camera_metrics, gathered_keypoints_metrics = self.gather_tp_fp_scores(accelerator)
        
        # 只在主进程计算metrics
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_tp_fp(gathered_tp_fp_scores, gathered_gt_count, gathered_attribute_matches, gathered_camera_metrics, gathered_keypoints_metrics)
        else:
            return {}


def build_detection_metrics(config: dict):
    """
    构建detection metrics计算器
    """
    num_classes = config["NUM_CLASSES"]
    
    metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        score_threshold=config.get("EVAL_SCORE_THRESHOLD", 0.5)
    )
    
    return metrics


