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
from torch import nn, Tensor, Callable
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
from data.SoccerNetGSR_ReID import role_mapping

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDetrHead(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, position_encoding, transformer, num_classes, num_queries, num_feature_levels, backbone_strides, backbone_num_channels, 
                 aux_loss=True, with_box_refine=False, two_stage=False):
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
        self.role_embed = nn.Linear(hidden_dim, num_role_classes)
        self.num_role_classes = num_role_classes
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
        
        self.camera_head = ConvCameraHead(input_channels=backbone_num_channels[0])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.role_embed.bias.data = torch.ones(num_role_classes) * bias_value
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
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.role_embed = nn.ModuleList([self.role_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, backbone_outputs):
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
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_role = self.role_embed[lvl](hs[lvl])
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
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_role = torch.stack(outputs_roles)

        # Use ConvCameraHead with reshaped features
        quaternion, translation, fov = self.camera_head(reshaped_local_features)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_roles': outputs_role[-1]}
        # Add camera predictions to output
        out['pred_camera'] = {
            'quaternion': quaternion,
            'translation': translation,
            'fov': fov
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_role)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        # Output the outputs of last decoder layer.
        # We need these outputs to generate the embeddings for objects.
        out["outputs"] = hs[-1]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_role):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_roles': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_role[:-1])]


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
        target_roles = torch.full(src_logits.shape[:2], self.num_role_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_roles[idx] = target_roles_o

        target_roles_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_roles_onehot.scatter_(2, target_roles.unsqueeze(-1), 1)

        target_roles_onehot = target_roles_onehot[:,:,:-1]
        loss_role = sigmoid_focal_loss(src_logits, target_roles_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_role': loss_role}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['role_error'] = 100 - accuracy(src_logits[idx], target_roles_o)[0]
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
            'roles': self.loss_roles
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
        if loss == "labels" or loss == "boxes" or loss == "masks" or loss == "roles":
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

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

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
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
    detr_args.gsr_camera_t_loss_weight = config["GSR_CAMERA_T_LOSS_WEIGHT"]
    detr_args.gsr_camera_r_loss_weight = config["GSR_CAMERA_R_LOSS_WEIGHT"]
    detr_args.gsr_camera_fl_loss_weight = config["GSR_CAMERA_FL_LOSS_WEIGHT"]
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
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    return head

def build_deformable_detr_criterion(config: dict):
    args = cvt_config_to_args(config)
    
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    # Add role loss coefficient
    weight_dict['loss_role'] = args.role_loss_coef
    # Add camera loss weights
    weight_dict["loss_T"] = args.gsr_camera_t_loss_weight
    weight_dict["loss_R"] = args.gsr_camera_r_loss_weight
    weight_dict["loss_fl"] = args.gsr_camera_fl_loss_weight
    
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
        losses = ['labels', 'boxes', 'cardinality', 'roles'],
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