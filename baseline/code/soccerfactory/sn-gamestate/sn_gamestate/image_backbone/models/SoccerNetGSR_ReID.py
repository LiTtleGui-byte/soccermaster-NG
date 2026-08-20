from __future__ import division, absolute_import

import torch
import torch.nn as nn
from collections import OrderedDict
from torchmetrics import Accuracy
from prtreid.losses import init_part_based_triplet_loss, CrossEntropyLoss
from prtreid.utils.constants import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS
import torch.nn.functional as F
import math
from data.SoccerNetGSR_ReID import role_mapping, jn_mapping, digit_head_mapping, digit_tail_mapping
from models.deformable_detr.deformable_detr import MLP
import warnings

class SoccerNetGSR_ReIDHead(nn.Module):
    def __init__(self, backbone_num_channels, output_reid_dim, num_pids, backbone_type='image'):
        super().__init__()
        self.backbone_num_channels = backbone_num_channels
        self.backbone_type = backbone_type
        self.global_feature_proj = MLP(backbone_num_channels, backbone_num_channels, output_reid_dim, 2)
        self.role_classifier = MLP(output_reid_dim, output_reid_dim, len(role_mapping), 2)
        self.pid_classifier = MLP(output_reid_dim, output_reid_dim, num_pids, 2)
        self.jn_holistic_classifier = MLP(output_reid_dim, output_reid_dim, len(jn_mapping), 2)
        self.digit_head_classifier = MLP(output_reid_dim, output_reid_dim, len(digit_head_mapping), 2)
        self.digit_tail_classifier = MLP(output_reid_dim, output_reid_dim, len(digit_tail_mapping), 2)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        global_features, local_features = backbone_outputs['global_features'], backbone_outputs['local_features']
        if self.backbone_type == 'video':
            global_features = global_features[:, 0]
            local_features = local_features[:, 0]
        
        reid_embeddings = self.global_feature_proj(global_features)
        
        role_logits = self.role_classifier(reid_embeddings)
        pid_logits = self.pid_classifier(reid_embeddings)
        jn_holistic_logits = self.jn_holistic_classifier(reid_embeddings)
        digit_head_logits = self.digit_head_classifier(reid_embeddings)
        digit_tail_logits = self.digit_tail_classifier(reid_embeddings)
        
        out = {'reid_embeddings': reid_embeddings, 'role_logits': role_logits, 'pid_logits': pid_logits, 'jn_holistic_logits': jn_holistic_logits, 'digit_head_logits': digit_head_logits, 'digit_tail_logits': digit_tail_logits}
        
        return out

class SoccerNetGSR_ReIDLoss(nn.Module):
    def __init__(self, weight_dict, margin=0.3, epsilon=1e-16):
        super().__init__()
        self.weight_dict = weight_dict
        self.epsilon = epsilon
        self.margin = margin
        self.role_loss = FocalLoss()
        self.pid_loss = FocalLoss()
        self.jn_holistic_loss = FocalLoss()
        self.digit_head_loss = FocalLoss()
        self.digit_tail_loss = FocalLoss()
        
    def forward(self, outputs, targets):
        losses = {}
        role_logits, pid_logits, jn_holistic_logits, digit_head_logits, digit_tail_logits = outputs['role_logits'], outputs['pid_logits'], outputs['jn_holistic_logits'], outputs['digit_head_logits'], outputs['digit_tail_logits']
        role_loss = self.role_loss(role_logits, targets['role'])
        pid_loss = self.pid_loss(pid_logits, targets['pid'])
        jn_holistic_loss = self.jn_holistic_loss(jn_holistic_logits, targets['jn_holistic'])
        digit_head_loss = self.digit_head_loss(digit_head_logits, targets['digit_head'])
        digit_tail_loss = self.digit_tail_loss(digit_tail_logits, targets['digit_tail'])
        losses['role_focal_loss'] = role_loss
        losses['pid_focal_loss'] = pid_loss
        losses['jn_holistic_focal_loss'] = jn_holistic_loss
        losses['digit_head_focal_loss'] = digit_head_loss
        losses['digit_tail_focal_loss'] = digit_tail_loss
        
        # Compute the pairwise distance matrix
        reid_embeddings = outputs['reid_embeddings']
        pairwise_dist = self._pairwise_distance_matrix(reid_embeddings, squared=False).unsqueeze(0) # [1, N, N]
        pid_triplet_loss, pid_trivial_triplets_ratio, pid_valid_triplets_ratio = self._hard_mine_triplet_loss(pairwise_dist, targets['pid'], margin=self.margin)
        team_triplet_loss, team_trivial_triplets_ratio, team_valid_triplets_ratio = self._hard_mine_triplet_loss(pairwise_dist, targets['team'], margin=self.margin)
        losses['pid_triplet_loss'] = pid_triplet_loss
        losses['team_triplet_loss'] = team_triplet_loss
        losses['pid_trivial_triplets_ratio'] = pid_trivial_triplets_ratio
        losses['team_trivial_triplets_ratio'] = team_trivial_triplets_ratio
        losses['pid_valid_triplets_ratio'] = pid_valid_triplets_ratio
        losses['team_valid_triplets_ratio'] = team_valid_triplets_ratio
        
        return losses, self.weight_dict

    def _pairwise_distance_matrix(self, embeddings, squared=False):
        """
        embeddings.shape = (N, C)
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        N, C = embeddings.shape
        
        dot_product = torch.matmul(embeddings, embeddings.transpose(1, 0))
        square_sum = dot_product.diagonal(dim1=0, dim2=1)
        distances = square_sum.unsqueeze(1) - 2 * dot_product + square_sum.unsqueeze(0)
        distances = F.relu(distances)

        if not squared:
            mask = torch.eq(distances, 0).float()
            distances = distances + mask * self.epsilon  # for numerical stability (infinite derivative of sqrt in 0)
            distances = torch.sqrt(distances)
            distances = distances * (1 - mask)

        return distances

    def _hard_mine_triplet_loss(self, batch_pairwise_dist, labels, margin):
        """
        A generic implementation of the batch-hard triplet loss.
        K (part-based) distance matrix between N samples are provided in tensor 'batch_pairwise_dist' of size [K, N, N].
        The standard batch-hard triplet loss is then computed for each of the K distance matrix, yielding a total of KxN
        triplet losses.
        When a pairwise distance matrix of size [1, N, N] is provided with K=1, this function behave like a standard
        batch-hard triplet loss.
        When a pairwise distance matrix of size [K, N, N] is provided, this function will apply the batch-hard triplet
        loss strategy K times, i.e. one time for each of the K part-based distance matrix. It will then average all
        KxN triplet losses for all K parts into one loss value.
        For the part-averaged triplet loss described in the paper, all part-based distance are first averaged before
        calling this function, and a pairwise distance matrix of size [1, N, N] is provided here.
        When the triplet loss is applied individually for each part, without considering the global/combined distance
        between two training samples (as implemented by 'PartIndividualTripletLoss'), then a (part-based) pairwise
        distance matrix of size [K, N, N] is given as input.
        Compute distance matrix; i.e. for each anchor a_i with i=range(0, batch_size) :
        - find the (a_i,p_i) pair with greatest distance s.t. a_i and p_i have the same label
        - find the (a_i,n_i) pair with smallest distance s.t. a_i and n_i have different label
        - compute triplet loss for each triplet (a_i, p_i, n_i), average them
        Source :
        - https://github.com/lyakaap/NetVLAD-pytorch/blob/master/hard_triplet_loss.py
        - https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/triplet_loss.py
        Args:
            batch_pairwise_dist: pairwise distances between samples, of size (K, N, N). A value of -1 means no distance
                could be computed between the two sample, that pair should therefore not be considered for triplet
                mining.
            labels: id labels for the batch, of size (N,)
        Returns:
            triplet_loss: scalar tensor containing the batch hard triplet loss, which is the result of the average of a
                maximum of KxN triplet losses. Triplets are generated for anchors with at least one valid negative and
                one valid positive. Invalid negatives and invalid positives are marked with a -1 distance in
                batch_pairwise_dist input tensor.
            trivial_triplets_ratio: scalar between [0, 1] indicating the ratio of hard triplets that are 'trivial', i.e.
                for which the triplet loss value is 0 because the margin condition is already satisfied.
            valid_triplets_ratio: scalar between [0, 1] indicating the ratio of hard triplets that are valid. A triplet 
                is invalid if the anchor could not be compared with any positive or negative sample. Two samples cannot 
                be compared if they have no mutually visible parts (therefore no distance could be computed).
        """
        max_value = torch.finfo(batch_pairwise_dist.dtype).max

        valid_pairwise_dist_mask = (batch_pairwise_dist != float(-1))

        # Get the hardest positive pairs
        # invalid positive distance were set to -1 to
        mask_anchor_positive = self._get_anchor_positive_mask(labels).unsqueeze(0)
        mask_anchor_positive = mask_anchor_positive * valid_pairwise_dist_mask
        valid_positive_dist = batch_pairwise_dist * mask_anchor_positive.float() - (~mask_anchor_positive).float()
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=-1)  # [K, N]

        # Get the hardest negative pairs
        mask_anchor_negative = self._get_anchor_negative_mask(labels).unsqueeze(0)
        mask_anchor_negative = mask_anchor_negative * valid_pairwise_dist_mask
        valid_negative_dist = batch_pairwise_dist * mask_anchor_negative.float() + (~mask_anchor_negative).float() * max_value
        hardest_negative_dist, _ = torch.min(valid_negative_dist, dim=-1)  # [K, N]

        # Hardest negative/positive with dist=float.max/-1 are invalid: no valid negative/positive found for this anchor
        # Do not generate triplet for such anchor
        valid_hardest_positive_dist_mask = hardest_positive_dist != -1
        valid_hardest_negative_dist_mask = hardest_negative_dist != max_value
        valid_triplets_mask = valid_hardest_positive_dist_mask * valid_hardest_negative_dist_mask  # [K, N]
        hardest_dist = torch.stack([hardest_positive_dist, hardest_negative_dist], 2)  # [K, N, 2]
        valid_hardest_dist = hardest_dist[valid_triplets_mask, :]  # [K*N, 2]

        if valid_hardest_dist.nelement() == 0:
            warnings.warn("CRITICAL WARNING: no valid triplets were generated for current batch")
            return None

        # Build valid triplets and compute triplet loss
        if self.margin > 0:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.hard_margin_triplet_loss(margin, valid_hardest_dist, valid_triplets_mask)
        else:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.soft_margin_triplet_loss(0.3, valid_hardest_dist, valid_triplets_mask)

        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def hard_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def soft_margin_triplet_loss(self, margin, valid_hardest_dist, valid_triplets_mask):
        triplet_losses = F.relu(valid_hardest_dist[:, 0] - valid_hardest_dist[:, 1] + margin)
        hard_margin_triplet_loss = torch.mean(triplet_losses)
        trivial_triplets_ratio = (triplet_losses == 0.).sum() / triplet_losses.nelement()
        valid_triplets_ratio = valid_triplets_mask.sum() / valid_triplets_mask.nelement()

        # valid_hardest_dist[:, 0] = hardest positive dist
        # valid_hardest_dist[:, 1] = hardest negative dist
        y = valid_hardest_dist[:, 0].new().resize_as_(valid_hardest_dist[:, 0]).fill_(1)
        soft_margin_triplet_loss = F.soft_margin_loss(valid_hardest_dist[:, 1] - valid_hardest_dist[:, 0], y)
        if soft_margin_triplet_loss == float('Inf'):
            print("soft_margin_triplet_loss = inf")
            return hard_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio
        return soft_margin_triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    @staticmethod
    def _get_anchor_positive_mask(labels):
        """
        To be a valid positive pair (a,p) :
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=(labels.get_device() if labels.is_cuda else None))
        indices_not_equal_mask = ~indices_equal_mask

        # Check if labels[i] == labels[j]
        labels_equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

        mask_anchor_positive = indices_not_equal_mask * labels_equal_mask

        return mask_anchor_positive

    @staticmethod
    def _get_anchor_negative_mask(labels):
        """
        To be a valid negative pair (a,n) :
            - a and n have different labels (and therefore are different embeddings)
        """

        # Check if labels[i] != labels[k]
        labels_not_equal_mask = torch.ne(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

        return labels_not_equal_mask

class GiLtLoss(nn.Module):
    """ The Global-identity Local-triplet 'GiLt' loss as described in our paper:
    'Somers V. & al, Body Part-Based Representation Learning for Occluded Person Re-Identification, WACV23'.
    Source: https://github.com/VlSomers/bpbreid
    The default weights for the GiLt strategy (as described in the paper) are provided in 'default_losses_weights': the
    identity loss is applied only on holistic embeddings and the triplet loss is applied only on part-based embeddings.
    'tr' denotes 'triplet' for the triplet loss and 'id' denotes 'identity' for the identity cross-entropy loss.
    """

    default_losses_weights = {
        GLOBAL: {'id': 1., 'tr': 0.},
        FOREGROUND: {'id': 1., 'tr': 0.},
        CONCAT_PARTS: {'id': 1., 'tr': 0.},
        PARTS: {'id': 0., 'tr': 1.}
    }

    def __init__(self,
                 losses_weights=None,
                 use_visibility_scores=False,
                 triplet_margin=0.3,
                 loss_name='part_averaged_triplet_loss',
                 use_gpu=False,
                 writer=None):
        super().__init__()
        if losses_weights is None:
            losses_weights = self.default_losses_weights
        self.pred_accuracy = Accuracy(top_k=1)
        if use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()
        self.losses_weights = losses_weights
        self.part_triplet_loss = init_part_based_triplet_loss(loss_name, margin=triplet_margin, writer=writer)
        self.identity_loss = CrossEntropyLoss(label_smooth=True)
        self.use_visibility_scores = use_visibility_scores

    def forward(self, embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids, mode=0): # mode=0 -> reid, mode=1 -> team_aff
        """
        Keys in the input dictionaries are from {'globl', 'foreg', 'conct', 'parts'} and correspond to the different
        types of embeddings. In the documentation below, we denote the batch size by 'N' and the number of parts by 'K'.
        :param embeddings_dict: a dictionary of embeddings, where the keys are the embedding types and the values are
            Tensors of size [N, D] or [N, K*D] or [N, K, D].
        :param visibility_scores_dict: a dictionary of visibility scores, where the keys are the embedding types and the
            values are Tensors of size [N] or [N, K].
        :param id_cls_scores_dict: a dictionary of identity classification scores, where the keys are the embedding types
            and the values are Tensors of size [N, num_classes] or [N, K, num_classes]
        :param pids: A Tensor of size [N] containing the person IDs.
        :return: a tupel with the total combined loss and a dictionnary with performance information for each individual
            loss.
        """
        loss_summary = {}
        losses = []
        # global, foreground and parts embeddings id loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            if mode == 1: continue
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            ce_w = self.losses_weights[key]['id']
            if ce_w > 0:
                parts_id_loss, parts_id_accuracy = self.compute_id_cls_loss(id_cls_scores_dict[key],
                                                                            visibility_scores_dict[key], pids)
                losses.append((ce_w, parts_id_loss))
                loss_info['c'] = parts_id_loss
                loss_info['a'] = parts_id_accuracy

            loss_summary[key] = loss_info

        # global, foreground and parts embeddings triplet loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            if mode == 1 and key != 'globl': continue
            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            tr_w = self.losses_weights[key]['tr']
            if tr_w > 0:
                parts_triplet_loss, parts_trivial_triplets_ratio, parts_valid_triplets_ratio = \
                    self.compute_triplet_loss(embeddings_dict[key], visibility_scores_dict[key], pids)
                losses.append((tr_w, parts_triplet_loss))
                loss_info['t'] = parts_triplet_loss
                loss_info['tt'] = parts_trivial_triplets_ratio
                loss_info['vt'] = parts_valid_triplets_ratio

            loss_summary[key] = loss_info

        # weighted sum of all losses
        if len(losses) == 0:
            return torch.tensor(0., device=(pids.get_device() if pids.is_cuda else None)), loss_summary
        else:
            loss = torch.stack([weight * loss for weight, loss in losses]).sum()
            return loss, loss_summary

    def compute_triplet_loss(self, embeddings, visibility_scores, pids):
        if self.use_visibility_scores:
            visibility = visibility_scores if len(visibility_scores.shape) == 2 else visibility_scores.unsqueeze(1)
        else:
            visibility = None
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)
        triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = self.part_triplet_loss(embeddings, pids,
                                                                                            parts_visibility=visibility)
        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    def compute_id_cls_loss(self, id_cls_scores, visibility_scores, pids):
        if len(id_cls_scores.shape) == 3:
            M = id_cls_scores.shape[1]
            id_cls_scores = id_cls_scores.flatten(0, 1)
            pids = pids.unsqueeze(1).expand(-1, M).flatten(0, 1)
            visibility_scores = visibility_scores.flatten(0, 1)
        weights = None
        if self.use_visibility_scores and visibility_scores.dtype is torch.bool:
            id_cls_scores = id_cls_scores[visibility_scores]
            pids = pids[visibility_scores]
        elif self.use_visibility_scores and visibility_scores.dtype is not torch.bool:
            weights = visibility_scores
        cls_loss = self.identity_loss(id_cls_scores, pids, weights)
        accuracy = self.pred_accuracy(id_cls_scores, pids)
        return cls_loss, accuracy

# For Role Classification
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def build_soccer_net_gsr_reid_head(config: dict):
    return SoccerNetGSR_ReIDHead(
        backbone_num_channels=768,
        output_reid_dim=config["SOCCER_NET_GSR_REID_OUTPUT_REID_DIM"],
        num_pids=config["SOCCER_NET_GSR_REID_NUM_PIDS"],
        backbone_type=config["BACKBONE_TYPE"]
    )

def build_soccer_net_gsr_reid_loss(config: dict):
    weight_dict = {'role_focal_loss': config["SOCCER_NET_GSR_REID_ROLE_FOCAL_LOSS_WEIGHT"],
                   'pid_focal_loss': config["SOCCER_NET_GSR_REID_PID_FOCAL_LOSS_WEIGHT"],
                   'pid_triplet_loss': config["SOCCER_NET_GSR_REID_PID_TRIPLET_LOSS_WEIGHT"],
                   'team_triplet_loss': config["SOCCER_NET_GSR_REID_TEAM_TRIPLET_LOSS_WEIGHT"],
                   'jn_holistic_focal_loss': config["SOCCER_NET_GSR_REID_JN_HOISTIC_FOCAL_LOSS_WEIGHT"],
                   'digit_head_focal_loss': config["SOCCER_NET_GSR_REID_DIGIT_HEAD_FOCAL_LOSS_WEIGHT"],
                   'digit_tail_focal_loss': config["SOCCER_NET_GSR_REID_DIGIT_TAIL_FOCAL_LOSS_WEIGHT"]}
    
    
    return SoccerNetGSR_ReIDLoss(
        weight_dict=weight_dict,
        margin=config["SOCCER_NET_GSR_REID_MARGIN"],
    )