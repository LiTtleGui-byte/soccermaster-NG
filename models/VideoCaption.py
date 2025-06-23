from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoCaptionHead(nn.Module):
    def __init__(self, loss_type='siglip_loss', backbone_type='image'):
        super().__init__()
        self.loss_type = loss_type
        self.backbone_type = backbone_type
        
        if loss_type == 'siglip_loss':
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
            self.logits_bias = nn.Parameter(torch.tensor(-10.0))
        elif loss_type == 'infonce_loss':
            self.temperature = nn.Parameter(torch.tensor(0.3))
        else:
            raise ValueError(f"Loss type {loss_type} not supported.")

    def forward(self, backbone_outputs, metas):
        global_features, local_features, text_features = backbone_outputs['global_features'], backbone_outputs['local_features'], backbone_outputs['text_features']
        
        # 处理视觉特征
        if self.backbone_type == 'video':
            vision_features = global_features.mean(dim=1)
        else:
            vision_features = global_features[:, 0]
        
        # 归一化特征
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # 计算基础相似度矩阵
        base_similarity_matrix = torch.matmul(text_features, vision_features.t())
        
        # 根据loss_type处理相似度矩阵
        if self.loss_type == 'siglip_loss':
            # 应用logit_scale和logits_bias
            processed_similarity_matrix = base_similarity_matrix * self.logit_scale.exp() + self.logits_bias
        elif self.loss_type == 'infonce_loss':
            # 应用temperature
            processed_similarity_matrix = base_similarity_matrix / self.temperature
        else:
            processed_similarity_matrix = base_similarity_matrix
        
        output = {
            'vision_features': vision_features,
            'text_features': text_features,
            'base_similarity_matrix': base_similarity_matrix,  # 原始相似度矩阵
            'processed_similarity_matrix': processed_similarity_matrix  # 处理后的相似度矩阵
        }
            
        return output
    
class VideoCaptionLoss(nn.Module):
    def __init__(self, weight_dict, loss_type='siglip_loss'):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_type = loss_type

    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出，包含processed_similarity_matrix等
            targets: 目标标签，包含target_label等
        """
        processed_similarity_matrix = outputs['processed_similarity_matrix']
        base_similarity_matrix = outputs['base_similarity_matrix']
        
        caption = [target['caption'] for target in targets]
        target_label = create_label_from_comment(caption, caption)
        target_label = target_label.to(processed_similarity_matrix.device)
        
        losses = {}
        
        if self.loss_type == 'siglip_loss':
            loss = self.compute_siglip_loss(processed_similarity_matrix, target_label)
            losses['siglip_loss'] = loss
        elif self.loss_type == 'infonce_loss':
            loss = self.compute_infonce_loss(processed_similarity_matrix, target_label)
            losses['infonce_loss'] = loss
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported.")
        
        # 使用原始相似度矩阵计算准确率
        top_1_accuracy, top_3_accuracy, top_5_accuracy = self.calculate_top_k_accuracy(base_similarity_matrix, target_label)
        losses['top_1_accuracy'] = top_1_accuracy
        losses['top_3_accuracy'] = top_3_accuracy
        losses['top_5_accuracy'] = top_5_accuracy
            
        return losses, self.weight_dict

    def compute_siglip_loss(self, processed_similarity_matrix, target_label):
        """
        计算SigLIP损失
        Args:
            processed_similarity_matrix: 已处理的相似度矩阵 [batch_size, batch_size]
            target_label: 目标标签 [batch_size, batch_size], 1表示正样本，-1表示负样本
        """
        # processed_similarity_matrix已经应用了logit_scale和logits_bias
        logits_per_image = processed_similarity_matrix.t()
        
        # SigLIP损失：对每个样本计算sigmoid损失
        loss = -F.logsigmoid(target_label * logits_per_image).sum() / target_label.shape[0]
        
        return loss

    def compute_infonce_loss(self, processed_similarity_matrix, target_label):
        """
        计算InfoNCE损失
        Args:
            processed_similarity_matrix: 已处理的相似度矩阵 [batch_size, batch_size]
            target_label: 目标标签 [batch_size, batch_size], 1表示正样本，-1表示负样本
        """
        # processed_similarity_matrix已经应用了temperature
        
        # 获取正样本和负样本的mask
        positive_samples = (target_label == 1)
        negative_samples = (target_label == -1)
        
        # 提取正样本和负样本的相似度
        pos_similarities = processed_similarity_matrix[positive_samples]
        neg_similarities = processed_similarity_matrix[negative_samples]
        
        # 计算InfoNCE损失
        if pos_similarities.numel() > 0:
            pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_similarities)))
        else:
            pos_loss = torch.tensor(0.0, device=processed_similarity_matrix.device)
            
        if neg_similarities.numel() > 0:
            neg_loss = torch.mean(F.softplus(-neg_similarities))
        else:
            neg_loss = torch.tensor(0.0, device=processed_similarity_matrix.device)
        
        loss = pos_loss + neg_loss
        return loss

    def calculate_top_k_accuracy(self, similarity_matrix, target_label):
        """
        计算Top-K准确率
        Args:
            similarity_matrix: 相似度矩阵 [batch_size, batch_size]
            target_label: 目标标签 [batch_size, batch_size]
        Returns:
            tuple: (top_1_accuracy, top_3_accuracy, top_5_accuracy)
        """
        batch_size = similarity_matrix.size(0)
        top_1_correct = torch.tensor(0.0, device=similarity_matrix.device)
        top_3_correct = torch.tensor(0.0, device=similarity_matrix.device)
        top_5_correct = torch.tensor(0.0, device=similarity_matrix.device)
        
        for i in range(batch_size):
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            
            # Top-1准确率
            if target_label[i, sorted_indices[0]] == 1:
                top_1_correct += 1
                
            # Top-3准确率
            if torch.any(target_label[i, sorted_indices[:3]] == 1):
                top_3_correct += 1
                
            # Top-5准确率  
            if torch.any(target_label[i, sorted_indices[:5]] == 1):
                top_5_correct += 1

        # 计算每个Top-K的准确率
        top_1_accuracy = top_1_correct / batch_size
        top_3_accuracy = top_3_correct / batch_size
        top_5_accuracy = top_5_correct / batch_size
        
        return top_1_accuracy, top_3_accuracy, top_5_accuracy

def build_video_caption_loss(config: dict):
    """构建VideoCaption损失函数"""
    loss_type = config["VIDEO_CAPTION_LOSS_TYPE"]
    
    if loss_type == "siglip_loss":
        weight_dict = {'siglip_loss': config["VIDEO_CAPTION_SIGLIP_LOSS_WEIGHT"]}
    elif loss_type == "infonce_loss":
        weight_dict = {'infonce_loss': config["VIDEO_CAPTION_INFONCE_LOSS_WEIGHT"]}
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return VideoCaptionLoss(
        weight_dict=weight_dict,
        loss_type=loss_type
    )

def create_label_from_comment(caption_text, special_categories = {
            "end of half game", "off side", "start of half game",
            "ball possession", "substitution"
        }):
    N = len(caption_text)
    tensor = torch.eye(N) * 2 - 1
    for i in range(N):
        for j in range(i + 1, N):
            if caption_text[i] == caption_text[j] and caption_text[i] in special_categories:
                tensor[i, j] = 1
                tensor[j, i] = 1
    return tensor

def build_video_caption_head(config: dict):
    assert config["BACKBONE_TYPE"] == "video"
    
    return VideoCaptionHead(
        loss_type=config["VIDEO_CAPTION_LOSS_TYPE"],
        backbone_type=config["BACKBONE_TYPE"]
    )