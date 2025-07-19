from __future__ import division, absolute_import

import torch
import torch.nn as nn
from data.video_caption import keywords_list
from models.caption_classification import CaptionClassificationLoss, CaptionClassificationMetrics

class CaptionClassificationHeadAlign(nn.Module):
    def __init__(self, input_dim=768, backbone_type='image', dropout_rate=0.1, use_attn_pool=False, 
                 use_transformers=False, num_transformer_encoder=2):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 分类类别数
            backbone_type: backbone类型，'image'或'video'
            dropout_rate: dropout比率
            use_attn_pool: 是否使用attention pooling，默认False
            use_transformers: 是否在pooling前使用transformer encoder，默认False
            num_transformer_encoder: transformer encoder的层数，默认2
        """
        super().__init__()
        assert backbone_type == 'video'
        self.backbone_type = backbone_type
        self.use_attn_pool = use_attn_pool
        self.use_transformers = use_transformers
        num_classes = len(keywords_list)
        
        # Transformer encoder layers (optional)
        if self.use_transformers:
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=input_dim * 4,
                dropout=dropout_rate,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer,
                num_layers=num_transformer_encoder
            )
        
        if self.use_attn_pool:
            self.query_token = nn.Parameter(torch.randn(1, 1, input_dim))
            # Multi-head attention for pooling
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            # Layer norm after attention pooling
            self.attn_pool_ln = nn.LayerNorm(input_dim)
        
        # 分类头网络
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(input_dim // 2, input_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(input_dim // 4, num_classes)
        # )
        self.classifier_ln1 = nn.LayerNorm(input_dim)
        self.classifier_ln2 = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化query token
        if self.use_attn_pool:
            nn.init.normal_(self.query_token, std=0.02)

    def forward(self, backbone_outputs, metas):
        """
        Args:
            backbone_outputs: backbone输出，包含global_features
            metas: 元数据
            
        Returns:
            包含logits的字典
        """
        global_features = backbone_outputs['global_features']
        global_features = self.classifier_ln1(global_features)
        
        # 可选的transformer encoder处理
        if self.use_transformers:
            global_features = self.transformer_encoder(global_features)
        
        # 根据是否使用attention pooling选择不同的特征提取方式
        if self.use_attn_pool:
            batch_size = global_features.size(0)
            query = self.query_token.expand(batch_size, -1, -1)  # [N, 1, D]
            
            attn_output, _ = self.attn_pool(
                query=query,  # [N, 1, D]
                key=global_features,  # [N, seq_len, D]
                value=global_features  # [N, seq_len, D]
            )
            
            vision_features = self.attn_pool_ln(attn_output.squeeze(1))  # [N, D]
        else:
            vision_features = global_features.mean(dim=1)  # [N, D]
        
        vision_features = self.classifier_ln2(vision_features)
        logits = self.classifier(vision_features)  # [N, num_classes]
        
        output = {
            'logits': logits,
            'features': vision_features
        }
        return output

def build_caption_classification_head_align(config: dict):
    """构建Caption分类头"""
    return CaptionClassificationHeadAlign(
        input_dim=768,
        backbone_type=config["BACKBONE_TYPE"],
        dropout_rate=config["CAPTION_CLASSIFICATION_DROPOUT_RATE"],
        use_attn_pool=config["CAPTION_CLASSIFICATION_USE_ATTN_POOL"],
        use_transformers=config["CAPTION_CLASSIFICATION_USE_TRANSFORMERS"],
        num_transformer_encoder=config["CAPTION_CLASSIFICATION_NUM_TRANSFORMER_ENCODER"]
    )


def build_caption_classification_loss_align(config: dict):
    """构建Caption分类损失函数"""
    weight_dict = {
        'classification_loss': config["CAPTION_CLASSIFICATION_LOSS_WEIGHT"]
    }
    
    return CaptionClassificationLoss(
        weight_dict=weight_dict,
        label_smoothing=config["CAPTION_CLASSIFICATION_LABEL_SMOOTHING"]
    )


def build_caption_classification_metrics_align(config: dict):
    return CaptionClassificationMetrics() 