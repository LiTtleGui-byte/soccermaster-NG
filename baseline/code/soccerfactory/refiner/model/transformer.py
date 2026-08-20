import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .utils import compute_affinity

class SoccerTrackerTransformer(nn.Module):
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 max_clip_frames: int = 50,
                 max_detections: int = 30,
                 config: dict = None):
        super().__init__()
        self.d_model = d_model
        self.max_clip_frames = max_clip_frames
        self.max_detections = max_detections
        
        # Store task configuration
        self.config = config or {}
        self.task_enabled = {
            'track': self.config['training']['tasks']['track'],
            'role': self.config['training']['tasks']['role'],
            'team': self.config['training']['tasks']['team'],
            'jersey': self.config['training']['tasks']['jersey'],
            'coord': self.config['training']['tasks']['coord'],
            'missing': self.config['training']['tasks']['missing']
        }
        
        # Get normalization method
        self.normalization_method = self.config.get('data', {}).get('normalization_method', 'z_score')
        
        # Get coordinate regression type
        self.coord_regression_type = self.config.get('model', {}).get('coord_regression_type', 'direct')
        self.zero_init_for_residual = self.config.get('model', {}).get('zero_init_for_residual', True)
        
        # Whether to use spatial embeddings
        self.use_spatial_embed = self.config.get('model', {}).get('use_spatial_embed', True)
        
        # Initialize normalization parameters for z_score method
        if self.normalization_method == 'z_score' and 'data' in self.config and 'coord_mean' in self.config['data'] and 'coord_std' in self.config['data']:
            self.register_buffer('coord_mean', torch.tensor(self.config['data']['coord_mean']))
            self.register_buffer('coord_std', torch.tensor(self.config['data']['coord_std']))
        else:
            self.coord_mean = None
            self.coord_std = None

        # Pitch dimensions and boundaries
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.pitch_x_margin = 10.0
        self.pitch_y_margin = 5.0
        self.coord_x_min = -((self.pitch_length / 2) + self.pitch_x_margin)  # -62.5
        self.coord_x_max = ((self.pitch_length / 2) + self.pitch_x_margin)   # 62.5
        self.coord_y_min = -((self.pitch_width / 2) + self.pitch_y_margin)   # -39.0
        self.coord_y_max = ((self.pitch_width / 2) + self.pitch_y_margin)    # 39.0

        # 特征编码层
        self.reid_feature_proj = nn.Sequential(
            nn.Linear(256, d_model//4),  # ReID特征
            nn.ReLU(),
            nn.LayerNorm(d_model//4)
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, d_model//4),    # 球场坐标
            nn.ReLU(),
            nn.LayerNorm(d_model//4)
        )
        self.role_proj = nn.Sequential(
            nn.Linear(4, d_model//8),    # 角色 one-hot 编码 (player, goalkeeper, referee, unknown)
            nn.ReLU(),
            nn.LayerNorm(d_model//8)
        )
        self.team_proj = nn.Sequential(
            nn.Linear(3, d_model//8),  # 队伍 one-hot 编码
            nn.ReLU(),
            nn.LayerNorm(d_model//8)
        )
        self.jersey_num_proj = nn.Sequential(
            nn.Embedding(100, d_model//8),  # 球衣号码 one-hot 编码  
            nn.LayerNorm(d_model//8)
        )
        self.track_id_proj = nn.Sequential(
            nn.Embedding(150, d_model//8),  # 轨迹ID嵌入，假设最大轨迹ID为1000
            nn.LayerNorm(d_model//8)
        )
        
        # 时空位置编码
        self.temporal_embed = nn.Embedding(max_clip_frames, d_model)
        if self.use_spatial_embed:
            self.spatial_embed = nn.Embedding(max_detections, d_model)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048,
            dropout=0.1, batch_first=True,
            # norm_first=True  # Apply normalization before each sub-layer (more stable)
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        # self.norm = nn.LayerNorm(d_model)  # Add an extra normalization layer for stability
        
        # 多任务解码头 - 根据配置决定是否创建
        if self.task_enabled['role']:
            # 角色分类 (player, goalkeeper, referee, unknown)
            # 不加激活函数 - CrossEntropyLoss 会在损失计算中应用 softmax
            self.role_head = nn.Linear(d_model, 4)
        
        if self.task_enabled['team']:
            # 队伍分类 - CrossEntropyLoss 会在损失计算中应用 softmax
            self.team_head = nn.Linear(d_model, 3)
        
        if self.task_enabled['jersey']:
            # 球衣号码分类 (0-99) - CrossEntropyLoss 会在损失计算中应用 softmax
            self.jersey_head = nn.Linear(d_model, 100)
        
        if self.task_enabled['coord']:
            # 坐标预测头 - 可以直接回归坐标或预测残差
            self.coord_refine = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
            
            # 如果是residual模式并且需要零初始化，则将最后一层的权重和偏置置零
            if self.coord_regression_type == 'residual' and self.zero_init_for_residual:
                # Initialize the last layer to output zeros
                nn.init.zeros_(self.coord_refine[-1].weight)
                nn.init.zeros_(self.coord_refine[-1].bias)
                
            # 对于direct方式，加入tanh激活以限制输出范围
            if self.coord_regression_type == 'direct':
                self.coord_refine.add_module('tanh', nn.Tanh())  # 输出[-1, 1]范围的归一化坐标
        
        # 轨迹关联模块
        if self.task_enabled['track']:
            self.track_affinity = nn.Sequential(
                nn.Linear(2*d_model, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
                # 不加sigmoid, 在损失函数中会处理为对比损失
            )
        
        # 漏检预测
        if self.task_enabled['missing']:
            self.missing_predictor = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()  # 二分类问题，输出0-1之间的概率
            )

    def forward(self, batch: dict) -> dict:
        # 特征融合
        B, T, N, _ = batch['feats'].shape
        device = batch['feats'].device
        
        # 基础特征投影
        reid_feat = self.reid_feature_proj(batch['feats'])      # (B,T,N,d//4)
        coord_feat = self.coord_proj(batch['coords'])         # (B,T,N,d//4)
        role_feat = self.role_proj(batch['roles'])            # (B,T,N,d//8)
        team_feat = self.team_proj(batch['teams'])            # (B,T,N,d//8)
        jersey_feat = self.jersey_num_proj(batch['JNs'])      # (B,T,N,d//8)
        track_feat = self.track_id_proj(batch['track_ids'])   # (B,T,N,d//8)
        
        # 合并特征
        base_feat = torch.cat([reid_feat, coord_feat, role_feat, team_feat, jersey_feat, track_feat], dim=-1)
        
        # 添加时间位置编码
        time_ids = torch.arange(T, device=device).expand(B, T)
        time_emb = self.temporal_embed(time_ids).unsqueeze(2) # (B,T,1,d)
        base_feat += time_emb
        
        # 添加空间位置编码
        if self.use_spatial_embed:
            space_ids = torch.arange(N, device=device).expand(B, T, N)
            space_emb = self.spatial_embed(space_ids)             # (B,T,N,d)
            base_feat += space_emb
        
        # 展平序列维度
        flat_feat = base_feat.view(B, T*N, self.d_model)      # (B,T*N,d)
        valid_mask = batch['visible_mask'].view(B, T*N)       # (B,T*N)
        
        # Transformer处理
        padding_mask = ~valid_mask
        # flat_feat = self.norm(flat_feat)  # Additional normalization before entering transformer
        encoded = self.transformer(
            flat_feat,
            src_key_padding_mask=padding_mask
        )  # (B, T*N, d)
        
        # 恢复空间维度
        encoded = encoded.view(B, T, N, self.d_model)        # (B,T,N,d)
        
        # 多任务预测 - 根据配置决定输出
        outputs = {}
        
        if self.task_enabled['role'] and hasattr(self, 'role_head'):
            outputs['role_logits'] = self.role_head(encoded)       # (B,T,N,4)
            # 在推理时，可以使用 torch.softmax(outputs['role_logits'], dim=-1) 获取概率
        
        if self.task_enabled['team'] and hasattr(self, 'team_head'):
            outputs['team_logits'] = self.team_head(encoded)       # (B,T,N,3)
            # 在推理时，可以使用 torch.softmax(outputs['team_logits'], dim=-1) 获取概率
        
        if self.task_enabled['jersey'] and hasattr(self, 'jersey_head'):
            outputs['jersey_logits'] = self.jersey_head(encoded)   # (B,T,N,100)
            # 在推理时，可以使用 torch.softmax(outputs['jersey_logits'], dim=-1) 获取概率
        
        if self.task_enabled['coord'] and hasattr(self, 'coord_refine'):
            # 坐标预测 - 根据配置决定是直接回归还是预测残差
            if self.coord_regression_type == 'direct':
                # 直接回归范围在[-1,1]的归一化坐标
                norm_coords = self.coord_refine(encoded)  # (B,T,N,2) 范围[-1, 1]
                
                # 将归一化坐标转换回实际球场坐标，根据所用的归一化方法
                if self.normalization_method == "minmax":
                    # Min-max normalization was used - denormalize directly
                    x_coords = norm_coords[..., 0] * (self.coord_x_max - self.coord_x_min) / 2 + (self.coord_x_max + self.coord_x_min) / 2
                    y_coords = norm_coords[..., 1] * (self.coord_y_max - self.coord_y_min) / 2 + (self.coord_y_max + self.coord_y_min) / 2
                else:
                    # Z-score normalization was used - use the pre-initialized tensors
                    # Make sure tensors are on the correct device
                    if self.coord_mean is None or self.coord_std is None:
                        # Fallback in case tensors weren't initialized properly
                        coord_mean = torch.tensor(self.config['data']['coord_mean'], device=device)
                        coord_std = torch.tensor(self.config['data']['coord_std'], device=device)
                    else:
                        # Use pre-initialized tensors
                        coord_mean = self.coord_mean.to(device) if self.coord_mean.device != device else self.coord_mean
                        coord_std = self.coord_std.to(device) if self.coord_std.device != device else self.coord_std
                    
                    # Denormalize: x = z * std + mean
                    coords_denorm = norm_coords * coord_std + coord_mean
                    x_coords = coords_denorm[..., 0]
                    y_coords = coords_denorm[..., 1]
                
                outputs['coords_pred'] = torch.stack([x_coords, y_coords], dim=-1)  # (B,T,N,2)
            
            else:  # 'residual' mode
                # 预测归一化空间中的残差
                coords_residual = self.coord_refine(encoded)  # (B,T,N,2) 范围[-1, 1]
                
                # 获取输入坐标（已经在归一化空间中）
                input_coords = batch['coords']  # (B,T,N,2)
                
                # 计算归一化空间中的最终坐标：输入坐标 + 残差
                norm_coords = input_coords + coords_residual
                
                # 将归一化坐标转换回实际球场坐标，根据所用的归一化方法
                if self.normalization_method == "minmax":
                    # Min-max normalization was used - denormalize directly
                    x_coords = norm_coords[..., 0] * (self.coord_x_max - self.coord_x_min) / 2 + (self.coord_x_max + self.coord_x_min) / 2
                    y_coords = norm_coords[..., 1] * (self.coord_y_max - self.coord_y_min) / 2 + (self.coord_y_max + self.coord_y_min) / 2
                else:
                    # Z-score normalization was used - use the pre-initialized tensors
                    # Make sure tensors are on the correct device
                    if self.coord_mean is None or self.coord_std is None:
                        # Fallback in case tensors weren't initialized properly
                        coord_mean = torch.tensor(self.config['data']['coord_mean'], device=device)
                        coord_std = torch.tensor(self.config['data']['coord_std'], device=device)
                    else:
                        # Use pre-initialized tensors
                        coord_mean = self.coord_mean.to(device) if self.coord_mean.device != device else self.coord_mean
                        coord_std = self.coord_std.to(device) if self.coord_std.device != device else self.coord_std
                    
                    # Denormalize: x = z * std + mean
                    coords_denorm = norm_coords * coord_std + coord_mean
                    x_coords = coords_denorm[..., 0]
                    y_coords = coords_denorm[..., 1]
                
                outputs['coords_pred'] = torch.stack([x_coords, y_coords], dim=-1)  # (B,T,N,2)
                
                # 存储残差以便损失计算（在归一化空间中）
                outputs['coords_residual'] = coords_residual
        
        if self.task_enabled['missing'] and hasattr(self, 'missing_predictor'):
            outputs['missing_probs'] = self.missing_predictor(encoded) # (B,T,N,1)
        
        # 轨迹关联计算 - 计算所有帧之间的关联度
        if self.task_enabled['track'] and hasattr(self, 'track_affinity'):
            # 检查batch中是否有预计算的帧对
            precomputed_pairs = None
            if 'track_frame_pairs' in batch:
                precomputed_pairs = batch['track_frame_pairs']
                
            track_affinity = self.compute_affinity(encoded, precomputed_pairs)   # 字典，包含所有帧对的关联度
            outputs['track_affinity'] = track_affinity
        
        return outputs
    
    def compute_affinity(self, encoded: torch.Tensor, precomputed_pairs=None) -> dict:
        """计算所有不同帧之间的关联度，限制在同一个batch样本内
        
        Args:
            encoded: 编码后的特征，形状为(B,T,N,d)
            precomputed_pairs: 预计算的帧对列表，如果提供则使用，否则根据参数生成
        """
        track_affinity = getattr(self, 'track_affinity', None)
        return compute_affinity(encoded, track_affinity, self.config, precomputed_pairs) 