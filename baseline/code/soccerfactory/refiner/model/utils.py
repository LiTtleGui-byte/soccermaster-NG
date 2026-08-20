import torch

def compute_affinity(encoded: torch.Tensor, track_affinity, config: dict, precomputed_pairs=None) -> dict:
    """计算所有不同帧之间的关联度，限制在同一个batch样本内
    
    Args:
        encoded: 编码后的特征，形状为(B,T,N,d)
        track_affinity: 用于计算关联度的模块
        config: 配置字典
        precomputed_pairs: 预计算的帧对列表，如果提供则使用，否则根据参数生成
    
    Returns:
        dict: 包含所有帧对的关联度的字典，键为(t1, t2)元组，值为形状为(B,N,N)的tensor
    """
    if track_affinity is None:
        return {}
        
    B, T, N, d = encoded.shape
    
    # 使用预计算的帧对或根据参数生成
    valid_pairs = precomputed_pairs
    if valid_pairs is None:
        # 获取最大帧间距，限制计算量
        max_frame_distance = config.get('model', {}).get('max_affinity_frame_distance', T)
        
        # 预先计算所有有效的帧对
        valid_pairs = []
        for t1 in range(T-1):
            max_t2 = min(t1 + max_frame_distance + 1, T)
            for t2 in range(t1+1, max_t2):
                valid_pairs.append((t1, t2))
    
    # 如果没有有效帧对，直接返回空字典
    if not valid_pairs:
        return {}
    
    # 创建一个字典来存储所有帧对之间的关联度
    all_affinities = {}
    
    # 将所有帧对转换为列表
    t1_list, t2_list = zip(*valid_pairs)
    num_pairs = len(valid_pairs)
    
    # 批量获取所有样本、所有帧对的特征
    # 使用索引从encoded中提取相应的帧
    # 获取所有批次、所有t1帧的特征 - (B,P,N,d)
    feat_t1_all = encoded[:, t1_list]
    
    # 获取所有批次、所有t2帧的特征 - (B,P,N,d)
    feat_t2_all = encoded[:, t2_list]
    
    # 预处理特征以计算所有可能的检测对
    # 扩展为 (B,P,N,1,d) 然后扩展为 (B,P,N,N,d)
    feat_t1_expanded = feat_t1_all.unsqueeze(3).expand(-1, -1, -1, N, -1)
    
    # 扩展为 (B,P,1,N,d) 然后扩展为 (B,P,N,N,d)
    feat_t2_expanded = feat_t2_all.unsqueeze(2).expand(-1, -1, N, -1, -1)
    
    # 合并特征对，形状为(B,P,N,N,2d)
    all_pairs = torch.cat([feat_t1_expanded, feat_t2_expanded], dim=-1)
    
    # 重塑为(B*P*N*N, 2d)以便一次性计算所有关联度
    all_pairs_flat = all_pairs.reshape(-1, 2*d)
    
    # 计算所有关联分数
    all_affinities_flat = track_affinity(all_pairs_flat).squeeze(-1)  # (B*P*N*N)
    
    # 重塑回(B,P,N,N)
    all_affinities_reshaped = all_affinities_flat.reshape(B, num_pairs, N, N)
    
    # 将关联度结果存储到字典中
    for pair_idx, (t1, t2) in enumerate(valid_pairs):
        all_affinities[(t1, t2)] = all_affinities_reshaped[:, pair_idx]
    
    return all_affinities 