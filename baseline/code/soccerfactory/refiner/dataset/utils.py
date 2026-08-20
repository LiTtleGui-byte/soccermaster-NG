import torch
from typing import Dict, List

def collate_fn(batch: List[Dict]) -> Dict:
    """自定义批次处理"""
    collated = {}
    
    # 处理特殊的字典类型 track association masks
    track_dict_keys = ['track_valid_pairs', 'track_pos_masks', 'track_neg_masks']
    
    # 检查这个批次是否包含跟踪关联数据
    has_track_data = 'track_frame_pairs' in batch[0]
    
    # 处理普通张量数据
    for key in batch[0]:
        if key not in track_dict_keys and key != 'track_frame_pairs':
            collated[key] = torch.stack([b[key] for b in batch])
    
    # 如果包含跟踪关联数据，处理特殊的字典类型数据
    if has_track_data:
        # 处理帧对列表 - 所有样本共享相同的帧对
        collated['track_frame_pairs'] = batch[0]['track_frame_pairs']
        
        # 处理掩码字典
        for key in track_dict_keys:
            if key in batch[0]:
                collated[key] = {}
                # 对每个帧对分别处理
                for frame_pair in batch[0][key]:
                    # 收集所有batch样本的同一帧对的掩码
                    masks = [b[key][frame_pair] for b in batch]
                    # 堆叠成批次
                    collated[key][frame_pair] = torch.stack(masks)
    
    # 生成全局注意力掩码 (B, T, N)
    # visible_mask = collated['visible_mask']  # (B, T, N)
    # time_mask = ~collated['time_mask']       # (B, T)
    # global_mask = ~(time_mask.unsqueeze(-1) | ~visible_mask)
    # collated['attention_mask'] = global_mask.float()
    
    return collated 