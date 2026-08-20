from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .soccer_sequence import SoccerSequenceDataset
from .soccer_sequence_cached import SoccerSequenceCachedDataset
from .soccer_sequence_gt_aug import SoccerSequenceGTAugDataset
# from .soccer_sequence_debug import SoccerSequenceDebugDataset
from .utils import collate_fn
import random
import numpy as np
import torch

def create_dataset(config: dict, split: str) -> Dataset:
    """Factory function to create the appropriate dataset based on configuration
    
    Args:
        config: The configuration dictionary
        split: The dataset split ('train', 'valid', 'test', or 'train+valid')
        
    Returns:
        The appropriate dataset instance
    """
    dataset_type = config['data'].get('dataset_type', 'SoccerSequenceDataset')
    
    # Handle special case for combined train+valid split
    if split == 'train+valid':
        # Create train dataset
        train_dataset = create_single_split_dataset(config, 'train', dataset_type)
        
        # Create valid dataset 
        valid_dataset = create_single_split_dataset(config, 'valid', dataset_type)
        
        # Combine datasets
        return ConcatDataset([train_dataset, valid_dataset])
    else:
        # Regular single split dataset
        return create_single_split_dataset(config, split, dataset_type)

def create_single_split_dataset(config: dict, split: str, dataset_type: str) -> Dataset:
    """Create a dataset for a single split"""
    # Common parameters for all datasets
    common_params = {
        'pipeline_outputs_root': config['data']['pipeline_outputs_root'],
        'pipeline_exp_name': config['data']['pipeline_exp_name'],
        'pipeline_exp_name_test': config['data']['pipeline_exp_name_test'],
        'pipeline_exp_name_valid': config['data']['pipeline_exp_name_valid'],
        'gt_root': config['data']['gt_root'],
        'metadata_path': config['data']['metadata_path'],
        'split': split,
        'max_frames': config['data']['max_frames'],
        'max_clip_frames': config['data']['max_clip_frames'],
        'max_detections_per_frame': config['data']['max_detections_per_frame'],
        'simulate_missing': config['data'].get('simulate_missing', False),
        'normalize_bbox': config['data'].get('normalize_bbox', True),
        'normalize_coords': config['data'].get('normalize_coords', True),
        'normalization_method': config['data'].get('normalization_method', 'z_score'),
        'bbox_mean': config['data'].get('bbox_mean', None),
        'bbox_std': config['data'].get('bbox_std', None),
        'coord_mean': config['data'].get('coord_mean', None),
        'coord_std': config['data'].get('coord_std', None),
        'use_all_frames_as_start': config['data'].get('use_all_frames_as_start', False),
        'augment_coords': config['data'].get('augment_coords', False),
        'augment_coords_mode': config['data'].get('augment_coords_mode', 'individual'),
        'augment_coords_sigma': config['data'].get('augment_coords_sigma', 1.0),
        'rotation_std': config['data'].get('rotation_std', 0.5),
        'translation_std': config['data'].get('translation_std', 0.2),
        'flip_x_prob': config['data'].get('flip_x_prob', 0.0),
        'flip_y_prob': config['data'].get('flip_y_prob', 0.0),
        'max_affinity_frame_distance': config['model'].get('max_affinity_frame_distance', 25),
        'track_enabled': config['training'].get('tasks', {}).get('track', False),
        'precompute_track_masks': config['data'].get('precompute_track_masks', False),
        'random_augment_track_ids': config['data'].get('random_augment_track_ids', False),
        'random_track_id_change_prob': config['data'].get('random_track_id_change_prob', 0.1),
        'random_augment_roles': config['data'].get('random_augment_roles', False),
        'random_role_change_prob': config['data'].get('random_role_change_prob', 0.1),
        'camera_drift_prob': config['data'].get('camera_drift_prob', 0.03),
        'drift_rotation_std': config['data'].get('drift_rotation_std', 2.0),
        'drift_translation_std': config['data'].get('drift_translation_std', 1.0),
        'wipe_mode': config['data'].get('wipe_mode', 'none'),
        'wipe_random_prob': config['data'].get('wipe_random_prob', 0.03),
        'wipe_continuous_prob': config['data'].get('wipe_continuous_prob', 0.05),
    }
    
    # Dataset type mapping
    dataset_map = {
        'SoccerSequenceDataset': SoccerSequenceDataset,
        'SoccerSequenceCachedDataset': SoccerSequenceCachedDataset,
        'SoccerSequenceGTAugDataset': SoccerSequenceGTAugDataset,
        # 'SoccerSequenceDebugDataset': SoccerSequenceDebugDataset,
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_type](**common_params) 

def create_dataloaders(config, seed=None):
    # Create datasets using the factory function
    dataset_type = config['data'].get('dataset_type', 'SoccerSequenceDataset')
    print(f"Creating datasets of type: {dataset_type}")
    
    # Get training split from config or default to 'train'
    train_split = config['data'].get('train_split', 'train')
    val_split = config['data'].get('val_split', 'test')
    
    print(f"Using '{train_split}' split for training and '{val_split}' split for validation")
    
    train_dataset = create_dataset(config, split=train_split)
    val_dataset = create_dataset(config, split=val_split)
    
    # Worker initialization function for reproducibility with DataLoader
    def worker_init_fn(worker_id):
        if seed is not None:
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
    
    # Set generator for shuffling reproducibility
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=g if seed is not None else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    return train_loader, val_loader