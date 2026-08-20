import torch.nn as nn
from .transformer import SoccerTrackerTransformer
from .timesformer import SoccerTrackerTransformerTimeSformer
from .aaformer import SoccerTrackerTransformerAAFormer
from . import utils  # Import utils module to ensure it's included

def create_model(config: dict) -> nn.Module:
    """Factory function to create the appropriate model based on configuration"""
    model_type = config['model']['type']
    model_config = config['model']
    
    # Common parameters for all models
    common_params = {
        'd_model': model_config['d_model'],
        'nhead': model_config['nhead'],
        'num_layers': model_config['num_layers'],
        'max_clip_frames': config['data']['max_clip_frames'],
        'max_detections': config['data']['max_detections_per_frame'],
        'config': config
    }
    
    # Model type mapping
    model_map = {
        'SoccerTrackerTransformer': SoccerTrackerTransformer,
        'SoccerTrackerTransformerTimeSformer': SoccerTrackerTransformerTimeSformer,
        'SoccerTrackerTransformerAAFormer': SoccerTrackerTransformerAAFormer,
        # Add more model types here as needed
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_map.keys())}")
    
    return model_map[model_type](**common_params) 