import os
import argparse
import yaml
import torch
import numpy as np
import pickle
import zipfile
import pandas as pd
import shutil
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
from copy import deepcopy

from model.factory import create_model
from torch.nn import functional as F
from dataset.dataset_utils import (
    PITCH_LENGTH, PITCH_WIDTH, PITCH_X_MARGIN, PITCH_Y_MARGIN,
    COORD_X_MIN, COORD_X_MAX, COORD_Y_MIN, COORD_Y_MAX,
    ROLE_MAP, TEAM_MAP,
    load_pipeline_data, load_gt_data, process_pipeline_video,
    process_gt_video, compute_video_matches, permute_gt_video,
    normalize_coordinates, denormalize_coordinates
)

# Add pitch visualization imports
PITCH_FILE = os.path.join(os.path.dirname(__file__), "Radar.png")
# Also check in the current directory if the file doesn't exist in the script directory
if not os.path.exists(PITCH_FILE):
    PITCH_FILE = "Radar.png"
    if not os.path.exists(PITCH_FILE):
        print(f"Warning: Radar.png not found in either {os.path.dirname(__file__)} or current directory.")
        # Use a fallback path, which will create a blank pitch if not found
        PITCH_FILE = None

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with SoccerTrackerTransformer')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input_pklz', type=str, required=True, help='Input pklz file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--metadata_path', type=str, required=True, help='Metadata file path')
    parser.add_argument('--split', type=str, default='test', help='Data split (train, valid, test, train+valid)')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--visualize_video', action='store_true', help='Save visualization frames as videos')
    parser.add_argument('--img_dir', type=str, help='Directory containing images for visualization, required if --visualize or --visualize_video is set')
    parser.add_argument('--vis_output_dir', type=str, help='Directory to save visualization results, defaults to {output_dir}/visualization if not specified')
    parser.add_argument('--video_output_dir', type=str, help='Directory to save visualization videos, defaults to {output_dir}/visualization_video if not specified')
    parser.add_argument('--gt_dir', type=str, help='Directory containing ground truth data, defaults to parent directory of img_dir')
    parser.add_argument('--video_fps', type=int, default=25, help='FPS for saved videos')
    return parser.parse_args()


def load_config(config_path):
    """Load and merge YAML configurations with support for inheritance"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle imports if present
    
    if 'imports' in config:
        base_config = {}
        for import_path in config['imports']:
            import_full_path = os.path.join(os.path.dirname(config_path), import_path)
            
            # Recursively load imported config to handle nested imports
            imported_config = load_config(import_full_path)
            base_config = deep_merge(base_config, imported_config)
        
        # Remove imports from config
        del config['imports']
        
        # Merge configurations, with current config taking precedence
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config


def deep_merge(base, override):
    """Deep merge two dictionaries with nested structure"""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def inference_on_video(model, vid_data, device, config, max_clip_frames=None):
    """Run inference on a single video by processing clips of max_clip_frames frames"""
    model.eval()
    
    # Get clip size from config if not provided as parameter
    if max_clip_frames is None:
        max_clip_frames = config['data']['max_clip_frames']
    max_frames = vid_data['feats'].shape[0]
    
    # Initialize outputs
    outputs = {}
    
    # Process in clips
    with torch.no_grad():
        for start_idx in tqdm(range(0, max_frames, max_clip_frames), desc="Processing clips", leave=False):
            end_idx = min(start_idx + max_clip_frames, max_frames)
            actual_clip_frames = end_idx - start_idx
            
            # Create clip data
            clip = {}
            for k, v in vid_data.items():
                # Handle the case where clip is shorter than max_clip_frames
                if actual_clip_frames < max_clip_frames:
                    # Create padding tensor
                    pad_shape = list(v[start_idx:end_idx].shape)
                    pad_shape[0] = max_clip_frames - actual_clip_frames
                    padding = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                    # Concatenate data and padding
                    clip[k] = torch.cat([v[start_idx:end_idx], padding], dim=0)
                else:
                    clip[k] = v[start_idx:end_idx]
            
            # Add time mask for padded frames
            clip['time_mask'] = torch.zeros(max_clip_frames, dtype=torch.bool)
            clip['time_mask'][:actual_clip_frames] = True
            
            # Move clip to device and add batch dimension
            for k, v in clip.items():
                clip[k] = v.unsqueeze(0).to(device)  # Add batch dimension with unsqueeze(0)
            
            # Forward pass
            clip_outputs = model(clip)
            
            # Store outputs
            for k, v in clip_outputs.items():
                if k == 'track_affinity':
                    # Handle track_affinity separately as it's a dictionary
                    if k not in outputs:
                        outputs[k] = {}
                    for (t1, t2), affinity in v.items():
                        # Adjust time indices based on clip position
                        t1_adjusted = t1 + start_idx
                        t2_adjusted = t2 + start_idx
                        outputs[k][(t1_adjusted, t2_adjusted)] = affinity
                else:
                    # For tensor outputs, only keep the valid frames
                    # Remove batch dimension (squeeze) and keep only valid frames
                    valid_output = v.squeeze(0)[:actual_clip_frames]
                    if k not in outputs:
                        outputs[k] = []
                    outputs[k].append(valid_output)
    
    # Concatenate outputs
    processed_outputs = {}
    for k, v in outputs.items():
        if k == 'track_affinity':
            processed_outputs[k] = v  # Already processed above
        else:
            processed_outputs[k] = torch.cat(v, dim=0)
    
    return processed_outputs


def update_predictions(original_preds, outputs, detection_mappings, task_enabled):
    """Update the original predictions with the model outputs"""
    # Create a copy of the original predictions
    updated_preds = original_preds.copy()
    
    # Process role predictions
    if task_enabled['role']:
        role_probs = F.softmax(outputs['role_logits'], dim=-1)  # (T,N,4)
        role_map = ['player', 'goalkeeper', 'referee', 'unknown']
        
        for t, frame_mapping in detection_mappings.items():
            for det_idx, original_idx in frame_mapping.items():
                if outputs['role_logits'].shape[1] > det_idx:
                    # Get predicted role
                    role_idx = role_probs[t, det_idx].argmax().item()
                    role = role_map[role_idx]
                    # Update prediction
                    updated_preds.at[original_idx, 'role'] = role
    # # Add debug information to compare original and updated role predictions
    # if task_enabled['role']:
    #     print("\nRole Prediction Comparison:")
    #     print("=" * 50)
    #     print(f"{'Original Index':<15}{'Original Role':<15}{'New Role':<15}{'Changed':<10}")
    #     print("-" * 50)
        
    #     for t, frame_mapping in detection_mappings.items():
    #         for det_idx, original_idx in frame_mapping.items():
    #             if outputs['role_logits'].shape[1] > det_idx:
    #                 # Get original role
    #                 original_role = original_preds.at[original_idx, 'role']
                    
    #                 # Get predicted role
    #                 role_idx = role_probs[t, det_idx].argmax().item()
    #                 new_role = role_map[role_idx]
                    
    #                 # Check if the role has changed
    #                 changed = original_role != new_role
                    
    #                 # Print comparison
    #                 print(f"{original_idx:<15}{str(original_role):<15}{new_role:<15}{'✓' if changed else '✗':<10}")
        
    #     print("=" * 50)
    
    # Process team predictions
    if task_enabled['team']:
        team_probs = F.softmax(outputs['team_logits'], dim=-1)  # (T,N,3)
        team_map = ['left', 'right', 'nan']
        
        for t, frame_mapping in detection_mappings.items():
            for det_idx, original_idx in frame_mapping.items():
                if outputs['team_logits'].shape[1] > det_idx:
                    # Get predicted team
                    team_idx = team_probs[t, det_idx].argmax().item()
                    team = team_map[team_idx]
                    # Update prediction
                    updated_preds.at[original_idx, 'team'] = team
    
    # Process jersey predictions
    if task_enabled['jersey']:
        jersey_probs = F.softmax(outputs['jersey_logits'], dim=-1)  # (T,N,100)
        
        for t, frame_mapping in detection_mappings.items():
            for det_idx, original_idx in frame_mapping.items():
                if outputs['jersey_logits'].shape[1] > det_idx:
                    # Get predicted jersey number
                    jersey_num = jersey_probs[t, det_idx].argmax().item()
                    # Update prediction if jersey number is valid
                    if jersey_num > 0:
                        updated_preds.at[original_idx, 'jersey_number'] = int(jersey_num)
    
    # Process coordinate predictions
    if task_enabled['coord']:
        for t, frame_mapping in detection_mappings.items():
            for det_idx, original_idx in frame_mapping.items():
                if outputs['coords_pred'].shape[1] > det_idx:
                    # Get predicted coordinates
                    x, y = outputs['coords_pred'][t, det_idx].tolist()
                    
                    # Update the bbox_pitch dictionary
                    if isinstance(updated_preds.at[original_idx, 'bbox_pitch'], dict):
                        updated_preds.at[original_idx, 'bbox_pitch']['x_bottom_middle'] = x
                        updated_preds.at[original_idx, 'bbox_pitch']['y_bottom_middle'] = y
                    else:
                        # Create a new bbox_pitch dictionary if none exists
                        updated_preds.at[original_idx, 'bbox_pitch'] = {
                            'x_bottom_middle': x,
                            'y_bottom_middle': y
                        }
    
    return updated_preds


def visualize_video_inputs_outputs(img_dir: str, image_id_list: List[str], vid_data: Dict, 
                                  outputs: Dict, vis_output_dir: str, vid: str, 
                                  task_enabled: Dict, gt_data_processed: Dict = None):
    """Visualize model inputs and outputs for a video
    
    Args:
        img_dir: Directory containing images
        image_id_list: List of image IDs 
        vid_data: Model input data
        outputs: Model output data
        vis_output_dir: Directory to save visualization results
        vid: Video ID
        task_enabled: Dictionary of enabled tasks
        gt_data_raw: Raw ground truth data (optional)
        gt_data_processed: Processed ground truth data (optional)
    """
    # Create output directory
    vid_output_dir = os.path.join(vis_output_dir, f'SNGS-{vid}')
    os.makedirs(vid_output_dir, exist_ok=True)
    
    # Define colors for visualization
    detection_color = (0, 255, 0)  # Green for bounding boxes
    
    # Team colors
    team_colors = {
        'left': (0, 0, 255),  # Blue for left team
        'right': (255, 0, 0),  # Red for right team
        'nan': (180, 180, 180)  # Grey for unknown team
    }
    
    # Role colors
    role_colors = {
        'player': (0, 255, 0),      # Green
        'goalkeeper': (255, 255, 0), # Yellow
        'referee': (255, 0, 255),    # Magenta
        'unknown': (180, 180, 180)   # Grey
    }
    
    # Role and team maps (same as in process_video_data)
    role_map = ['player', 'goalkeeper', 'referee', 'unknown']
    team_map = ['left', 'right', 'nan']
    
    for t, image_id in enumerate(tqdm(image_id_list, desc=f"Visualizing {vid}", leave=False)):
        # Load image
        img_path = os.path.join(img_dir, f"{int(image_id[-6:]):06d}.jpg")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Convert to RGB for visualization
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get visible detections for this frame
        visible_mask = vid_data['visible_mask'][t].bool()
        num_detections = visible_mask.sum().item()
        
        # if num_detections == 0:
        #     # If no detections, save original image
        #     output_path = os.path.join(vid_output_dir, f"frame_{int(image_id):06d}.jpg")
        #     cv2.imwrite(output_path, img)
        #     continue
        
        # Get bounding boxes
        bboxes = vid_data['bbox_ltwh'][t][visible_mask].cpu().numpy()
        
        # Draw frame count
        draw_frame_count(vis_img, t, len(image_id_list))
        
        # 1. Ground Truth DataFrame (if available)
        gt_detections = []
        
        # Option 1: Use processed ground truth data if available
        if gt_data_processed is not None and 'visible_mask' in gt_data_processed:
            gt_visible_mask = gt_data_processed['visible_mask'][t].bool()
            
            for i in range(len(gt_visible_mask)):
                if gt_visible_mask[i]:
                    # Extract ground truth data
                    gt_coords = gt_data_processed['coords'][t, i].tolist()
                    
                    # Get role
                    role = None
                    if 'roles' in gt_data_processed:
                        role_idx = gt_data_processed['roles'][t, i].argmax().item()
                        role = [k for k, v in ROLE_MAP.items() if v == role_idx][0]
                    
                    # Get team
                    team = None
                    if 'teams' in gt_data_processed:
                        team_idx = gt_data_processed['teams'][t, i].argmax().item()
                        team = [k for k, v in TEAM_MAP.items() if v == team_idx][0]
                    
                    # Get jersey number
                    jersey_number = None
                    if 'JNs' in gt_data_processed and gt_data_processed['JNs'][t, i].item() > 0:
                        jersey_number = int(gt_data_processed['JNs'][t, i].item())
                    
                    # Create detection dictionary with pitch coordinates
                    gt_detection = {
                        'role': role,
                        'team': team,
                        'jersey_number': jersey_number,
                        'bbox_pitch': {
                            'x_bottom_middle': gt_coords[0],
                            'y_bottom_middle': gt_coords[1]
                        }
                    }
                    gt_detections.append(gt_detection)
            
            # Convert to DataFrame if we have detections
            gt_df = pd.DataFrame(gt_detections) if gt_detections else None
            
        # Option 3: Use gt data from vid_data (e.g. when loading from training data)
        # elif 'gt_visible_mask' in vid_data and 'gt_coords' in vid_data:
        #     gt_visible_mask = vid_data['gt_visible_mask'][t].bool()
        #     for i in range(len(gt_visible_mask)):
        #         if gt_visible_mask[i]:
        #             # Extract ground truth data
        #             gt_coords = vid_data['gt_coords'][t, i].tolist()
                    
        #             # Get role if available
        #             role = None
        #             if 'gt_roles' in vid_data:
        #                 role_idx = vid_data['gt_roles'][t, i].argmax().item()
        #                 role = [k for k, v in ROLE_MAP.items() if v == role_idx][0]
                    
        #             # Get team if available
        #             team = None
        #             if 'gt_teams' in vid_data:
        #                 team_idx = vid_data['gt_teams'][t, i].argmax().item()
        #                 team = [k for k, v in TEAM_MAP.items() if v == team_idx][0]
                    
        #             # Get jersey number if available
        #             jersey_number = None
        #             if 'gt_JNs' in vid_data and vid_data['gt_JNs'][t, i].item() > 0:
        #                 jersey_number = int(vid_data['gt_JNs'][t, i].item())
                    
        #             # Create detection dictionary with pitch coordinates
        #             gt_detection = {
        #                 'role': role,
        #                 'team': team,
        #                 'jersey_number': jersey_number,
        #                 'bbox_pitch': {
        #                     'x_bottom_middle': gt_coords[0],
        #                     'y_bottom_middle': gt_coords[1]
        #                 }
        #             }
        #             gt_detections.append(gt_detection)
            
        #     # Convert to DataFrame if we have detections
        #     gt_df = pd.DataFrame(gt_detections) if gt_detections else None
        # else:
        #     gt_df = None
        
        # 2. Input Model DataFrame
        input_detections = []
        for i, bbox in enumerate(bboxes):
            det_idx = torch.where(visible_mask)[0][i].item()
            
            # Get role from vid_data
            role = None
            if 'roles' in vid_data:
                role_idx = vid_data['roles'][t, det_idx].argmax().item()
                role = role_map[role_idx]
                
            # Get team from vid_data
            team = None
            if 'teams' in vid_data:
                team_idx = vid_data['teams'][t, det_idx].argmax().item()
                team = team_map[team_idx]
            
            # Get jersey number from vid_data
            jersey_number = None
            if 'JNs' in vid_data:
                jersey_number = vid_data['JNs'][t, det_idx].item()
                if jersey_number > 0:
                    jersey_number = int(jersey_number)
            
            # Get coordinates from input data
            coords = None
            if 'coords' in vid_data:
                coords = vid_data['coords'][t, det_idx].tolist()
            
            # Create detection dictionary with pitch coordinates
            if coords:
                input_detection = {
                    'bbox_ltwh': bbox,
                    'role': role,
                    'team': team,
                    'jersey_number': jersey_number,
                    'bbox_pitch': {
                        'x_bottom_middle': coords[0],
                        'y_bottom_middle': coords[1]
                    },
                    'track_id': int(vid_data['track_ids'][t, det_idx].item()) if vid_data['track_ids'][t, det_idx] > 0 else None
                }
                input_detections.append(input_detection)
        
        # 3. Output Model DataFrame (using model predictions)
        # First copy input detections, then update with model outputs
        output_detections = deepcopy(input_detections)
        
        # Update output detections with model predictions
        for i, detection in enumerate(output_detections):
            det_idx = torch.where(visible_mask)[0][i].item()
            
            # Update role if task is enabled
            if task_enabled['role']:
                role_idx = outputs['role_logits'][t, det_idx].argmax().item()
                detection['role'] = role_map[role_idx]
            
            # Update team if task is enabled
            if task_enabled['team']:
                team_idx = outputs['team_logits'][t, det_idx].argmax().item()
                detection['team'] = team_map[team_idx]
            
            # Update jersey number if task is enabled
            if task_enabled['jersey']:
                jersey_idx = outputs['jersey_logits'][t, det_idx].argmax().item()
                if jersey_idx > 0:
                    detection['jersey_number'] = jersey_idx
                elif jersey_idx == 0:
                    detection['jersey_number'] = None
            
            # Update coordinates if task is enabled
            if task_enabled['coord']:
                coords = outputs['coords_pred'][t, det_idx].tolist()
                detection['bbox_pitch'] = {
                    'x_bottom_middle': coords[0],
                    'y_bottom_middle': coords[1]
                }
        
        # Convert to DataFrames
        input_df = pd.DataFrame(input_detections)
        output_df = pd.DataFrame(output_detections)
        
        # Draw the pitch visualizations
        if gt_df is not None and not gt_df.empty:
            draw_pitch(vis_img, gt_df, scale=4, group="ground truth")
        
        # Always draw input and output pitches
        draw_pitch(vis_img, input_df, scale=4, group="input")
        draw_pitch(vis_img, output_df, scale=4, group="output")
        
        # Create visualization of inputs - bounding boxes, labels, etc.
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox.astype(int)
            det_idx = torch.where(visible_mask)[0][i].item()
            
            # Get role and team if available
            role = None
            team = None
            
            if task_enabled['role'] and 'role_logits' in outputs:
                role_idx = outputs['role_logits'][t, det_idx].argmax().item()
                role = role_map[role_idx]
                
            if task_enabled['team'] and 'team_logits' in outputs:
                team_idx = outputs['team_logits'][t, det_idx].argmax().item()
                team = team_map[team_idx]
            
            # Get jersey number if available
            jersey_number = None
            if vid_data['JNs'][t, det_idx] > 0:
                jersey_number = int(vid_data['JNs'][t, det_idx].item())
            elif task_enabled['jersey'] and 'jersey_logits' in outputs:
                # Use predicted jersey number if available
                jersey_idx = outputs['jersey_logits'][t, det_idx].argmax().item()
                if jersey_idx > 0:
                    jersey_number = jersey_idx
            
            # Get role and team from vid_data if not already set
            if 'teams' in vid_data and vid_data['teams'][t, det_idx].sum() > 0 and not team:
                team_idx = vid_data['teams'][t, det_idx].argmax().item()
                team = team_map[team_idx]
            
            if 'roles' in vid_data and vid_data['roles'][t, det_idx].sum() > 0 and not role:
                role_idx = vid_data['roles'][t, det_idx].argmax().item()
                role = role_map[role_idx]
            
            # Determine box color based on team and role
            if team and team in team_colors:
                box_color = team_colors[team]
            elif role and role in role_colors:
                box_color = role_colors[role]
            else:
                box_color = detection_color
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), box_color, 1, lineType=cv2.LINE_AA)
            
            # Draw track ID if available
            track_id = None
            if vid_data['track_ids'][t, det_idx] > 0:
                track_id = int(vid_data['track_ids'][t, det_idx].item())
                
                # Draw track ID at the top of the bbox
                draw_text(
                    vis_img,
                    f"ID: {track_id}",
                    (x, y - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color_txt=(255, 255, 255),
                    color_bg=box_color,
                    alignH="l",
                    alignV="b",
                )
            
            # Draw role, team, and jersey number on separate lines
            line_spacing = 20  # Spacing between lines
            
            # Draw role if available
            if role:
                draw_text(
                    vis_img,
                    f"{role}",
                    (x, y + h + 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color_txt=(255, 255, 255),
                    color_bg=box_color,
                    alignH="l",
                    alignV="t",
                )
            
            # Draw team if available
            if team and team != "nan":
                draw_text(
                    vis_img,
                    f"T: {team}",
                    (x, y + h + 5 + line_spacing),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color_txt=(255, 255, 255),
                    color_bg=box_color,
                    alignH="l",
                    alignV="t",
                )
            
            # Draw jersey number if available
            if jersey_number:
                draw_text(
                    vis_img,
                    f"JN: {jersey_number}",
                    (x, y + h + 5 + 2 * line_spacing),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color_txt=(255, 255, 255),
                    color_bg=box_color,
                    alignH="l",
                    alignV="t",
                )
                
        # Convert back to BGR for saving with OpenCV
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Save visualization
        output_path = os.path.join(vid_output_dir, f"frame_{int(image_id):06d}.jpg")
        cv2.imwrite(output_path, vis_img)


# Helper function to draw text with background
def draw_text(img, text, pos, fontFace, fontScale, thickness, color_txt=(0, 0, 0), 
             color_bg=None, alpha_bg=0.7, alignH="l", alignV="b"):
    """
    Draw text with background on image
    
    Args:
        img: Image to draw on
        text: Text to draw
        pos: Position (x, y)
        fontFace: Font face
        fontScale: Font scale
        thickness: Text thickness
        color_txt: Text color
        color_bg: Background color
        alpha_bg: Background alpha
        alignH: Horizontal alignment ('l', 'c', 'r')
        alignV: Vertical alignment ('t', 'c', 'b')
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(
        text, fontFace=fontFace, fontScale=fontScale, thickness=thickness
    )
    text_w, text_h = text_size
    
    # Adjust y position based on vertical alignment
    if alignV == "b":
        txt_pos_y = y
    elif alignV == "t":
        txt_pos_y = y + text_h
    elif alignV == "c":
        txt_pos_y = y + text_h // 2
    else:
        raise ValueError("alignV must be one of 't', 'b', 'c'")

    # Adjust x position based on horizontal alignment
    if alignH == "l":
        txt_pos_x = x
    elif alignH == "r":
        txt_pos_x = x - text_w
    elif alignH == "c":
        txt_pos_x = x - text_w // 2
    else:
        raise ValueError("alignH must be one of 'l', 'r', 'c'")

    # Draw background
    if color_bg is not None:
        padding = 3
        rect_pos_x = txt_pos_x - padding
        rect_pos_y = txt_pos_y + padding
        rect_w = text_w + 2 * padding
        rect_h = text_h + 2 * padding
        
        # Calculate background rectangle coordinates
        x_start = rect_pos_x
        y_start = txt_pos_y - text_h - padding
        x_end = rect_pos_x + rect_w
        y_end = rect_pos_y
        
        # Ensure coordinates are within image bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(img.shape[1] - 1, x_end)
        y_end = min(img.shape[0] - 1, y_end)
        
        # Draw background rectangle
        if x_end > x_start and y_end > y_start:
            background_area = img[y_start:y_end, x_start:x_end].copy()
            background_color = np.array(color_bg, dtype=background_area.dtype)
            background = np.ones_like(background_area) * background_color
            img[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                background_area, 1 - alpha_bg, background, alpha_bg, 0.0
            )

    # Draw text
    cv2.putText(
        img,
        text,
        (txt_pos_x, txt_pos_y),
        fontFace=fontFace,
        fontScale=fontScale,
        color=color_txt,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    
    return text_size


# Helper function to draw frame count
def draw_frame_count(img, frame_idx, total_frames):
    """Draw frame count on image"""
    draw_text(
        img,
        f"Frame: {frame_idx+1}/{total_frames}",
        (10, 25),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        thickness=2,
        color_txt=(255, 255, 255),
        color_bg=(0, 0, 0),
        alignH="l",
        alignV="t",
    )


# Function to draw the pitch visualization
def draw_pitch(patch, detections, scale=3, delta=255, group="predictions", vis_pitch_corners=False, pitch_corners=None):
    """Draw a top-down view of the pitch with player positions
    
    Args:
        patch: Image to draw on
        detections: DataFrame containing detections with pitch coordinates
        scale: Scale factor for pitch visualization
        delta: Horizontal offset for multiple pitch visualizations
        group: Group label for the visualization
        vis_pitch_corners: Whether to visualize pitch corners
        pitch_corners: Pitch corner coordinates if available
    """
    if detections is None or len(detections) == 0:
        return
        
    # Define pitch dimensions
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    
    # Position the pitch visualizations side by side
    # Ground truth on the left, input in the middle, output on the right
    if group == "input":
        sign = -1  # Left side
        offset = -delta * 2  # Move further left
    elif group == "output":
        sign = 0  # Center
        offset = 0
    elif group == "ground truth":
        sign = 1  # Right side
        offset = delta * 2  # Move further right
    else:
        sign = 0  # Default to center
        offset = 0
    
    # Calculate radar position
    image_height, image_width = patch.shape[:2]
    radar_center_x = int(image_width/2 + offset)
    radar_center_y = int(image_height - pitch_height * scale / 2)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(image_height - pitch_height * scale)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    
    # Check if radar would be out of bounds
    if radar_top_x < 0:
        print(f"Warning: Radar for {group} would be out of bounds (left edge), adjusting position")
    elif radar_top_x + radar_width > image_width:
        print(f"Warning: Radar for {group} would be out of bounds (right edge), adjusting position")
    elif radar_top_y < 0:
        print(f"Warning: Radar for {group} would be out of bounds (top edge), adjusting position")
    elif radar_top_y + radar_height > image_height:
        print(f"Warning: Radar for {group} would be out of bounds (bottom edge), adjusting position")
        
    # Adjust position to fit within image
    if radar_top_x < 0 or radar_top_x + radar_width > image_width or radar_top_y < 0 or radar_top_y + radar_height > image_height:
        radar_top_x = max(0, min(radar_top_x, image_width - radar_width))
        radar_top_y = max(0, min(radar_top_y, image_height - radar_height))
        radar_width = min(radar_width, image_width - radar_top_x)
        radar_height = min(radar_height, image_height - radar_top_y)
        radar_center_x = radar_top_x + radar_width // 2
        radar_center_y = radar_top_y + radar_height // 2
    
    # Load radar image
    if PITCH_FILE and os.path.exists(PITCH_FILE):
        radar_img = cv2.resize(cv2.imread(PITCH_FILE), (radar_width, radar_height))
        
        # Invert colors for better visibility
        radar_img = cv2.bitwise_not(radar_img)
        
        # Add team color indicators on sides
        cv2.line(radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255))  # Blue for left team
        cv2.line(radar_img, (radar_img.shape[1]-1, 0), (radar_img.shape[1]-1, radar_img.shape[0]), thickness=6, color=(255, 0, 0))  # Red for right team
    else:
        # Create a blank pitch with field markings
        radar_img = np.ones((radar_height, radar_width, 3), dtype=np.uint8) * 255
        
        # Draw field outline
        cv2.rectangle(radar_img, (0, 0), (radar_width-1, radar_height-1), (0, 128, 0), 2)
        
        # Draw center line
        cv2.line(radar_img, (radar_width//2, 0), (radar_width//2, radar_height), (0, 128, 0), 2)
        
        # Draw center circle
        cv2.circle(radar_img, (radar_width//2, radar_height//2), radar_height//5, (0, 128, 0), 2)
        
        # Add team color indicators on sides
        cv2.line(radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255))  # Blue for left team
        cv2.line(radar_img, (radar_img.shape[1]-1, 0), (radar_img.shape[1]-1, radar_img.shape[0]), thickness=6, color=(255, 0, 0))  # Red for right team
    
    # Blend radar image onto patch
    alpha = 0.5
    # Apply double blending for better visibility
    try:
        patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
        :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
        :], 1-alpha, radar_img, alpha, 0.0)
        patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
        :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
        :], 1-alpha, radar_img, alpha, 0.0)
    except:
        print(f"Warning: Could not blend radar image, patch shape: {patch.shape}, radar region: {radar_top_y}:{radar_top_y + radar_height}, {radar_top_x}:{radar_top_x + radar_width}")
    
    # Draw title for the visualization
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        thickness=2,
        color_txt=(255, 255, 255),
        color_bg=None,
        alignH="c",
        alignV="b",
    )
    
    # Draw players on the radar
    for _, detection in detections.iterrows():
        if 'bbox_pitch' not in detection or not isinstance(detection['bbox_pitch'], dict):
            continue
            
        # Get player position
        x_middle = detection['bbox_pitch'].get('x_bottom_middle', None)
        y_middle = detection['bbox_pitch'].get('y_bottom_middle', None)
        
        if x_middle is None or y_middle is None:
            continue
            
        # Clip coordinates to prevent extreme values
        x_middle = np.clip(x_middle, -10000, 10000)
        y_middle = np.clip(y_middle, -10000, 10000)
        
        # Normalize coordinates to radar space
        radar_x = int(radar_center_x + x_middle * scale)
        radar_y = int(radar_center_y + y_middle * scale)
        
        # Check if point is within the radar
        if radar_x < radar_top_x or radar_x >= radar_top_x + radar_width or radar_y < radar_top_y or radar_y >= radar_top_y + radar_height:
            continue
        
        # # Determine color based on team and role
        # color = (0, 0, 0)  # Default black
        # if 'team' in detection and 'role' in detection:
        #     if detection['role'] == 'referee':
        #         color = (255, 255, 0)  # Yellow for referee
        #     elif detection['team'] == 'left':
        #         color = (0, 0, 255)  # Blue for left team
        #     elif detection['team'] == 'right':
        #         color = (255, 0, 0)  # Red for right team
        
        # # Determine label text based on role and jersey number
        # text = None
        # if 'role' in detection:
        #     if detection['role'] == 'goalkeeper':
        #         text = "GK"
        #     elif detection['role'] == 'referee':
        #         text = "REF"
        #         color = (255, 255, 0)  # Yellow for referee
        #     elif 'jersey_number' in detection and detection['jersey_number'] is not None:
        #         # Show jersey number for players
        #         text = f"{detection['jersey_number']}"
        
        if "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
            
        text = None
        if "jersey_number" in detection:
            if "role" in detection and detection.role == "player":
                if (isinstance(detection.jersey_number, (float, int)) and np.isnan(detection.jersey_number)) or detection.jersey_number is None or detection.jersey_number == 0:
                    text = None
                else:
                    text = f"{int(detection.jersey_number)}"
        
        if "role" in detection:
            if detection.role == "goalkeeper":
                text = "GK"
            elif detection.role == "referee":
                text = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                text = "OT"
                color = (0, 255, 0)
        
        # Draw player marker
        if text:
            # Draw text for the player
            draw_text(
                patch,
                text,
                (radar_x, radar_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.11*scale,
                thickness=1,
                color_txt=color,
                color_bg=None,
                alignH="c",
                alignV="c",
            )
        else:
            # Draw circle for the player
            cv2.circle(
                patch,
                (radar_x, radar_y),
                scale,
                color=color,
                thickness=-1  # Filled circle
            )
    
    # Draw pitch corners if available
    if vis_pitch_corners and pitch_corners is not None:
        # Draw the four corners of the pitch
        try:
            pitch_corners = np.clip(pitch_corners, -10000, 10000)
            
            # Draw the four lines connecting the corners
            for i in range(4):
                pt1 = (
                    radar_center_x + int(pitch_corners[i, 0] * scale / 2),
                    radar_center_y + int(pitch_corners[i, 1] * scale / 2)
                )
                pt2 = (
                    radar_center_x + int(pitch_corners[(i+1)%4, 0] * scale / 2),
                    radar_center_y + int(pitch_corners[(i+1)%4, 1] * scale / 2)
                )
                cv2.line(patch, pt1, pt2, color=(255, 218, 185), thickness=2)
        except:
            print("Warning: Could not draw pitch corners")


def process_gt_frame(gt_data: Dict, image_id: str, max_detections: int) -> Dict:
    """Process ground truth data for a single frame specific to visualization
    
    This function extracts annotations for a single frame from the ground truth data,
    focusing on information needed for visualization purposes.
    
    Args:
        gt_data: Ground truth data dictionary
        image_id: Image ID to process
        max_detections: Maximum number of detections to process
        
    Returns:
        DataFrame containing processed ground truth data for the frame, or None if no data
    """
    if gt_data is None:
        return None
        
    frame_data = {
        'bbox_ltwh': [],
        'coords': [],
        'roles': [],
        'teams': [],
        'jersey_numbers': [],
        'track_ids': [],
    }
    
    # Try different image ID formats
    # Some datasets use numeric IDs, others use string IDs with prefixes
    image_id_variants = [
        image_id,                                  # Original ID
        str(int(image_id)),                        # Remove leading zeros
        f"{int(image_id):06d}",                    # Six-digit format
        f"{int(image_id[-6:]):06d}" if len(image_id) > 6 else image_id  # Last 6 digits
    ]
    
    # Check if any of the image ID variants are labeled
    is_labeled = False
    for img_id in image_id_variants:
        for image_anno in gt_data['images']:
            if str(image_anno['image_id']) == img_id:
                is_labeled = image_anno['has_labeled_person'] and image_anno['has_labeled_pitch'] and image_anno['has_labeled_camera']
                image_id = img_id  # Use the matching image ID for further processing
                break
        if is_labeled:
            break
    
    if not is_labeled:
        return None
    
    # Process annotations for this image
    for anno in gt_data['annotations']:
        if str(anno['image_id']) != image_id:
            continue
        
        if anno['supercategory'] != 'object' or anno['attributes']['role'] == 'ball':
            continue
        
        # Get bounding box
        bbox = anno['bbox_image']
        frame_data['bbox_ltwh'].append([bbox['x'], bbox['y'], bbox['w'], bbox['h']])
        
        # Get coordinates
        bbox_pitch = anno['bbox_pitch']
        # Clip coordinates to ensure they're in a reasonable range
        x = np.clip(bbox_pitch['x_bottom_middle'], COORD_X_MIN, COORD_X_MAX)
        y = np.clip(bbox_pitch['y_bottom_middle'], COORD_Y_MIN, COORD_Y_MAX)
        frame_data['coords'].append([x, y])
        
        # Get role
        role = anno['attributes']['role']
        frame_data['roles'].append(role)
        
        # Get team
        team = anno['attributes']['team']
        frame_data['teams'].append(team)
        
        # Get jersey number
        jn = anno['attributes'].get('jersey', None)
        frame_data['jersey_numbers'].append(int(jn) if jn and 0 < int(jn) < 100 else None)
        
        # Get track ID
        track_id = anno.get('track_id', None)
        frame_data['track_ids'].append(track_id if track_id and 0 < track_id < 150 else None)
        
        # Stop if we've reached max detections
        if len(frame_data['bbox_ltwh']) >= max_detections:
            break
    
    # Convert to DataFrame for easier processing
    if len(frame_data['bbox_ltwh']) > 0:
        return pd.DataFrame({
            'bbox_ltwh': frame_data['bbox_ltwh'],
            'bbox_pitch': [{'x_bottom_middle': x[0], 'y_bottom_middle': x[1]} for x in frame_data['coords']],
            'role': frame_data['roles'],
            'team': frame_data['teams'],
            'jersey_number': frame_data['jersey_numbers'],
            'track_id': frame_data['track_ids']
        })
    else:
        return None


def save_frames_as_video(frames_dir: str, output_path: str, fps: int = 25):
    """Save a sequence of frames as a video file
    
    Args:
        frames_dir: Directory containing frames (sorted by name)
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    # Get all frame files
    frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    if first_frame is None:
        print(f"Could not read frame: {os.path.join(frames_dir, frame_files[0])}")
        return
    
    height, width, _ = first_frame.shape
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_file in tqdm(frame_files, desc=f"Creating video for {os.path.basename(frames_dir)}"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
    
    # Release video writer
    video.release()
    print(f"Video saved to {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup visualization
    should_visualize = args.visualize or args.visualize_video
    if should_visualize:
        if not args.img_dir:
            raise ValueError("--img_dir must be specified when --visualize or --visualize_video is set")
        
        # Create visualization output directory for images
        if args.visualize:
            if args.vis_output_dir:
                vis_output_dir = args.vis_output_dir
            else:
                # If visualization output directory not specified, use checkpoint directory visualization folder
                checkpoint_dir = os.path.dirname(args.checkpoint)
                vis_output_dir = os.path.join(checkpoint_dir, "visualization")
            
            os.makedirs(vis_output_dir, exist_ok=True)
            print(f"Visualization results will be saved to {vis_output_dir}")
        else:
            vis_output_dir = None
        
        # Create video output directory if needed
        if args.visualize_video:
            if args.video_output_dir:
                video_output_dir = args.video_output_dir
            else:
                # If video output directory not specified, use checkpoint directory visualization_video folder
                checkpoint_dir = os.path.dirname(args.checkpoint)
                video_output_dir = os.path.join(checkpoint_dir, "visualization_video")
            
            os.makedirs(video_output_dir, exist_ok=True)
            print(f"Visualization videos will be saved to {video_output_dir}")
        else:
            video_output_dir = None
    
    # Create model
    print(f"Creating model of type: {config['model']['type']}")
    model = create_model(config)
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Get task flags from config
    task_enabled = {
        'track': config['training']['tasks']['track'],
        'role': config['training']['tasks']['role'],
        'team': config['training']['tasks']['team'],
        'jersey': config['training']['tasks']['jersey'],
        'coord': config['training']['tasks']['coord'],
        'missing': config['training']['tasks']['missing']
    }
    print(f"Enabled tasks: {[k for k, v in task_enabled.items() if v]}")
    
    # Load normalization parameters
    normalize_bbox = config['data'].get('normalize_bbox', False)
    normalize_coords = config['data'].get('normalize_coords', True)
    normalization_method = config['data'].get('normalization_method', 'minimax')
    bbox_mean = np.array(config['data'].get('bbox_mean')) if config['data'].get('bbox_mean') is not None else None
    bbox_std = np.array(config['data'].get('bbox_std')) if config['data'].get('bbox_std') is not None else None
    coord_mean = np.array(config['data'].get('coord_mean')) if config['data'].get('coord_mean') is not None else None
    coord_std = np.array(config['data'].get('coord_std')) if config['data'].get('coord_std') is not None else None
    
    # Load metadata
    with open(args.metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Get video IDs from metadata
    if args.split == 'valid':
        split = 'validation'
        vid_id_list = [vid["name"].split('-')[1] for vid in metadata[split]]
    elif args.split == 'train+valid':
        # Combine video IDs from both train and validation splits
        train_vid_ids = [vid["name"].split('-')[1] for vid in metadata['train']]
        valid_vid_ids = [vid["name"].split('-')[1] for vid in metadata['validation']]
        vid_id_list = train_vid_ids + valid_vid_ids
        print(f"Using combined train+valid split with {len(train_vid_ids)} train videos and {len(valid_vid_ids)} validation videos")
    else:
        split = args.split
        vid_id_list = [vid["name"].split('-')[1] for vid in metadata[split]]
    # vid_id_list = ['119']
    # vid_id_list = [vid_id_list[0]]
    
    
    # Create output pklz path
    output_pklz_path = os.path.join(args.output_dir, f'refined_{os.path.basename(args.input_pklz)}')
    
    # assert not os.path.exists(output_pklz_path), f"Output file {output_pklz_path} already exists. It will be overwritten."
    if os.path.exists(output_pklz_path):
        os.remove(output_pklz_path)
    
    # Process each video
    with zipfile.ZipFile(args.input_pklz) as input_zf, zipfile.ZipFile(output_pklz_path, 'w') as output_zf:
        for vid in tqdm(vid_id_list, desc="Processing videos"):
            # Load video data
            with input_zf.open(f'{vid}.pkl') as f:
                preds = pickle.load(f)
            with input_zf.open(f'{vid}_image.pkl') as f:
                image_preds = pickle.load(f)
            
            # Process video data
            vid_data, detection_mappings = process_pipeline_video(
                vid, preds, image_preds,
                config['data']['max_frames'], config['data']['max_detections_per_frame'],
                ROLE_MAP, TEAM_MAP,
                coord_x_min=COORD_X_MIN, 
                coord_x_max=COORD_X_MAX, 
                coord_y_min=COORD_Y_MIN, 
                coord_y_max=COORD_Y_MAX
            )
            
            vis_vid_data = deepcopy(vid_data)
            
            # Manually normalize coordinates if needed
            if normalize_coords:
                # Get all visible coordinates
                visible_mask = vid_data['visible_mask'].bool()
                
                # Process all coordinates in a vectorized way
                for t in range(vid_data['coords'].shape[0]):
                    # Get mask for visible detections in this frame
                    frame_visible_mask = visible_mask[t]
                    
                    if frame_visible_mask.any():
                        # Get visible coordinates for this frame
                        frame_coords = vid_data['coords'][t][frame_visible_mask]
                        
                        # Normalize them
                        normalized_coords = normalize_coordinates(
                            frame_coords,
                            normalization_method,
                            coord_mean=coord_mean,
                            coord_std=coord_std,
                            coord_x_min=COORD_X_MIN,
                            coord_x_max=COORD_X_MAX,
                            coord_y_min=COORD_Y_MIN,
                            coord_y_max=COORD_Y_MAX
                        )
                        
                        # Update the coordinates in vid_data
                        vid_data['coords'][t][frame_visible_mask] = normalized_coords
            
            # Run inference
            tqdm.write(f"Processing video: {vid}")
            outputs = inference_on_video(model, vid_data, device, config, config['data']['max_clip_frames'])
            
            # Update predictions
            updated_preds = update_predictions(preds, outputs, detection_mappings, task_enabled)
            
            # Visualization
            if should_visualize:
                image_id_list = sorted(image_preds.id.unique().tolist(), key=lambda x: int(x))
                
                # Determine the correct split directory for this video if using train+valid
                current_split = args.split
                if args.split == 'train+valid':
                    # Check if this video is in train or validation
                    if vid in [v["name"].split('-')[1] for v in metadata['train']]:
                        current_split = 'train'
                    else:
                        current_split = 'valid'  # Will be mapped to 'validation' when loading GT
                
                img_dir = os.path.join(args.img_dir, current_split, f'SNGS-{vid}', 'img1')
                
                # Check if image directory exists
                if not os.path.exists(img_dir):
                    print(f"Warning: Image directory {img_dir} not found, skipping visualization for video {vid}")
                else:
                    # Load ground truth data if available
                    if args.gt_dir:
                        gt_root = args.gt_dir
                    else:
                        gt_root = args.img_dir # Assuming GT data is in parent of img_dir
                    
                    # Use the imported function to load ground truth data - use the correct split for this video
                    gt_data_raw = load_gt_data(gt_root, current_split, vid)
                    gt_data_processed = process_gt_video(
                        gt_data_raw, image_id_list,
                        config['data']['max_frames'], 
                        config['data']['max_detections_per_frame'],
                        ROLE_MAP, TEAM_MAP,
                        COORD_X_MIN, COORD_X_MAX, COORD_Y_MIN, COORD_Y_MAX
                    )
                    
                    # Visualization directory for images
                    if args.visualize:
                        visualize_video_inputs_outputs(
                            img_dir, 
                            image_id_list, 
                            vis_vid_data, 
                            outputs, 
                            vis_output_dir, 
                            vid, 
                            task_enabled,
                            gt_data_processed  # Pass processed GT data for visualization
                        )
                    
                    # Create video if requested
                    if args.visualize_video:
                        # If images weren't generated for visualization, generate them for video
                        if not args.visualize:
                            temp_vis_dir = os.path.join(video_output_dir, "temp_frames", f'SNGS-{vid}')
                            os.makedirs(temp_vis_dir, exist_ok=True)
                            visualize_video_inputs_outputs(
                                img_dir, 
                                image_id_list, 
                                vis_vid_data, 
                                outputs, 
                                os.path.dirname(temp_vis_dir), 
                                vid, 
                                task_enabled,
                                gt_data_processed  # Pass processed GT data for visualization
                            )
                            # Create video from temp frames
                            video_path = os.path.join(video_output_dir, f'SNGS-{vid}.mp4')
                            save_frames_as_video(temp_vis_dir, video_path, args.video_fps)
                            # Clean up temp frames if needed
                            shutil.rmtree(os.path.dirname(temp_vis_dir))
                        else:
                            # Use frames that were already generated for visualization
                            frames_dir = os.path.join(vis_output_dir, f'SNGS-{vid}')
                            video_path = os.path.join(video_output_dir, f'SNGS-{vid}.mp4')
                            save_frames_as_video(frames_dir, video_path, args.video_fps)
            
            # Write updated predictions and image_preds to the output zip
            # Create temporary files
            temp_pred_path = f'temp_{vid}.pkl'
            temp_image_path = f'temp_{vid}_image.pkl'
            
            # Save updated predictions
            with open(temp_pred_path, 'wb') as f:
                pickle.dump(updated_preds, f)
            
            # Save image predictions
            with open(temp_image_path, 'wb') as f:
                pickle.dump(image_preds, f)
            
            # Add files to the output zip
            output_zf.write(temp_pred_path, f'{vid}.pkl')
            output_zf.write(temp_image_path, f'{vid}_image.pkl')
            
            # Remove temporary files
            os.remove(temp_pred_path)
            os.remove(temp_image_path)
    
    print(f"Inference completed successfully. Updated predictions saved to {output_pklz_path}")
    if args.visualize:
        print(f"Visualizations saved to {vis_output_dir}")
    if args.visualize_video:
        print(f"Visualization videos saved to {video_output_dir}")


if __name__ == '__main__':
    main() 