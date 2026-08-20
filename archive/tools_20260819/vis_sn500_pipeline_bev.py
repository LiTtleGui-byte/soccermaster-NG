#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化SN500数据集的pkl文件 - Pitch View版本
在球场俯视图上可视化人物位置、track_id、role、team
"""

import os
import sys
import pickle
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from soccermaster.data.soccernet_gsr_reid import role_mapping

# 反向映射role_mapping
role_mapping_reverse = {v: k for k, v in role_mapping.items()}

# Pitch文件路径
PITCH_FILE = "/mnt/vision_user/sports/code/Soccer-Backbone/paper/pitch.png"


def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0, 
              thickness=2, color_txt=(255, 255, 255), color_bg=None,
              alignH="l", alignV="t"):
    """
    在图像上绘制文本，支持对齐方式
    
    Args:
        img: 图像
        text: 文本内容
        pos: 位置 (x, y)
        font: 字体
        scale: 缩放比例
        thickness: 厚度
        color_txt: 文本颜色
        color_bg: 背景颜色，None表示无背景
        alignH: 水平对齐 'l'(左), 'c'(中), 'r'(右)
        alignV: 垂直对齐 't'(上), 'c'(中), 'b'(下)
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    
    # 水平对齐
    if alignH == "c":
        x = x - text_w // 2
    elif alignH == "r":
        x = x - text_w
    
    # 垂直对齐
    if alignV == "c":
        y = y + text_h // 2
    elif alignV == "b":
        y = y
    else:  # 't'
        y = y + text_h
    
    # 绘制背景
    if color_bg is not None:
        cv2.rectangle(img, (x - 2, y - text_h - 2), 
                     (x + text_w + 2, y + baseline + 2), 
                     color_bg, -1)
    
    # 绘制文本
    cv2.putText(img, text, (x, y), font, scale, color_txt, thickness, cv2.LINE_AA)


def extract_frame_info(frame_data, frame_data_image):
    """
    从帧数据中提取可视化所需的结构化信息（Pitch View版本）
    
    Args:
        frame_data: 帧数据（DataFrame）
        frame_data_image: 帧数据图像（DataFrame）
    
    Returns:
        dict: 包含以下字段的字典
            - persons: list of dict, 每个dict包含:
                - x_middle: float (球场坐标系)
                - y_middle: float (球场坐标系)
                - track_id: int
                - role: str
                - team: str or None
                - jersey_number: int or None
            - valid: bool，是否成功读取
    """
    # 提取人物信息
    persons = []
    
    for id, row in frame_data.iterrows():
        role = row['role']
        
        # 跳过球
        if role == 'ball':
            continue
        
        # 提取bbox_pitch信息
        bbox_pitch = row.get('bbox_pitch')
        if bbox_pitch is None or not isinstance(bbox_pitch, dict):
            continue
        
        x_middle = bbox_pitch.get('x_bottom_middle')
        y_middle = bbox_pitch.get('y_bottom_middle')
        
        if x_middle is None or y_middle is None:
            continue
        
        # 裁剪坐标到合理范围
        x_middle = np.clip(x_middle, -10000, 10000)
        y_middle = np.clip(y_middle, -10000, 10000)
        
        # 提取其他属性
        track_id = row.get('track_id')
        jersey_number = row.get('jersey_number')
        team = row.get('team')
        
        persons.append({
            'x_middle': x_middle,
            'y_middle': y_middle,
            'track_id': int(track_id) if not pd.isna(track_id) else None,
            'role': role,
            'team': team if not pd.isna(team) else None,
            'jersey_number': int(jersey_number) if not pd.isna(jersey_number) else None
        })
    
    return {
        'valid': True,
        'persons': persons
    }


def visualize_structured_data(frame_info, pitch_file=PITCH_FILE, scale=4):
    """
    在球场俯视图上可视化人物位置（Pitch View版本）
    
    Args:
        frame_info: 包含可视化信息的字典，由 extract_frame_info 函数生成
            - persons: list of dict
            - valid: bool
        pitch_file: 球场图片路径
        scale: 缩放比例
    
    Returns:
        image: 可视化后的图像
    """
    if not frame_info['valid']:
        return None
    
    # 球场尺寸（米）
    pitch_width = int(105 + 2 * 4.86)  # 球场宽度 + 2 * 边距
    pitch_height = int(68 + 2 * 1.94)   # 球场高度 + 2 * 边距
    
    # 读取并准备球场图片
    if pitch_file is not None and os.path.exists(pitch_file):
        pitch_img = cv2.imread(str(pitch_file))
        if pitch_img is not None:
            # 调整大小
            pitch_img = cv2.resize(pitch_img, (pitch_width * scale, pitch_height * scale))
        else:
            # 如果读取失败，创建白色背景
            pitch_img = np.ones((pitch_height * scale, pitch_width * scale, 3), dtype=np.uint8) * 255
    else:
        # 创建白色背景
        pitch_img = np.ones((pitch_height * scale, pitch_width * scale, 3), dtype=np.uint8) * 255
    
    # 计算球场中心点（在图像坐标系中）
    radar_center_x = pitch_width * scale // 2
    radar_center_y = pitch_height * scale // 2
    
    # 绘制所有人物
    for person in frame_info['persons']:
        x_middle = person['x_middle']
        y_middle = person['y_middle']
        track_id = person['track_id']
        role = person['role']
        team = person.get('team')
        jersey_number = person.get('jersey_number')
        
        # 计算在图像上的位置
        pixel_x = radar_center_x + int(x_middle * scale)
        pixel_y = radar_center_y + int(y_middle * scale)
        
        # 确定颜色和显示文本
        if role == "referee":
            color = (0, 165, 255)  # 橙色 (BGR)
            text = "RE"
        else:
            # 根据team确定颜色
            if team == "left":
                color = (0, 0, 255)  # 红色
            elif team == "right":
                color = (255, 0, 0)  # 蓝色
            else:
                color = (0, 0, 0)  # 黑色（未知team）
            
            # 使用track_id作为显示文本
            text = str(track_id) if track_id is not None else "?"
        
        # 绘制文本或圆点
        if text is not None:
            draw_text(
                pitch_img,
                text,
                (pixel_x, pixel_y),
                scale=0.1 * scale,
                thickness=3,
                color_txt=color,
                alignH="c",
                alignV="c"
            )
        else:
            # 如果没有文本，绘制圆点
            cv2.circle(
                pitch_img,
                (pixel_x, pixel_y),
                scale,
                color=color,
                thickness=-1
            )
    
    return pitch_img


def visualize_frame(frame_data, frame_data_image, output_path, scale=4):
    """
    可视化单帧图像在pitch view上（主函数）
    
    Args:
        frame_data: 帧数据
        frame_data_image: 帧数据图像
        output_path: 输出路径
        scale: 缩放比例
    """
    # 第一步：提取结构化信息
    frame_info = extract_frame_info(frame_data, frame_data_image)
    
    if not frame_info['valid']:
        return
    
    # 第二步：在pitch view上可视化
    visualized_image = visualize_structured_data(frame_info, scale=scale)
    
    if visualized_image is None:
        return
    
    # 保存图像
    cv2.imwrite(output_path, visualized_image)


def main():
    parser = argparse.ArgumentParser(description='可视化SN500数据集 (Pitch View)')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/paper/pipeline_bev_vis',
                       help='输出目录')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='最大处理帧数，-1表示处理所有帧')
    parser.add_argument('--scale', type=int, default=16,
                       help='球场图片的缩放比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从pkl路径中提取序列名称
    sequence_name = 'SNGS-124'
    
    # 创建序列特定的输出目录
    sequence_output_dir = output_dir / sequence_name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    
    pklz_path = '/mnt/vision_user/sports/code/Soccer-Backbone/paper/yolo_v8x6_person_nbjw_cam_qwen25vl_legibility_filter2_qwen25vl_role_7b_remove_outside_concat_jn_reid_testset_72b/sn-gamestate.pklz'
    
    # 加载pkl数据
    zf = zipfile.ZipFile(pklz_path)
    data = pickle.load(zf.open(f'{sequence_name[-3:]}.pkl'))
    
    # 处理track_id，使其连续
    # 获取所有唯一的track_id（排除NaN）
    unique_track_ids = data['track_id'].dropna().unique()
    unique_track_ids = sorted(unique_track_ids)
    
    # 创建从旧track_id到新track_id的映射
    track_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_track_ids, start=1)}
    
    # 应用映射到数据中
    data['track_id'] = data['track_id'].map(lambda x: track_id_mapping.get(x, x) if pd.notna(x) else x)
    
    print(f"Track ID remapping complete: {len(track_id_mapping)} unique tracks")
    
    data_image = pickle.load(zf.open(f'{sequence_name[-3:]}_image.pkl'))
    
    # 获取所有帧ID并排序
    image_ids = sorted(data_image['id'].to_list())
    
    # 限制处理的帧数
    if args.max_frames > 0:
        image_ids = image_ids[:args.max_frames]
    
    print(f"Processing {len(image_ids)} frames...")
    
    # 处理每一帧
    for frame_id in tqdm(image_ids, desc="Visualizing frames"):
        frame_data = data.loc[data['image_id'] == frame_id]
        frame_data_image = data_image.loc[data_image['id'] == frame_id]
        
        # 构建输出路径
        output_filename = f"frame_{frame_id[-6:]}.jpg"
        output_path = sequence_output_dir / output_filename
        
        # 可视化并保存
        visualize_frame(frame_data, frame_data_image, str(output_path), scale=args.scale)
    
    print(f"Visualization complete! Results saved to: {sequence_output_dir}")
    print(f"Total frames processed: {len(image_ids)}")


if __name__ == '__main__':
    main()

