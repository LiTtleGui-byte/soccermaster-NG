#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化SoccerNetGSR数据集的Ground Truth - Pitch View版本
在球场俯视图上可视化人物位置、track_id、role、team
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

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


def extract_frame_info(frame_annotations, image_path, frame_height=1080, frame_width=1920):
    """
    从Ground Truth annotations中提取可视化所需的结构化信息（Pitch View版本）
    
    Args:
        frame_annotations: 该帧的所有annotations列表（来自Labels-GameState.json）
        image_path: 图像路径（用于记录，实际不需要读取）
        frame_height: 图像高度（用于记录）
        frame_width: 图像宽度（用于记录）
    
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
    
    for anno in frame_annotations:
        if anno['supercategory'] == 'object':
            role = anno['attributes']['role']
            if role == 'ball':
                continue
            
            # 提取bbox_pitch信息
            bbox_pitch = anno.get('bbox_pitch')
            if bbox_pitch is None or not isinstance(bbox_pitch, dict):
                continue
            
            x_middle = bbox_pitch.get('x_bottom_middle')
            y_middle = bbox_pitch.get('y_bottom_middle')
            
            if x_middle is None or y_middle is None:
                continue
            
            # 裁剪坐标到合理范围
            x_middle = np.clip(x_middle, -10000, 10000)
            y_middle = np.clip(y_middle, -10000, 10000)
            
            track_id = anno['track_id']
            jersey = anno['attributes'].get('jersey')
            team = anno['attributes'].get('team')
            
            persons.append({
                'x_middle': x_middle,
                'y_middle': y_middle,
                'track_id': track_id,
                'role': role,
                'team': team,
                'jersey_number': jersey
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
    # pitch_width = 105 + 2 * 10  # 球场宽度 + 2 * 边距
    # pitch_height = 68 + 2 * 5   # 球场高度 + 2 * 边距
    pitch_width = int(105 + 2 * 4.86)  # 球场宽度 + 2 * 边距
    pitch_height = int(68 + 2 * 1.94)   # 球场高度 + 2 * 边距
    
    # 读取并准备球场图片
    if pitch_file is not None and os.path.exists(pitch_file):
        pitch_img = cv2.imread(str(pitch_file))
        if pitch_img is not None:
            # 调整大小
            pitch_img = cv2.resize(pitch_img, (pitch_width * scale, pitch_height * scale))
            # 将绿色替换为指定颜色 #B0DC66 -> RGB(176,220,102) -> BGR(102,220,176)
            # hsv = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2HSV)
            # # 定义绿色的HSV范围（缩小范围，避免匹配到白色）
            # lower_green = np.array([40, 60, 60])
            # upper_green = np.array([80, 200, 180])
            # # 创建绿色掩码
            # mask = cv2.inRange(hsv, lower_green, upper_green)
            # # 将绿色区域替换为指定颜色 #B0DC66 -> RGB(176,220,102) -> BGR(102,220,176)
            # pitch_img[mask > 0] = (102, 220, 176)  # BGR格式
            # 反色
            # pitch_img = cv2.bitwise_not(pitch_img)
            # 绘制左右边界线
            # 左边界（红色，代表left队）
            # cv2.line(pitch_img, (0, 0), (0, pitch_img.shape[0]), 
            #         thickness=6, color=(0, 0, 255))
            # 右边界（蓝色，代表right队）
            # cv2.line(pitch_img, (pitch_img.shape[1]-1, 0), 
            #         (pitch_img.shape[1]-1, pitch_img.shape[0]), 
            #         thickness=6, color=(255, 0, 0))
        else:
            # 如果读取失败，创建白色背景
            pitch_img = np.ones((pitch_height * scale, pitch_width * scale, 3), dtype=np.uint8) * 255
    else:
        # 创建白色背景
        pitch_img = np.ones((pitch_height * scale, pitch_width * scale, 3), dtype=np.uint8) * 255
    
    # 计算球场中心点（在图像坐标系中）
    radar_center_x = pitch_width * scale // 2
    radar_center_y = pitch_height * scale // 2
    
    # 绘制标题
    # draw_text(
    #     pitch_img,
    #     "Ground Truth - Pitch View",
    #     (radar_center_x, 20),
    #     scale=0.8,
    #     thickness=2,
    #     color_txt=(0, 0, 0),
    #     alignH="c",
    #     alignV="t"
    # )
    
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
            text = str(track_id)
        
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


def visualize_frame(frame_annotations, output_path, scale=4):
    """
    可视化单帧图像在pitch view上（主函数）
    
    Args:
        frame_annotations: 该帧的所有annotations列表
        output_path: 输出路径
        scale: 缩放比例
    """
    # 第一步：提取结构化信息
    frame_info = extract_frame_info(frame_annotations, image_path=None)
    
    if not frame_info['valid']:
        return
    
    # 第二步：在pitch view上可视化
    visualized_image = visualize_structured_data(frame_info, scale=scale)
    
    if visualized_image is None:
        return
    
    # 保存图像
    cv2.imwrite(output_path, visualized_image)


def main():
    parser = argparse.ArgumentParser(description='可视化SoccerNetGSR数据集的Ground Truth (Pitch View)')
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS',
                       help='数据集根目录')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='数据集split')
    parser.add_argument('--sequence_name', type=str, default='SNGS-124',
                       help='序列名称')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/paper/bev_gt_vis',
                       help='输出目录')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='最大处理帧数，-1表示处理所有帧')
    parser.add_argument('--scale', type=int, default=16,
                       help='球场图片的缩放比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建序列特定的输出目录
    sequence_output_dir = output_dir / args.sequence_name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载Labels-GameState.json
    sequence_dir = Path(args.data_dir) / args.split / args.sequence_name
    labels_path = sequence_dir / "Labels-GameState.json"
    
    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        return
    
    print(f"Loading annotations from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    # 构建image_id到annotations的映射
    image_id_to_annotations = {}
    for anno in labels_data['annotations']:
        image_id = anno['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(anno)
    
    # 获取所有images信息
    images_info = labels_data['images']
    
    # 限制处理的帧数
    if args.max_frames > 0:
        images_info = images_info[:args.max_frames]
    
    print(f"Processing {len(images_info)} frames...")
    
    # 处理每一帧
    for image_info in tqdm(images_info, desc="Visualizing frames"):
        image_id = image_info['image_id']
        file_name = image_info['file_name']
        
        # 获取该帧的所有annotations
        frame_annotations = image_id_to_annotations.get(image_id, [])
        
        # 构建输出路径
        output_filename = f"frame_{file_name}"
        output_path = sequence_output_dir / output_filename
        
        # 可视化并保存
        visualize_frame(frame_annotations, str(output_path), scale=args.scale)
    
    print(f"Visualization complete! Results saved to: {sequence_output_dir}")
    print(f"Total frames processed: {len(images_info)}")


if __name__ == '__main__':
    main()

