#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化SoccerNetGSR数据集的Ground Truth
包括：人物bbox、jersey_number、role、team以及GT的球场线条
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

from data.soccernet_gsr_reid import role_mapping

# 反向映射role_mapping
role_mapping_reverse = {v: k for k, v in role_mapping.items()}


def extract_frame_info(frame_annotations, image_path, frame_height=1080, frame_width=1920):
    """
    从Ground Truth annotations中提取可视化所需的结构化信息
    
    Args:
        frame_annotations: 该帧的所有annotations列表（来自Labels-GameState.json）
        image_path: 图像路径
        frame_height: 图像高度
        frame_width: 图像宽度
    
    Returns:
        dict: 包含以下字段的字典
            - image: numpy array (BGR格式)
            - persons: list of dict, 每个dict包含:
                - bbox: (x, y, w, h) tuple
                - track_id: int
                - attributes: list of dict
                - color: (B, G, R) tuple
            - field_lines: list of dict, 每个dict包含:
                - points: list of (x, y) tuples (像素坐标)
                - line_type: 'line' or 'circle'
                - color: (B, G, R) tuple
            - detected_keypoints: list (空，GT中没有)
            - detected_lines: list (空，GT中没有)
            - valid: bool，是否成功读取
    """
    # 读取图像
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return {'valid': False}
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to read image: {image_path}")
        return {'valid': False}
    
    frame_height, frame_width = image.shape[:2]
    
    # 提取人物信息和球场线条
    persons = []
    field_lines = []
    
    for anno in frame_annotations:
        if anno['supercategory'] == 'object':
            # 提取人物bbox和属性
            bbox = anno['bbox_image']
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
            
            track_id = anno['track_id']
            role = anno['attributes']['role']
            if role == 'ball':
                continue
            jersey = anno['attributes'].get('jersey')
            team = anno['attributes'].get('team')
            
            # 组织attributes
            attributes = []
            
            # 只显示球员的球衣号码和队伍信息
            if role == 'player':
                if team is not None:
                    attributes.append({
                        'key': 'T',
                        'value': str(team),
                        'color': (0, 0, 255) if team == 'left' else (255, 0, 0)  # 左队蓝色，右队红色
                    })
                if jersey is not None:
                    jn = str(jersey)
                else:
                    jn = 'null'
                attributes.append({
                    'key': 'JN',
                    'value': jn,
                    'color': (0, 0, 0)
                })
            
            attributes.append({
                'key': 'R',
                'value': str(role),
                'color': (0, 0, 0)
            })
            
            persons.append({
                'bbox': (x, y, w, h),
                'track_id': track_id,
                'attributes': attributes,
                'color': (0, 255, 0)  # 绿色bbox
            })
            
        elif anno['supercategory'] == 'pitch':
            # 提取球场线条（GT lines）
            lines = anno.get('lines', {})
            
            for line_name, line_points in lines.items():
                if len(line_points) == 0:
                    continue
                
                # 将归一化坐标转换为像素坐标
                pixel_points = []
                for point in line_points:
                    px = int(point['x'] * frame_width)
                    py = int(point['y'] * frame_height)
                    # 确保坐标在图像范围内
                    px = max(0, min(frame_width - 1, px))
                    py = max(0, min(frame_height - 1, py))
                    pixel_points.append((px, py))
                
                # GT线条使用红色
                field_lines.append({
                    'points': pixel_points,
                    'line_type': 'circle' if 'Circle' in line_name else 'line',
                    'color': (0, 0, 255)  # 红色 (BGR)
                })
    
    return {
        'valid': True,
        'image': image,
        'persons': persons,
        'field_lines': field_lines,
        'detected_keypoints': [],  # GT中没有检测到的keypoints
        'detected_lines': [],  # GT中没有检测到的lines
        'frame_height': frame_height,
        'frame_width': frame_width
    }


def visualize_structured_data(frame_info):
    """
    根据结构化的信息进行可视化
    
    Args:
        frame_info: 包含可视化信息的字典，由 extract_frame_info 函数生成
            - image: numpy array
            - persons: list of dict
            - field_lines: list of dict
            - frame_height: int
            - frame_width: int
    
    Returns:
        image: 可视化后的图像
    """
    if not frame_info['valid']:
        return None
    
    image = frame_info['image'].copy()
    frame_height = frame_info['frame_height']
    frame_width = frame_info['frame_width']
    
    # 绘制投影的球场线条（蓝色）
    line_thickness = 2
    for field_line in frame_info['field_lines']:
        points = field_line['points']
        color = field_line['color']
        line_type = field_line['line_type']
        
        if line_type == 'circle':
            # 对于圆形，绘制多边形
            if len(points) >= 2:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], False, color, line_thickness)
        else:
            # 对于直线，绘制线段
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(image, points[i], points[i + 1], color, line_thickness)
    
    # 绘制检测到的线条（红色）
    detected_line_thickness = 2
    detected_line_color = (0, 0, 255)  # 红色 (BGR)
    for detected_line in frame_info.get('detected_lines', []):
        points = detected_line['points']
        if len(points) >= 2:
            # 绘制连续的线段
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i + 1], detected_line_color, detected_line_thickness)
    
    # 绘制检测到的关键点（黄色）
    keypoint_color = (0, 255, 255)  # 黄色 (BGR)
    keypoint_radius = 5
    for keypoint in frame_info.get('detected_keypoints', []):
        point = keypoint['point']
        cv2.circle(image, point, keypoint_radius, keypoint_color, -1)  # 填充圆
        # 可选：在关键点旁边显示ID
        # cv2.putText(image, str(keypoint['keypoint_id']), 
        #            (point[0] + 5, point[1] - 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, keypoint_color, 1)
    
    # 绘制人物bbox和文本
    bbox_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # 减小字体大小
    font_thickness = 1
    
    for person in frame_info['persons']:
        x, y, w, h = person['bbox']
        color = person['color']
        track_id = person.get('track_id')
        attributes = person.get('attributes', [])
        
        # 绘制bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, bbox_thickness)
        
        # 1. 绘制ID在bbox左上角外侧
        if track_id is not None:
            id_text = f"ID: {track_id}"
            (id_text_w, id_text_h), id_baseline = cv2.getTextSize(id_text, font, font_scale, font_thickness)
            
            # ID位置：bbox左上角上方
            id_x = x
            id_y = y - 5
            
            # 确保不超出图像上边界
            if id_y - id_text_h < 0:
                id_y = y + id_text_h + 5  # 如果上方空间不够，放在bbox内部上方
            
            # 绘制半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay,
                         (id_x - 2, id_y - id_text_h - 2),
                         (id_x + id_text_w + 2, id_y + id_baseline + 2),
                         (255, 255, 255),  # 白色背景
                         -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # 绘制ID文本
            cv2.putText(image, id_text, (id_x, id_y),
                       font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        # 2. 绘制其他attributes在bbox底部，自下而上
        if len(attributes) > 0:
            
            # 计算所有attribute的尺寸
            attr_info = []
            for attr in attributes:
                attr_text = f"{attr['key']}: {attr['value']}"
                (attr_w, attr_h), attr_baseline = cv2.getTextSize(attr_text, font, font_scale, font_thickness)
                attr_info.append({
                    'text': attr_text,
                    'width': attr_w,
                    'height': attr_h,
                    'baseline': attr_baseline,
                    'color': attr['color']
                })
            
            # 从底部开始绘制，自下而上
            current_y = y + h  # 从bbox底部开始
            text_spacing = 20  # 文本间距
            
            for i, info in enumerate(attr_info):
                # 创建半透明覆盖层
                overlay = image.copy()
                # 绘制半透明背景
                bg_x1 = x - 2
                bg_y1 = current_y - info['height'] - 2
                bg_x2 = x + info['width'] + 2
                bg_y2 = current_y + info['baseline'] + 2
                
                # 确保不超出图像下边界
                if bg_y2 > frame_height:
                    current_y = frame_height - info['baseline'] - 2
                    bg_y1 = current_y - info['height'] - 2
                    bg_y2 = current_y + info['baseline'] + 2
                
                cv2.rectangle(overlay,
                             (bg_x1, bg_y1),
                             (bg_x2, bg_y2),
                             (255, 255, 255),  # 白色背景
                             -1)
                
                # 绘制文本（在混合后的图像上）
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                
                cv2.putText(image, info['text'], (x, current_y),
                           font, font_scale, info['color'], font_thickness, cv2.LINE_AA)
                
                # 向上移动到下一个attribute的位置
                current_y -= text_spacing
    
    return image


def visualize_frame(frame_annotations, image_path, output_path):
    """
    可视化单帧图像（主函数）
    
    Args:
        frame_annotations: 该帧的所有annotations列表
        image_path: 图像路径
        output_path: 输出路径
    """
    # 第一步：提取结构化信息
    frame_info = extract_frame_info(frame_annotations, image_path)
    
    if not frame_info['valid']:
        return
    
    # 第二步：可视化结构化信息
    visualized_image = visualize_structured_data(frame_info)
    
    if visualized_image is None:
        return
    
    # 保存图像
    cv2.imwrite(output_path, visualized_image)


def main():
    parser = argparse.ArgumentParser(description='可视化SoccerNetGSR数据集的Ground Truth')
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS',
                       help='数据集根目录')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='数据集split')
    parser.add_argument('--sequence_name', type=str, default='SNGS-187',
                       help='序列名称')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/paper/gt_vis',
                       help='输出目录')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='最大处理帧数，-1表示处理所有帧')
    
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
        
        # 构建图像路径
        image_path = sequence_dir / 'img1' / file_name
        
        # 构建输出路径
        output_filename = f"frame_{file_name}"
        output_path = sequence_output_dir / output_filename
        
        # 可视化并保存
        visualize_frame(frame_annotations, str(image_path), str(output_path))
    
    print(f"Visualization complete! Results saved to: {sequence_output_dir}")
    print(f"Total frames processed: {len(images_info)}")


if __name__ == '__main__':
    main()

