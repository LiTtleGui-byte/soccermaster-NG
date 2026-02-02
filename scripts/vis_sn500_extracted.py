#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化SN500数据集的pkl文件
包括：人物bbox、jersey_number、legibility_score、role以及投影的球场线条
"""

import os
import sys
import pickle
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.utils import lines_dict, clip_keypoints_to_image, add_x_y_to_lines
from data.soccernet_gsr_reid import role_mapping

# 反向映射role_mapping
role_mapping_reverse = {v: k for k, v in role_mapping.items()}

def get_visible_lines_coords(K, Rt, frame_height, frame_width):
    """
    获取可见线段和圆的归一化坐标(0-1)，并裁剪到图像边界内
    
    Args:
        K: 内参矩阵
        Rt: 外参矩阵
        frame_height: 图像高度
        frame_width: 图像宽度
    
    Returns:
        dict: 包含可见线段和圆的坐标字典
              key为线段/圆的名称，value为端点列表(归一化到0-1)
    """
    def get_intersection(p1, p2):
        # 计算线段与z=0.1平面的交点
        if p1[2] == p2[2]:  # 平行于z=0.1平面
            return None
        t = (0.1 - p1[2]) / (p2[2] - p1[2])
        if 0 <= t <= 1:  # 交点在线段上
            return p1 + t * (p2 - p1)
        return None

    visible_lines = {}
    
    # 处理lines_dict中的线段
    for line_name, line in lines_dict.items():
        w1 = line[0]
        w2 = line[1]
        i1 = Rt @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = Rt @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        
        # 如果两点都在相机后方，则跳过
        if i1[2] <= 0.1 and i2[2] <= 0.1:
            continue
            
        # 如果有一个点在相机后方，计算与z=0.1平面的交点
        if i1[2] <= 0.1 or i2[2] <= 0.1:
            i1_3d = i1[:3]
            i2_3d = i2[:3]
            intersection = get_intersection(i1_3d, i2_3d)
            if intersection is not None:
                if i1[2] <= 0.1:
                    i1[:3] = intersection
                else:
                    i2[:3] = intersection
        
        i1 = K @ i1
        i2 = K @ i2
        i1 /= i1[-1]
        i2 /= i2[-1]
        
        # 归一化坐标到0-1
        p1_norm = [i1[0] / frame_width, i1[1] / frame_height]
        p2_norm = [i2[0] / frame_width, i2[1] / frame_height]
        visible_lines[line_name] = [p1_norm, p2_norm]

    # 处理圆形
    r = 9.15
    
    # Circle left (pts1) - 采样20个点
    pts1 = []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = K @ (Rt @ pos)
        ipos /= ipos[-1]
        pts1.append([ipos[0] / frame_width, ipos[1] / frame_height])
    visible_lines["Circle left"] = pts1

    # Circle right (pts2) - 采样20个点
    pts2 = []
    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = K @ (Rt @ pos)
        ipos /= ipos[-1]
        pts2.append([ipos[0] / frame_width, ipos[1] / frame_height])
    visible_lines["Circle right"] = pts2

    # Circle central (pts3) - 采样20个点
    pts3 = []
    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = K @ (Rt @ pos)
        ipos /= ipos[-1]
        pts3.append([ipos[0] / frame_width, ipos[1] / frame_height])
    visible_lines["Circle central"] = pts3

    # 裁剪到图像边界内
    clipped_lines = clip_keypoints_to_image(visible_lines, frame_width, frame_height)
    clipped_lines = add_x_y_to_lines(clipped_lines)
    
    return clipped_lines

def load_pkl_data(pkl_path):
    """加载pkl文件"""
    print(f"Loading pkl file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} frames")
    return data


def extract_frame_info(frame_data, image_path, frame_height=1080, frame_width=1920):
    """
    从帧数据中提取可视化所需的结构化信息
    
    Args:
        frame_data: 帧数据字典
        image_path: 图像路径
        frame_height: 图像高度
        frame_width: 图像宽度
    
    Returns:
        dict: 包含以下字段的字典
            - image: numpy array (BGR格式)
            - persons: list of dict, 每个dict包含:
                - bbox: (x, y, w, h) tuple
                - text_lines: list of str
                - color: (B, G, R) tuple
            - field_lines: list of dict, 每个dict包含:
                - points: list of (x, y) tuples (像素坐标)
                - line_type: 'line' or 'circle'
                - color: (B, G, R) tuple
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
    
    # 提取人物信息
    persons = []
    if 'people' in frame_data:
        for person in frame_data['people']:
            # 获取bbox (ltwh格式)
            bbox = person['bbox_ltwh']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 准备文本信息
            # role = role_mapping_reverse.get(role_mapping.get(person['role'], 0), 'unknown')
            role = person['role']
            jersey_number = person['jersey_number'] if person['jersey_number'] is not None else 'null'
            if jersey_number != 'null':
                jersey_number = int(jersey_number)
            legibility_score = person['legibility_score']
            
            # 组合文本
            text_lines = [
                f"ID: {person['id']}",
                f"Role: {role}",
                f"JN: {jersey_number}",
                f"Leg: {legibility_score:.2f}"
            ]
            
            persons.append({
                'bbox': (x, y, w, h),
                'text_lines': text_lines,
                'color': (0, 255, 0)  # 绿色bbox
            })
    
    # 提取球场线条信息
    field_lines = []
    if frame_data.get('valid_cam_params', False):
        K = frame_data['K']
        R = frame_data['R']
        try:
            # 获取可见的球场线条坐标（归一化）
            visible_lines = get_visible_lines_coords(K, R, frame_height, frame_width)
            
            for line_name, points in visible_lines.items():
                if len(points) == 0:
                    continue
                
                # 将归一化坐标转换为像素坐标
                pixel_points = []
                for point in points:
                    px = int(point['x'] * frame_width)
                    py = int(point['y'] * frame_height)
                    # 确保坐标在图像范围内
                    px = max(0, min(frame_width - 1, px))
                    py = max(0, min(frame_height - 1, py))
                    pixel_points.append((px, py))
                
                # 根据线条类型设置颜色
                if 'Circle' in line_name:
                    field_lines.append({
                        'points': pixel_points,
                        'line_type': 'circle',
                        'color': (255, 0, 0)  # 蓝色
                    })
                else:
                    field_lines.append({
                        'points': pixel_points,
                        'line_type': 'line',
                        'color': (0, 0, 255)  # 红色
                    })
        except Exception as e:
            print(f"Warning: Failed to extract field lines: {e}")
    
    return {
        'valid': True,
        'image': image,
        'persons': persons,
        'field_lines': field_lines,
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
    
    # 绘制球场线条
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
                cv2.line(image, points[0], points[1], color, line_thickness)
    
    # 绘制人物bbox和文本
    bbox_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for person in frame_info['persons']:
        x, y, w, h = person['bbox']
        color = person['color']
        text_lines = person['text_lines']
        
        # 绘制bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, bbox_thickness)
        
        # 计算文本位置（bbox右侧）
        text_x = x + w + 5
        text_y = y + 20
        
        # 先计算所有文本的尺寸和位置
        text_info = []
        max_text_w = 0
        for text_line in text_lines:
            (text_w, text_h), baseline = cv2.getTextSize(text_line, font, font_scale, font_thickness)
            max_text_w = max(max_text_w, text_w)
            text_info.append((text_w, text_h, baseline))
        
        # 确保文本不超出图像边界
        if text_x + max_text_w > frame_width:
            text_x = x - max_text_w - 10
        
        # 创建半透明覆盖层用于绘制文本背景
        overlay = image.copy()
        
        # 在覆盖层上绘制所有文本的灰白色背景
        for i, (text_line, (text_w, text_h, baseline)) in enumerate(zip(text_lines, text_info)):
            current_y = text_y + i * 20
            
            # 确保不超出图像高度
            if current_y + text_h > frame_height:
                current_y = frame_height - text_h - 5
            
            # 绘制灰白色背景
            cv2.rectangle(overlay, 
                         (text_x - 2, current_y - text_h - 2), 
                         (text_x + text_w + 2, current_y + baseline + 2), 
                         (220, 220, 220),  # 灰白色
                         -1)
        
        # 混合覆盖层，实现半透明效果
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 绘制文本
        for i, text_line in enumerate(text_lines):
            current_y = text_y + i * 20
            
            # 确保不超出图像高度
            if current_y + text_info[i][1] > frame_height:
                current_y = frame_height - text_info[i][1] - 5
            
            cv2.putText(image, text_line, (text_x, current_y), 
                       font, font_scale, (0, 0, 0), font_thickness)
    
    return image


def visualize_frame(frame_id, frame_data, image_path, output_path, sequence_name):
    """
    可视化单帧图像（主函数）
    
    Args:
        frame_id: 帧ID
        frame_data: 帧数据
        image_path: 图像路径
        output_path: 输出路径
        sequence_name: 序列名称
    """
    # 第一步：提取结构化信息
    frame_info = extract_frame_info(frame_data, image_path)
    
    if not frame_info['valid']:
        return
    
    # 第二步：可视化结构化信息
    visualized_image = visualize_structured_data(frame_info)
    
    if visualized_image is None:
        return
    
    # 保存图像
    cv2.imwrite(output_path, visualized_image)


def main():
    parser = argparse.ArgumentParser(description='可视化SN500数据集')
    parser.add_argument('--pkl_path', type=str, 
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS/extracted_info/SNGS-10001.pkl',
                       help='PKL文件路径')
    parser.add_argument('--image_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10001/img1',
                       help='图像目录')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/paper',
                       help='输出目录')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='最大处理帧数，-1表示处理所有帧')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从pkl路径中提取序列名称
    sequence_name = Path(args.pkl_path).stem
    
    # 创建序列特定的输出目录
    sequence_output_dir = output_dir / sequence_name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载pkl数据
    data = load_pkl_data(args.pkl_path)
    
    # 获取所有帧ID并排序
    frame_ids = sorted(data.keys())
    
    # 限制处理的帧数
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
    
    print(f"Processing {len(frame_ids)} frames...")
    
    # 处理每一帧
    for frame_id in tqdm(frame_ids, desc="Visualizing frames"):
        frame_data = data[frame_id]
        
        # 构建图像路径（注意frame_id是从1开始的）
        image_filename = f"{frame_id:06d}.jpg"
        image_path = os.path.join(args.image_dir, image_filename)
        
        # 构建输出路径
        output_filename = f"frame_{frame_id:06d}.jpg"
        output_path = sequence_output_dir / output_filename
        
        # 可视化并保存
        visualize_frame(frame_id, frame_data, image_path, str(output_path), sequence_name)
    
    print(f"Visualization complete! Results saved to: {sequence_output_dir}")
    print(f"Total frames processed: {len(frame_ids)}")


if __name__ == '__main__':
    main()

