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
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from soccermaster.data.utils import lines_dict, clip_keypoints_to_image, add_x_y_to_lines
from soccermaster.data.soccernet_gsr_reid import role_mapping
from scripts.vis_sn500_extracted import get_visible_lines_coords

# 反向映射role_mapping
role_mapping_reverse = {v: k for k, v in role_mapping.items()}


def projection_from_cam_params_traditional(cam_params):
    """
    从相机参数计算投影矩阵
    
    Args:
        cam_params: 相机参数字典
    
    Returns:
        K: 内参矩阵 (3x3)
        Rt: 外参矩阵 (3x4)
        P: 投影矩阵 (3x4)
    """
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])
    
    # 内参矩阵
    K = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    
    # 外参矩阵
    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Rt = rotation @ It
    
    # 投影矩阵
    P = K @ Rt
    return K, Rt, P


def extract_frame_info(frame_data, frame_data_image, image_path, frame_height=1080, frame_width=1920):
    """
    从帧数据中提取可视化所需的结构化信息
    
    Args:
        frame_data: 帧数据字典
        frame_data_image: 帧数据图像字典
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
            - detected_keypoints: list of dict, 每个dict包含:
                - point: (x, y) tuple (像素坐标)
                - keypoint_id: int
                - confidence: float
            - detected_lines: list of dict, 每个dict包含:
                - points: list of (x, y) tuples (像素坐标)
                - line_name: str
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
    for id, row in frame_data.iterrows():
        x, y, w, h = row['bbox_ltwh']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # 提取属性
        role = row['role']
        jersey_number = row['jersey_number']
        legibility_score = row['legibility_score']
        team = row['team']
        track_id = row['track_id']
        
        # 将属性组织成字典（按显示顺序）
        attributes = []
        
        # 只显示球员的球衣号码和队伍信息
        if role == 'player':
            if not pd.isna(team):
                attributes.append({
                    'key': 'T',
                    'value': str(team),
                    'color': (0, 0, 255) if team == 'left' else (255, 0, 0)  # 左队蓝色，右队红色
                })
            if not pd.isna(jersey_number):
                jn = str(int(jersey_number))
            else:
                jn = 'null'
            attributes.append({
                'key': 'JN',
                'value': jn,
                'color': (0, 0, 0)
            })
            if not pd.isna(legibility_score):
                attributes.append({
                    'key': 'L',
                    'value': f'{legibility_score:.2f}',
                    'color': (0, 0, 0)
                })
                
                
        attributes.append({
            'key': 'R',
            'value': str(role),
            'color': (0, 0, 0)
        })
        

        
        persons.append({
            'bbox': (x, y, w, h),
            'track_id': int(track_id) if not pd.isna(track_id) else None,
            'attributes': attributes,
            'color': (0, 255, 0)  # 绿色bbox
        })
    
    # 提取球场线条信息（投影）
    field_lines = []
    
    # 提取检测到的关键点
    detected_keypoints = []
    
    # 提取检测到的线条
    detected_lines = []
    
    # 从frame_data_image中提取相机参数和检测信息
    if frame_data_image is not None and len(frame_data_image) > 0:
        try:
            # 获取相机参数
            cam_params = frame_data_image.iloc[0]["parameters"] if hasattr(frame_data_image.iloc[0], "parameters") else frame_data_image.iloc[0].get("parameters")
            
            # 检查相机参数是否有效，提取投影线条
            if isinstance(cam_params, dict) and len(cam_params.keys()) > 0:
                # 从相机参数计算投影矩阵
                K, R, P = projection_from_cam_params_traditional(cam_params)
                
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
                    
                    # 投影线条统一使用蓝色
                    field_lines.append({
                        'points': pixel_points,
                        'line_type': 'circle' if 'Circle' in line_name else 'line',
                        'color': (255, 0, 0)  # 蓝色 (BGR)
                    })
            
            # 提取检测到的关键点
            if 'keypoints' in frame_data_image.columns:
                keypoints_data = frame_data_image.iloc[0]['keypoints']
                if isinstance(keypoints_data, dict):
                    for kp_id, kp_info in keypoints_data.items():
                        if isinstance(kp_info, dict) and 'x' in kp_info and 'y' in kp_info:
                            x = int(kp_info['x'])
                            y = int(kp_info['y'])
                            confidence = kp_info.get('p', 1.0)
                            detected_keypoints.append({
                                'point': (x, y),
                                'keypoint_id': kp_id,
                                'confidence': confidence
                            })
            
            # 提取检测到的线条
            if 'lines' in frame_data_image.columns:
                lines_data = frame_data_image.iloc[0]['lines']
                if isinstance(lines_data, dict):
                    for line_name, line_points in lines_data.items():
                        if isinstance(line_points, list) and len(line_points) > 0:
                            # 将归一化坐标转换为像素坐标
                            pixel_points = []
                            for point in line_points:
                                if isinstance(point, dict) and 'x' in point and 'y' in point:
                                    px = int(point['x'] * frame_width)
                                    py = int(point['y'] * frame_height)
                                    # 确保坐标在图像范围内
                                    px = max(0, min(frame_width - 1, px))
                                    py = max(0, min(frame_height - 1, py))
                                    pixel_points.append((px, py))
                            
                            if len(pixel_points) > 0:
                                detected_lines.append({
                                    'points': pixel_points,
                                    'line_name': line_name
                                })
        except Exception as e:
            print(f"Warning: Failed to extract field lines/keypoints: {e}")
    
    return {
        'valid': True,
        'image': image,
        'persons': persons,
        'field_lines': field_lines,
        'detected_keypoints': detected_keypoints,
        'detected_lines': detected_lines,
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
                cv2.line(image, points[0], points[1], color, line_thickness)
    
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


def visualize_frame(frame_id, frame_data, frame_data_image, image_path, output_path, sequence_name):
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
    frame_info = extract_frame_info(frame_data, frame_data_image, image_path)
    
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
    parser.add_argument('--image_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS/test',
                       help='图像目录')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/vision_user/sports/code/Soccer-Backbone/paper/pipeline_gsr_test',
                       help='输出目录')
    parser.add_argument('--max_frames', type=int, default=-1,
                       help='最大处理帧数，-1表示处理所有帧')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从pkl路径中提取序列名称
    sequence_name = 'SNGS-187'
    
    # 创建序列特定的输出目录
    sequence_output_dir = output_dir / sequence_name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)
    
    pklz_path = '/mnt/vision_user/sports/code/Soccer-Backbone/paper/yolo_v8x6_person_nbjw_cam_qwen25vl_legibility_filter2_qwen25vl_role_7b_remove_outside_concat_jn_reid_testset_72b/sn-gamestate.pklz'
    
    # 加载pkl数据
    zf = zipfile.ZipFile(pklz_path)
    data = pickle.load(zf.open(f'{sequence_name[-3:]}.pkl'))
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
        
        # 构建图像路径（注意frame_id是从1开始的）
        image_filename = f"{frame_id[-6:]}.jpg"
        image_path = os.path.join(args.image_dir, sequence_name, 'img1', image_filename)
        
        # 构建输出路径
        output_filename = f"frame_{frame_id[-6:]}.jpg"
        output_path = sequence_output_dir / output_filename
        
        # 可视化并保存
        visualize_frame(frame_id, frame_data, frame_data_image, image_path, str(output_path), sequence_name)
    
    print(f"Visualization complete! Results saved to: {sequence_output_dir}")
    print(f"Total frames processed: {len(image_ids)}")


if __name__ == '__main__':
    main()

