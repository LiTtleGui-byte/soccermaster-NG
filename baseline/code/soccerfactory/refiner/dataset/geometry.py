import numpy as np

def compute_iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """批量计算IoU矩阵"""
    # 转换为xyxy格式
    b1_x1 = bboxes1[:, 0]
    b1_y1 = bboxes1[:, 1]
    b1_x2 = bboxes1[:, 0] + bboxes1[:, 2]
    b1_y2 = bboxes1[:, 1] + bboxes1[:, 3]
    
    b2_x1 = bboxes2[:, 0]
    b2_y1 = bboxes2[:, 1]
    b2_x2 = bboxes2[:, 0] + bboxes2[:, 2]
    b2_y2 = bboxes2[:, 1] + bboxes2[:, 3]
    
    # 计算交集区域
    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)
    
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    
    # 计算各自面积
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # 计算IoU
    union = area1[:, None] + area2 - inter_area
    iou_matrix = inter_area / np.clip(union, 1e-6, None)
    
    return iou_matrix 