import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import zipfile
import pickle
import copy
from PIL import Image
from data.pnlcalib_utils.utils_keypoints import KeypointsDB
from data.pnlcalib_utils.utils_lines import LineKeypointsDB
from tqdm import tqdm
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def correct_lines_labels(data):
    """修正lines标签的函数，参考soccernet_gsr_detection.py"""
    if 'Goal left post left' in data.keys():
        data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
        del data['Goal left post left']
    return data

def process_annotation(args):
    """处理单个annotation的函数"""
    seq_dir, lines_data, image_id = args
    try:
        image_path = os.path.join(seq_dir, 'img1', f'{image_id:06d}.jpg')
        image = Image.open(image_path).convert("RGB")
        image = v2.functional.to_tensor(image)
        image = v2.functional.resize(image, (224, 224))
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image = v2.functional.normalize(image, mean=mean, std=std)
        
        # 测试KeypointsDB
        keypoints_error = False
        lines_error = False
        
        # 测试LineKeypointsDB
        try:
            lines_db = LineKeypointsDB(lines_data, image)
            lines_target = lines_db.get_tensor()
        except Exception as e:
            lines_error = True
        
        try:
            keypoints_db = KeypointsDB(correct_lines_labels(lines_data), image)
            target, mask = keypoints_db.get_tensor_w_mask()
        except Exception as e:
            keypoints_error = True
        
            
        return {
            'success': True, 
            'keypoints_error': keypoints_error,
            'lines_error': lines_error
        }
    except Exception as e:
        return {'success': False, 'exception': str(e)}

def process_pseudo_data(extra_data_path, data_dir):
    """处理pseudo data的函数"""
    print(f"\n正在处理 pseudo data...")
    
    # 收集所有任务
    tasks = []
    # sequence_info = {}
    
    with zipfile.ZipFile(extra_data_path) as zf:
        name_list = zf.namelist()
        sequence_names = [name[:5] for name in name_list if name.endswith('_image.pkl')]
        
        for name in tqdm(sequence_names, desc="加载序列数据"):
            processed_sequence_name = f'SNGS-{name}'
            sequence_dir = os.path.join(data_dir, 'SoccerNetGS', 'sn500', processed_sequence_name)
            
            # 从图片目录获取序列长度
            img_dir = os.path.join(sequence_dir, 'img1')
            if os.path.exists(img_dir):
                # num_frames = len(os.listdir(img_dir))
                # sequence_info[processed_sequence_name] = num_frames
                
                # 读取image data
                with zf.open(f'{name}_image.pkl') as f:
                    image_data = pickle.load(f)  # pandas dataframe
                
                for id, row in image_data.iterrows():
                    frame_idx = int(row['id'][-6:]) - 1
                    lines = row["lines"]
                    if not isinstance(lines, dict):
                        lines = {}
                    # if len(lines) > 0:  # 只处理有lines数据的frame
                    tasks.append((sequence_dir, lines, frame_idx + 1))  # +1因为图片是1-indexed
            else:
                print(f"图片目录不存在: {img_dir}")
                exit()
    
    total_annotations = len(tasks)
    total_cnt = 0
    keypoints_error_cnt = 0
    lines_error_cnt = 0
    
    # 线程锁用于安全更新计数器
    lock = threading.Lock()
    
    # 使用ThreadPoolExecutor进行多线程处理
    max_workers = min(8, os.cpu_count() * 2)  # 限制最大线程数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_annotation, task): task for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=total_annotations, desc=f"处理pseudo data关键点和线段数据") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                
                with lock:
                    if result['success']:
                        total_cnt += 1
                        if result['keypoints_error']:
                            keypoints_error_cnt += 1
                        if result['lines_error']:
                            lines_error_cnt += 1
                    else:
                        print(f"处理失败: {result['exception']}")
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'KeyPoints错误': keypoints_error_cnt,
                        'Lines错误': lines_error_cnt,
                        'KeyPoints错误率': f'{keypoints_error_cnt/total_cnt:.2%}' if total_cnt > 0 else '0%',
                        'Lines错误率': f'{lines_error_cnt/total_cnt:.2%}' if total_cnt > 0 else '0%',
                        '线程数': max_workers
                    })
    
    return total_cnt, keypoints_error_cnt, lines_error_cnt

if __name__ == "__main__":
    # 设置数据路径
    data_dir = './datasets/SN-GSR-2024/'
    extra_data_path = './datasets/SN-GSR-2024/pseudo_pklz/sn500_1000_step3.pklz'
    
    # 检查文件是否存在
    if not os.path.exists(extra_data_path):
        print(f"错误: 找不到pseudo data文件: {extra_data_path}")
        sys.exit(1)
    
    print("=== Pseudo Data KeypointsDB 和 LineKeypointsDB 错误率统计 ===")
    print(f"数据文件: {extra_data_path}")
    
    # 处理pseudo data
    total_cnt, keypoints_error_cnt, lines_error_cnt = process_pseudo_data(extra_data_path, data_dir)
    
    # 打印统计结果
    print(f"\n=== 统计结果 ===")
    print(f"总处理数量: {total_cnt}")
    print(f"KeypointsDB 错误数量: {keypoints_error_cnt}")
    print(f"LineKeypointsDB 错误数量: {lines_error_cnt}")
    
    if total_cnt > 0:
        keypoints_error_rate = keypoints_error_cnt / total_cnt
        lines_error_rate = lines_error_cnt / total_cnt
        
        print(f"\nKeypointsDB 错误率: {keypoints_error_rate:.4f} ({keypoints_error_rate:.2%})")
        print(f"LineKeypointsDB 错误率: {lines_error_rate:.4f} ({lines_error_rate:.2%})")
        
        # 统计两者都出错的情况
        print(f"\n详细统计:")
        print(f"  KeypointsDB:     {keypoints_error_cnt:>4}/{total_cnt:>4} = {keypoints_error_rate:.4f} ({keypoints_error_rate:.2%})")
        print(f"  LineKeypointsDB: {lines_error_cnt:>4}/{total_cnt:>4} = {lines_error_rate:.4f} ({lines_error_rate:.2%})")
    else:
        print("没有找到可处理的数据")