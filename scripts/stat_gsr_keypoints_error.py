import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from PIL import Image
from data.pnlcalib_utils.utils_keypoints import KeypointsDB
from tqdm import tqdm
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_annotation(args):
    """处理单个annotation的函数"""
    seq_dir, anno = args
    try:
        lines = anno['lines']
        image_id = anno['image_id'][-6:]
        image_path = os.path.join(seq_dir, 'img1', f'{image_id}.jpg')
        image = Image.open(image_path).convert("RGB")
        image = v2.functional.to_tensor(image)
        image = v2.functional.resize(image, (512, 512))
        image_db = KeypointsDB(lines, image)
        
        try:
            target, mask = image_db.get_tensor_w_mask()
            return {'success': True, 'error': False}
        except Exception as e:
            return {'success': True, 'error': True, 'exception': str(e)}
    except Exception as e:
        return {'success': False, 'exception': str(e)}

def process_split(split, metadata, root):
    """处理单个split的函数"""
    print(f"\n正在处理 {split} split...")
    
    # 收集当前split的所有任务
    split_tasks = []
    for seq_info in metadata[split]:
        seq_name = seq_info['name']
        seq_dir = os.path.join(root, split, seq_name)
        gt_path = os.path.join(seq_dir, 'Labels-GameState.json')
        gt = json.load(open(gt_path, 'r'))
        
        for anno in gt['annotations']:
            if anno['supercategory'] == 'pitch':
                split_tasks.append((seq_dir, anno))
    
    total_annotations = len(split_tasks)
    total_cnt = 0
    total_error_cnt = 0
    
    # 线程锁用于安全更新计数器
    lock = threading.Lock()
    
    # 使用ThreadPoolExecutor进行多线程处理
    max_workers = min(8, os.cpu_count() * 2)  # 限制最大线程数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_annotation, task): task for task in split_tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=total_annotations, desc=f"处理{split}关键点数据") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                
                with lock:
                    if result['success']:
                        total_cnt += 1
                        if result['error']:
                            total_error_cnt += 1
                        elif 'exception' in result:
                            print(f"其他异常: {result['exception']}")
                    else:
                        print(f"处理失败: {result['exception']}")
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        '错误数': total_error_cnt,
                        '错误率': f'{total_error_cnt/total_cnt:.2%}' if total_cnt > 0 else '0%',
                        '线程数': max_workers
                    })
    
    return total_cnt, total_error_cnt

if __name__ == "__main__":
    root = './datasets/SN-GSR-2024/SoccerNetGS/'
    meta_path = os.path.join(root, 'sequences_info.json')
    metadata = json.load(open(meta_path, 'r'))
    splits = ['train', 'valid', 'test']
    metadata['valid'] = metadata['validation']
    
    # 存储每个split的统计结果
    split_results = {}
    total_all_cnt = 0
    total_all_error_cnt = 0
    
    # 分别处理每个split
    for split in splits:
        total_cnt, total_error_cnt = process_split(split, metadata, root)
        split_results[split] = {
            'total_cnt': total_cnt,
            'total_error_cnt': total_error_cnt,
            'error_rate': total_error_cnt / total_cnt if total_cnt > 0 else 0
        }
        
        total_all_cnt += total_cnt
        total_all_error_cnt += total_error_cnt
        
        print(f"\n{split} split 统计结果:")
        print(f"  总数量: {total_cnt}")
        print(f"  错误数量: {total_error_cnt}")
        print(f"  错误率: {total_error_cnt / total_cnt:.4f} ({total_error_cnt / total_cnt:.2%})")
    
    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    for split in splits:
        result = split_results[split]
        print(f"{split:>5}: {result['total_error_cnt']:>4}/{result['total_cnt']:>4} = {result['error_rate']:.4f} ({result['error_rate']:.2%})")
    
    print(f"\n总计: {total_all_error_cnt}/{total_all_cnt} = {total_all_error_cnt/total_all_cnt:.4f} ({total_all_error_cnt/total_all_cnt:.2%})")