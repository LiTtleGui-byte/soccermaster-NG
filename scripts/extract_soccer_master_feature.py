# ------------------------------------------------------------------------
# Feature Extraction Script for Soccer Master Model (Multi-GPU Version)
# ------------------------------------------------------------------------
import torch
import torch.multiprocessing as mp
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from pathlib import Path
import sys
import math
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 根据你的环境导入
try:
    from models.multi_task import MultiTaskingSigLIP
    from configs.util import load_super_config, yaml_to_dict
    from utils.misc import set_seed
except ImportError:
    # 用于演示，如果找不到模块不报错，实际运行时需要正确的环境
    pass
from torchvision import transforms

# ------------------------------------------------------------------------
# Worker Function (运行在独立进程中)
# ------------------------------------------------------------------------
def worker_func(gpu_id, video_list, config, checkpoint_path, output_dir, base_dir, overwrite, chunk_size, progress_queue):
    """
    每个GPU独立运行的Worker函数
    """
    try:
        # 1. 设置当前进程的设备
        device = f'cuda:{gpu_id}'
        print(f"[GPU {gpu_id}] 初始化中... 将处理 {len(video_list)} 个视频")

        # 2. 在当前进程/GPU上初始化模型
        # 注意：这里我们重新实例化FeatureExtractor，确保模型加载在正确的显卡上
        extractor = FeatureExtractor(config, checkpoint_path, device=device)

        # 3. 循环处理分配给该GPU的视频列表
        for video_path in video_list:
            try:
                extractor.extract_from_video(
                    video_path=str(video_path),
                    output_path=None,
                    output_dir=output_dir,
                    base_dir=base_dir,
                    overwrite=overwrite,
                    chunk_size=chunk_size
                )
                # 发送成功信号
                if progress_queue is not None:
                    progress_queue.put(1)
            except Exception as e:
                print(f"[GPU {gpu_id}] 处理 {os.path.basename(video_path)} 失败: {e}")
                traceback.print_exc()
                # 发送失败/完成信号，避免主进程死锁
                if progress_queue is not None:
                    progress_queue.put(0)
                    
        print(f"[GPU {gpu_id}] 任务完成")

    except Exception as e:
        print(f"[GPU {gpu_id}] 发生致命错误: {e}")
        traceback.print_exc()

# ------------------------------------------------------------------------
# Feature Extractor Class
# ------------------------------------------------------------------------
class FeatureExtractor:
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = device
        self.num_frames = config.get('NUM_FRAMES', 30)
        self.fps = config.get('FPS', 1.0)
        
        set_seed(config.get("SEED", 42))
        
        # 构建模型 (Moving model creation here to ensure it runs inside the process)
        # print(f"Loading model on {device}...") 
        self.model = MultiTaskingSigLIP(config=config, logger=None)
        self.model.load_checkpoint(checkpoint_path, ckpt_type="soccer_master", logger=None, load_heads=False)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_video_frames(self, video_path, fps=None, start=None, duration=None):
        if fps is None:
            fps = self.config.get("FPS", 1.0)

        cap = cv2.VideoCapture(video_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not fps_video or fps_video <= 1e-2:
            cap.release()
            raise ValueError(f"无法读取视频: {video_path}")
        
        drop_extra_frames = fps_video / fps
        
        # 计算起止范围 (帧索引)
        start_frame = int(fps_video * start) if start is not None else 0
        if duration is not None:
            limit = (start + duration) if start else duration
            end_frame = min(total_frames, int(fps_video * limit))
        else:
            end_frame = total_frames

        # 预先计算需要提取的帧索引
        # 原逻辑：i_frame 从 1 开始增加。第一次 read 对应 i_frame=1 (即 index=0)
        # 条件为 i_frame % drop_extra_frames < 1
        target_indices = []
        for idx in range(start_frame, end_frame):
            i_frame = idx + 1
            if (i_frame % drop_extra_frames < 1):
                target_indices.append(idx)

        frames = []
        last_idx = -1
        
        for idx in target_indices:
            # 如果是连续的一帧，直接 read 即可，否则跳转
            if idx != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(self.transform(frame_pil))
            last_idx = idx
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"无法从视频中提取帧: {video_path}")
        
        return torch.stack(frames, dim=0)
    
    @torch.no_grad()
    def extract_features(self, input_data, chunk_size=None):
        input_data = input_data.to(self.device)
        
        if input_data.dim() == 4:
            total_frames = input_data.shape[0]
            if chunk_size is None:
                chunk_size = self.num_frames
            
            if total_frames <= chunk_size:
                output = self.model.backbone(input_data.unsqueeze(0))
                return output['global_features'].squeeze(0).cpu().numpy()
            
            # Chunking logic
            all_features = []
            num_chunks = (total_frames + chunk_size - 1) // chunk_size
            
            # Simplify tqdm for worker processes to avoid clutter
            iterator = range(num_chunks)
            
            for i in iterator:
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total_frames)
                chunk = input_data[start:end]
                
                curr_size = chunk.shape[0]
                if curr_size < chunk_size:
                    padding = chunk[-1:].repeat(chunk_size - curr_size, 1, 1, 1)
                    chunk = torch.cat([chunk, padding], dim=0)
                
                output = self.model.backbone(chunk.unsqueeze(0))
                feats = output['global_features'].squeeze(0)[:curr_size]
                all_features.append(feats.cpu())
            
            return torch.cat(all_features, dim=0).numpy()
            
        elif input_data.dim() == 5:
            output = self.model.backbone(input_data)
            return output['global_features'].squeeze(0).cpu().numpy()
    
    def extract_from_video(self, video_path, output_path=None, output_dir=None, base_dir=None, overwrite=False, fps=None, start=None, duration=None, chunk_size=None):
        if output_path is None:
            vid_path_obj = Path(video_path)
            if output_dir is not None and base_dir is not None:
                try:
                    rel_path = vid_path_obj.relative_to(Path(base_dir))
                    output_path = Path(output_dir) / rel_path.parent / f"{rel_path.stem}_soccer_master_features.npy"
                except ValueError:
                    # Fallback if not relative
                    output_path = Path(output_dir) / f"{vid_path_obj.stem}_soccer_master_features.npy"
            else:
                output_path = vid_path_obj.parent / f"{vid_path_obj.stem}_soccer_master_features.npy"
        
        if os.path.exists(output_path) and not overwrite:
            return str(output_path)
        
        # print(f"Processing: {video_path}") # Reduce logging in multi-gpu
        frames = self.load_video_frames(video_path, fps=fps, start=start, duration=duration)
        features = self.extract_features(frames, chunk_size=chunk_size)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        return str(output_path)

# ------------------------------------------------------------------------
# Main Controller
# ------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='批量提取Soccer Master模型视频特征 (多GPU并行版)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入视频目录路径', default='/mnt/vision_user/sports/datasets/soccernet_v2')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录', default='/mnt/vision_user/sports/datasets/soccernet_v2_soccer_master_features')
    parser.add_argument('--super_config', type=str, default=None, help='Super config文件路径')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的特征文件')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用的GPU数量')
    parser.add_argument('--file_extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mkv', '.mov'], help='文件扩展名')
    parser.add_argument('--chunk_size', type=int, default=None, help='推理时的Chunk size')
    return parser.parse_args()

def main():
    # 设置启动方法为 spawn，这对 CUDA 多进程是必须的
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_args()
    
    # 1. 加载配置
    print("加载配置文件...")
    cfg = yaml_to_dict(args.config)
    if args.super_config:
        cfg = load_super_config(cfg, args.super_config)
    elif "SUPER_CONFIG_PATH" in cfg:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    # 2. 扫描所有文件
    print(f"扫描目录: {args.input_dir}")
    input_path = Path(args.input_dir)
    all_files = []
    for ext in args.file_extensions:
        all_files.extend(list(input_path.rglob(f"*{ext}")))
    
    # 简单的过滤（可选）
    all_files = [f for f in all_files if '720' in f.name]
    
    if not all_files:
        print("未找到视频文件。")
        return

    total_files = len(all_files)
    print(f"共找到 {total_files} 个视频文件。")
    print(f"准备使用 {args.num_gpus} 个GPU进行并行处理。")

    # 3. 将文件列表切分给不同的GPU
    # 如果文件少于GPU数，减少GPU使用数
    num_gpus = min(args.num_gpus, total_files)
    
    # 将列表均匀切分成 num_gpus 份
    # numpy array_split 处理不均匀切分很方便
    chunks = np.array_split(all_files, num_gpus)
    
    # 4. 启动多进程
    processes = []
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    for rank in range(num_gpus):
        file_chunk = chunks[rank].tolist() # 转换回 Python list
        if not file_chunk:
            continue
            
        p = mp.Process(
            target=worker_func,
            args=(
                rank,               # gpu_id
                file_chunk,         # 该GPU负责的文件列表
                cfg,                # 配置字典
                args.checkpoint,    # 权重路径
                args.output_dir,    # 输出目录
                str(input_path),    # 基础目录(用于相对路径)
                args.overwrite,     # 是否覆盖
                args.chunk_size,    # chunk size
                progress_queue      # 进度队列
            )
        )
        p.start()
        processes.append(p)
    
    # 5. 主进程监控进度
    with tqdm(total=total_files, desc="Total Progress") as pbar:
        processed_count = 0
        while processed_count < total_files:
            # 阻塞等待消息，或者每秒检查一次进程存活状态
            try:
                # 设置timeout防止死锁，如果处理很快这里会迅速更新
                _ = progress_queue.get(timeout=1.0)
                processed_count += 1
                pbar.update(1)
            except:
                # 检查子进程是否都还活着
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
    
    # 6. 等待所有进程结束
    for p in processes:
        p.join()
        
    print("\n✅ 所有任务处理完成！")

if __name__ == '__main__':
    main()