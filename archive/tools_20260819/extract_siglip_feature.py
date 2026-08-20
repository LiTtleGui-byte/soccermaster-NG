# ------------------------------------------------------------------------
# Feature Extraction Script for SigLIP Model (Multi-GPU Version)
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

# Import transformers for SigLIP
try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    print("请安装 transformers: pip install transformers")
    sys.exit(1)

from torchvision import transforms

# ------------------------------------------------------------------------
# Worker Function (运行在独立进程中)
# ------------------------------------------------------------------------
def worker_func(gpu_id, video_list, model_path, output_dir, base_dir, overwrite, chunk_size, progress_queue):
    """
    每个GPU独立运行的Worker函数
    """
    try:
        # 1. 设置当前进程的设备
        device = f'cuda:{gpu_id}'
        print(f"[GPU {gpu_id}] 初始化中... 将处理 {len(video_list)} 个视频")

        # 2. 在当前进程/GPU上初始化模型
        extractor = FeatureExtractor(model_path, device=device)

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
    def __init__(self, model_path, device='cuda', fps=1.0):
        """
        初始化 SigLIP 特征提取器
        
        Args:
            model_path: SigLIP 模型路径
            device: 计算设备
            fps: 视频采样帧率
        """
        self.device = device
        self.fps = fps
        
        print(f"[{device}] 加载 SigLIP 模型: {model_path}")
        
        # 加载 SigLIP 模型和处理器
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16  # 使用半精度加速
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"加载模型失败，尝试本地路径: {e}")
            # 如果是相对路径，尝试从workspace根目录加载
            full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path)
            self.model = AutoModel.from_pretrained(
                full_path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            self.processor = AutoProcessor.from_pretrained(
                full_path,
                trust_remote_code=True
            )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"[{device}] SigLIP 模型加载完成")
    
    def load_video_frames(self, video_path, fps=None, start=None, duration=None):
        """
        从视频中加载帧
        
        Args:
            video_path: 视频路径
            fps: 采样帧率
            start: 开始时间(秒)
            duration: 持续时间(秒)
        
        Returns:
            PIL Image 列表
        """
        if fps is None:
            fps = self.fps

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
            frames.append(frame_pil)
            last_idx = idx
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"无法从视频中提取帧: {video_path}")
        
        return frames
    
    @torch.no_grad()
    def extract_features(self, frames, chunk_size=30):
        """
        从帧序列中提取 SigLIP 特征
        
        Args:
            frames: PIL Image 列表
            chunk_size: 每次处理的帧数
        
        Returns:
            numpy array of features [num_frames, feature_dim]
        """
        total_frames = len(frames)
        
        if total_frames <= chunk_size:
            # 一次性处理
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取视觉编码器的输出
            outputs = self.model.vision_model(**inputs)
            
            # 获取 global features (pooled output)
            # SigLIP 的 pooler_output 或 last_hidden_state[:,0,:] 都是 global features
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                global_features = outputs.pooler_output
            else:
                # 如果没有 pooler_output，使用 [CLS] token (第一个位置)
                global_features = outputs.last_hidden_state[:, 0, :]
            
            return global_features.cpu().float().numpy()
        
        # Chunking logic - 分块处理
        all_features = []
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_frames)
            chunk_frames = frames[start:end]
            
            # 处理当前块
            inputs = self.processor(images=chunk_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.vision_model(**inputs)
            
            # 获取 global features
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                global_features = outputs.pooler_output
            else:
                global_features = outputs.last_hidden_state[:, 0, :]
            
            all_features.append(global_features.cpu().float())
        
        # 合并所有特征
        return torch.cat(all_features, dim=0).numpy()
    
    def extract_from_video(self, video_path, output_path=None, output_dir=None, 
                          base_dir=None, overwrite=False, fps=None, 
                          start=None, duration=None, chunk_size=30):
        """
        从视频中提取特征并保存
        
        Args:
            video_path: 输入视频路径
            output_path: 输出文件路径(可选)
            output_dir: 输出目录(可选)
            base_dir: 基础目录，用于计算相对路径(可选)
            overwrite: 是否覆盖已存在的文件
            fps: 采样帧率
            start: 开始时间(秒)
            duration: 持续时间(秒)
            chunk_size: 处理块大小
        
        Returns:
            输出文件路径
        """
        # 确定输出路径
        if output_path is None:
            vid_path_obj = Path(video_path)
            if output_dir is not None and base_dir is not None:
                try:
                    rel_path = vid_path_obj.relative_to(Path(base_dir))
                    output_path = Path(output_dir) / rel_path.parent / f"{rel_path.stem}_siglip_features.npy"
                except ValueError:
                    # Fallback if not relative
                    output_path = Path(output_dir) / f"{vid_path_obj.stem}_siglip_features.npy"
            else:
                output_path = vid_path_obj.parent / f"{vid_path_obj.stem}_siglip_features.npy"
        
        # 检查是否已存在
        if os.path.exists(output_path) and not overwrite:
            return str(output_path)
        
        # 加载视频帧
        frames = self.load_video_frames(video_path, fps=fps, start=start, duration=duration)
        
        # 提取特征
        features = self.extract_features(frames, chunk_size=chunk_size)
        
        # 保存特征
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        
        return str(output_path)

# ------------------------------------------------------------------------
# Main Controller
# ------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='批量提取SigLIP模型视频特征 (多GPU并行版)')
    parser.add_argument('--model_path', type=str, 
                       default='pretrained_models/google/siglip2-large-patch16-512',
                       help='SigLIP模型路径')
    parser.add_argument('--input_dir', type=str, help='输入视频目录路径', default='/mnt/vision_user/sports/datasets/soccernet_v2')
    parser.add_argument('--output_dir', type=str, help='输出目录', default='/mnt/vision_user/sports/datasets/soccernet_v2_siglip_features')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的特征文件')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用的GPU数量')
    parser.add_argument('--file_extensions', type=str, nargs='+', 
                       default=['.mp4', '.avi', '.mkv', '.mov'], 
                       help='文件扩展名')
    parser.add_argument('--chunk_size', type=int, default=30, 
                       help='推理时的Chunk size')
    parser.add_argument('--fps', type=float, default=1.0, 
                       help='视频采样帧率')
    parser.add_argument('--filter_keyword', type=str, default=None,
                       help='文件名过滤关键词(可选)')
    return parser.parse_args()

def main():
    # 设置启动方法为 spawn，这对 CUDA 多进程是必须的
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_args()
    
    # 1. 扫描所有文件
    print(f"扫描目录: {args.input_dir}")
    input_path = Path(args.input_dir)
    all_files = []
    for ext in args.file_extensions:
        all_files.extend(list(input_path.rglob(f"*{ext}")))
    
    # 可选的文件名过滤
    if args.filter_keyword:
        all_files = [f for f in all_files if args.filter_keyword in f.name]
        print(f"应用过滤关键词: {args.filter_keyword}")
        
    all_files = [f for f in all_files if '720' in f.name]
    
    if not all_files:
        print("未找到视频文件。")
        return

    total_files = len(all_files)
    print(f"共找到 {total_files} 个视频文件。")
    print(f"准备使用 {args.num_gpus} 个GPU进行并行处理。")
    print(f"SigLIP 模型路径: {args.model_path}")

    # 2. 将文件列表切分给不同的GPU
    num_gpus = min(args.num_gpus, total_files)
    
    # 将列表均匀切分成 num_gpus 份
    chunks = np.array_split(all_files, num_gpus)
    
    # 3. 启动多进程
    processes = []
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    for rank in range(num_gpus):
        file_chunk = chunks[rank].tolist()
        if not file_chunk:
            continue
            
        p = mp.Process(
            target=worker_func,
            args=(
                rank,               # gpu_id
                file_chunk,         # 该GPU负责的文件列表
                args.model_path,    # 模型路径
                args.output_dir,    # 输出目录
                str(input_path),    # 基础目录(用于相对路径)
                args.overwrite,     # 是否覆盖
                args.chunk_size,    # chunk size
                progress_queue      # 进度队列
            )
        )
        p.start()
        processes.append(p)
    
    # 4. 主进程监控进度
    with tqdm(total=total_files, desc="Total Progress") as pbar:
        processed_count = 0
        while processed_count < total_files:
            try:
                _ = progress_queue.get(timeout=1.0)
                processed_count += 1
                pbar.update(1)
            except:
                # 检查子进程是否都还活着
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
    
    # 5. 等待所有进程结束
    for p in processes:
        p.join()
        
    print("\n✅ 所有任务处理完成！")

if __name__ == '__main__':
    main()
