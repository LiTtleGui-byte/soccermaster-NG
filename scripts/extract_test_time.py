from SoccerNet.DataLoader import Frame, FrameCV
import time
import cv2
from PIL import Image
import torch
from tqdm import tqdm

# def load_video_frames(video_path, fps=None, start=None, duration=None):
#     fps =1.0

#     cap = cv2.VideoCapture(video_path)
#     fps_video = cap.get(cv2.CAP_PROP_FPS)
    
#     if not fps_video or fps_video <= 1e-2:
#         cap.release()
#         raise ValueError(f"无法读取视频: {video_path}")
    
#     drop_extra_frames = fps_video / fps
#     frames = []
#     i_frame = 0
#     ret, frame = cap.read()
    
#     while ret:
#         i_frame += 1
#         if start is not None and i_frame < fps_video * start:
#             ret, frame = cap.read()
#             continue
#         if duration is not None:
#             limit = (start + duration) if start else duration
#             if i_frame > fps_video * limit:
#                 break
        
#         if (i_frame % drop_extra_frames < 1):
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_pil = Image.fromarray(frame_rgb)
#             frames.append(frame_pil)
        
#         ret, frame = cap.read()
    
#     cap.release()
    
#     if len(frames) == 0:
#         raise ValueError(f"无法从视频中提取帧: {video_path}")
    
#     return torch.stack(frames, dim=0)

def load_video_frames(video_path, fps=None, start=None, duration=None):
    fps = 1.0

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
    
    for idx in tqdm(target_indices):
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
    
    return torch.stack(frames, dim=0)

if __name__ == "__main__":
    path = '/mnt/vision_user/sports/datasets/soccernet_v2/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv'
    start_time = time.time()
    # frames = FrameCV(path, FPS=1.0)
    frames = load_video_frames(path)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")