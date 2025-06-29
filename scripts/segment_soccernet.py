import cv2
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def get_parts_from_metadata(metadata_path):
    positions1 = []
    labels1 = []
    replays1 = []
    positions2 = []
    labels2 = []
    replays2 = []
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    annotations = metadata['annotations']
    for anno in annotations:
        if anno['gameTime'][0] == '1':
            positions1.append(anno['position'])
            labels1.append(anno['label'])
            replays1.append(anno['replay'])
        elif anno['gameTime'][0] == '2':
            positions2.append(anno['position'])
            labels2.append(anno['label'])
            replays2.append(anno['replay'])
    parts1 = []
    for i in range(len(positions1)):
        part = {
            'start_position': int(positions1[i-1]) if i > 0 else 0,
            'end_position': int(positions1[i]),
            'label': labels1[i],
            'replay': replays1[i],
        }
        parts1.append(part)
    parts2 = []
    for i in range(len(positions2)):
        part = {
            'start_position': int(positions2[i-1]) if i > 0 else 0,
            'end_position': int(positions2[i]),
            'label': labels2[i],
            'replay': replays2[i],
        }
        parts2.append(part)
    return parts1, parts2

def get_main_camera_valid_parts(video_path, parts, discard_width=10, real_time_only=True, len_clip=750):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    valid_indices = []
    for part in parts:
        if part['label'] != 'Main camera center':
            continue
        if real_time_only and part['replay'] != 'real-time':
            continue
        
        start_frame = max(0, int((part['start_position'] / 1000.0) * fps) + discard_width + 1) # 要
        end_frame = min(total_frames - 1, int((part['end_position'] / 1000.0) * fps) - discard_width - 1) # 要
        # 从start_frame到end_frame开始切段，每段最长是len_clip
        current_frame = start_frame
        while current_frame <= end_frame:
            segment_end = min(current_frame + len_clip - 1, end_frame)
            valid_indices.append({
                'start_frame': current_frame,
                'end_frame': segment_end
            })
            current_frame = segment_end + 1
            
    return valid_indices

def save_video_clip(video_path, save_dir, start_frame, end_frame):
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 设置目标分辨率 (1080p)
    target_height = 1080
    target_width = 1920
    
    frame_count = 0
    for frame_idx in range(start_frame, end_frame + 1):
        # 设置视频位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧 {frame_idx}")
            continue
        
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        frame_count += 1
        filename = f"{frame_count:06d}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # cv2.imwrite(filepath, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(filepath, resized_frame)
    
    # 释放资源
    cap.release()
    print(f"完成！共保存了 {frame_count} 张图片到 {save_dir}")

def process_single_clip(args):
    """处理单个视频片段的函数，用于多线程"""
    league, season, game, half, start_frame, end_frame, clip_id, src_vid_root, save_root = args
    try:
        video_src_path = os.path.join(src_vid_root, league, season, game, f'{half}_720p.mkv')
        video_save_dir = os.path.join(save_root, f"SNGS-{clip_id+10000:05d}")
        save_video_clip(video_src_path, video_save_dir, start_frame, end_frame)
        return f"成功处理: SNGS-{clip_id+10000:05d}"
    except Exception as e:
        return f"处理失败: SNGS-{clip_id+10000:05d}, 错误: {str(e)}"

if __name__ == '__main__':
    src_vid_root = '/remote-home/haolinyang/public/sports/SoccerNet/dataset-720p'
    metadata_root = '/remote-home/haolinyang/public/sports/SoccerNet/dataset-cameras'
    save_root = '/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/sn500'
    os.makedirs(save_root, exist_ok=True)

    # 用来记录(league, season, game, half, indices) -> clip_id
    sn_2_clip = []
    clip_id = 1
    
    # leagues = os.listdir(metadata_root)
    # leagues.sort()
    # for league in leagues:
    #     league_path = os.path.join(metadata_root, league)
    #     seasons = os.listdir(league_path)
    #     seasons.sort()
    #     for season in seasons:
    #         season_path = os.path.join(league_path, season)
    #         games = os.listdir(season_path)
    #         games.sort()
    #         for game in games:
    #             game_path = os.path.join(season_path, game)
    #             video_dir = os.path.join(src_vid_root, league, season, game)
    #             assert os.path.exists(video_dir), f"Video directory not found: {video_dir}"
    #             video_1_path = os.path.join(video_dir, '1_720p.mkv')
    #             video_2_path = os.path.join(video_dir, '2_720p.mkv')
    #             assert os.path.exists(video_1_path), f"Video 1 not found: {video_1_path}"
    #             assert os.path.exists(video_2_path), f"Video 2 not found: {video_2_path}"
    #             metadata_path = os.path.join(game_path, 'Labels-cameras.json')
    #             assert os.path.exists(metadata_path), f"Metadata file not found: {metadata_path}"
    #             parts1, parts2 = get_parts_from_metadata(metadata_path)
    #             valid_indices1 = get_main_camera_valid_parts(video_1_path, parts1)
    #             valid_indices2 = get_main_camera_valid_parts(video_2_path, parts2)
                
    #             # 记录第一半场的片段
    #             for indices in valid_indices1:
    #                 sn_2_clip.append([league, season, game, 1, indices['start_frame'], indices['end_frame'], clip_id])
    #                 clip_id += 1
                
    #             # 记录第二半场的片段
    #             for indices in valid_indices2:
    #                 sn_2_clip.append([league, season, game, 2, indices['start_frame'], indices['end_frame'], clip_id])
    #                 clip_id += 1
    
    # with open('/remote-home/haolinyang/sports/Soccer-Backbone/tmp/sn_2_clip.json', 'w') as f:
    #     json.dump(sn_2_clip, f)
        
    sn_2_clip = json.load(open('/remote-home/haolinyang/sports/Soccer-Backbone/tmp/sn_2_clip.json', 'r'))
        
    # sequences_info_path = '/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/sequences_info.json'
    # sequences_info_save_path = '/remote-home/haolinyang/datasets/SN-GSR-2024/bak/sequences_info_sn500.json'
    # sequences_info = json.load(open(sequences_info_path, 'r'))
    # sequences_info['sn500'] = []
    
    # for league, season, game, half, start_frame, end_frame, clip_id in sn_2_clip:
    #     sequences_info['sn500'].append({"id": clip_id - 1, "name": f"SNGS-{clip_id+10000:05d}", "n_frames": end_frame - start_frame + 1})
    
    # with open(sequences_info_save_path, 'w') as f:
    #     json.dump(sequences_info, f)
    
    # 准备多线程参数
    args_list = []
    for league, season, game, half, start_frame, end_frame, clip_id in sn_2_clip:
        args_list.append((league, season, game, half, start_frame, end_frame, clip_id, src_vid_root, save_root))
    
    # 使用16个线程处理
    max_workers = 16
    print(f"开始使用 {max_workers} 个线程处理 {len(args_list)} 个视频片段...")
    
    # 初始化失败日志文件和线程锁
    fail_log_path = '/remote-home/haolinyang/sports/Soccer-Backbone/scripts/fail.txt'
    file_lock = threading.Lock()
    
    # 清空或创建失败日志文件
    with open(fail_log_path, 'w', encoding='utf-8') as f:
        f.write(f"视频处理失败日志 - 开始时间: {os.popen('date').read().strip()}\n")
        f.write("=" * 50 + "\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_args = {executor.submit(process_single_clip, args): args for args in args_list}
        
        # 使用tqdm显示进度
        completed = 0
        failed_count = 0
        with tqdm(total=len(args_list), desc="处理视频片段") as pbar:
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                    # print(result)  # 如果需要看详细结果可以取消注释
                except Exception as e:
                    args = future_to_args[future]
                    league, season, game, half, start_frame, end_frame, clip_id, _, _ = args
                    error_msg = f"任务失败: {args}, 错误: {e}"
                    print(error_msg)
                    
                    # 线程安全地写入失败日志
                    with file_lock:
                        with open(fail_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"CLIP_ID: SNGS-{clip_id+10000:05d}\n")
                            f.write(f"League: {league}, Season: {season}, Game: {game}, Half: {half}\n")
                            f.write(f"Frames: {start_frame}-{end_frame}\n")
                            f.write(f"错误: {str(e)}\n")
                            f.write("-" * 30 + "\n")
                    
                    failed_count += 1
                
                completed += 1
                pbar.update(1)
    
    # 写入总结信息到失败日志
    with open(fail_log_path, 'a', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"处理完成时间: {os.popen('date').read().strip()}\n")
        f.write(f"总任务数: {len(args_list)}\n")
        f.write(f"成功任务数: {len(args_list) - failed_count}\n")
        f.write(f"失败任务数: {failed_count}\n")
        if failed_count == 0:
            f.write("恭喜！所有任务都成功完成！\n")
    
    print(f"所有任务完成！共处理了 {len(args_list)} 个视频片段，成功 {len(args_list) - failed_count} 个，失败 {failed_count} 个")
    if failed_count > 0:
        print(f"失败任务详情已记录在: {fail_log_path}")