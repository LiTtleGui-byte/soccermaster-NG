import cv2
import os

if __name__ == '__main__':
    vid_path = '/remote-home/haolinyang/public/sports/SoccerNet/dataset-720p/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv'
    position = 59752 # milliseconds
    save_dir = '/remote-home/haolinyang/sports/Soccer-Backbone/tmp_vis'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {vid_path}")
        exit(1)
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")
    
    # 计算position对应的帧索引
    target_frame = int((position / 1000.0) * fps)
    print(f"目标时间: {position}ms, 对应帧: {target_frame}")
    
    # 计算要提取的帧范围（前后各15帧，总共30帧）
    start_frame = max(0, target_frame - 15)
    end_frame = min(total_frames - 1, target_frame + 14)
    
    print(f"提取帧范围: {start_frame} 到 {end_frame}")
    
    # 提取并保存帧
    for frame_idx in range(start_frame, end_frame + 1):
        # 设置视频位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧 {frame_idx}")
            continue
        
        # 计算该帧对应的毫秒数
        frame_milliseconds = int((frame_idx / fps) * 1000)
        
        # 生成文件名
        filename = f"{frame_idx}_{frame_milliseconds}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # 保存帧
        cv2.imwrite(filepath, frame)
        print(f"已保存: {filename}")
    
    # 释放资源
    cap.release()
    print(f"完成！共提取了 {end_frame - start_frame + 1} 帧")
    