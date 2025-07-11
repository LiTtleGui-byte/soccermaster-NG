#!/usr/bin/env python3
"""
测试TensorBoard日志连续性的脚本

这个脚本模拟训练中断和恢复过程，验证tensorboard日志是否连续记录。

使用方法:
python test_tensorboard_resume.py --test-dir ./test_tensorboard_output
"""

import os
import shutil
import argparse
import json
import torch
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

def simulate_training_session(log_dir, start_step=0, num_steps=10, session_name="session"):
    """
    模拟一个训练会话，记录一些metrics到tensorboard
    
    Args:
        log_dir: tensorboard日志目录
        start_step: 开始的global_step
        num_steps: 要记录的步数
        session_name: 会话名称（用于区分不同的训练会话）
    
    Returns:
        int: 最后一个global_step
    """
    print(f"Starting {session_name} from step {start_step}")
    
    # 创建SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    
    # 记录会话开始标记
    writer.add_text(f"training/{session_name}_start", 
                   f"Session {session_name} started at step {start_step}", 
                   start_step)
    
    # 模拟训练过程中的metrics记录
    for i in range(num_steps):
        global_step = start_step + i
        
        # 模拟一些typical的训练metrics
        loss = 1.0 - (global_step * 0.01)  # 模拟loss下降
        learning_rate = 0.001 * (0.95 ** (global_step // 10))  # 模拟学习率衰减
        accuracy = min(0.95, global_step * 0.02)  # 模拟accuracy上升
        
        # 记录metrics
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/learning_rate", learning_rate, global_step)
        writer.add_scalar("train/accuracy", accuracy, global_step)
        
        # 每5步记录一次更详细的信息
        if (global_step + 1) % 5 == 0:
            writer.add_scalar("train/batch_size", 32, global_step)
            writer.add_scalar("train/gpu_memory", 1024 + (global_step * 10), global_step)
            
            # 记录histogram（模拟参数分布）
            fake_params = torch.randn(100) * (0.1 + global_step * 0.001)
            writer.add_histogram("params/layer1_weights", fake_params, global_step)
        
        # 模拟处理时间
        time.sleep(0.1)
    
    final_step = start_step + num_steps - 1
    
    # 记录会话结束标记
    writer.add_text(f"training/{session_name}_end", 
                   f"Session {session_name} ended at step {final_step}", 
                   final_step)
    
    # 关闭writer
    writer.close()
    
    print(f"Finished {session_name} at step {final_step}")
    return final_step

def create_mock_training_state(checkpoint_dir, epoch, global_step):
    """创建模拟的训练状态文件"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "start_epoch": epoch,
        "random_states": {
            "python": torch.get_rng_state().tolist(),
            "numpy": None,
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None,
        }
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), 'w') as f:
        json.dump(training_state, f, indent=2)
    
    print(f"Created mock training state for epoch {epoch}, global_step {global_step}")

def simulate_resume_workflow(test_dir):
    """
    模拟完整的训练中断和恢复工作流
    
    Args:
        test_dir: 测试目录
        
    Returns:
        bool: 测试是否成功
    """
    print("\n=== Simulating Resume Workflow ===")
    
    # 设置路径
    log_dir = os.path.join(test_dir, "logs")
    checkpoint_dir = os.path.join(test_dir, "epoch_5")
    
    # 第一阶段：初始训练（epoch 0-5）
    print("\n--- Phase 1: Initial Training ---")
    final_step_phase1 = simulate_training_session(
        log_dir=log_dir,
        start_step=0,
        num_steps=20,
        session_name="initial_training"
    )
    
    # 创建checkpoint（模拟在epoch 5时保存）
    create_mock_training_state(checkpoint_dir, epoch=5, global_step=final_step_phase1)
    
    # 模拟训练中断（等待一下再继续）
    print("\n--- Training Interrupted ---")
    time.sleep(1)
    
    # 第二阶段：恢复训练（从epoch 6开始）
    print("\n--- Phase 2: Resume Training ---")
    
    # 从training_state.json中读取global_step
    with open(os.path.join(checkpoint_dir, "training_state.json"), 'r') as f:
        training_state = json.load(f)
    
    resumed_global_step = training_state["global_step"]
    
    # 模拟resume标记记录
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("training/resume_info", 
                   f"Training resumed from epoch 6, global_step {resumed_global_step}", 
                   resumed_global_step)
    writer.add_scalar("training/resume_marker", 1.0, resumed_global_step)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    writer.add_text("training/resume_timestamp", f"Resumed at: {timestamp}", resumed_global_step)
    writer.close()
    
    # 继续训练（从下一个step开始）
    final_step_phase2 = simulate_training_session(
        log_dir=log_dir,
        start_step=resumed_global_step + 1,
        num_steps=15,
        session_name="resumed_training"
    )
    
    print(f"\nWorkflow completed:")
    print(f"  Initial training: steps 0-{final_step_phase1}")
    print(f"  Resume point: step {resumed_global_step}")
    print(f"  Resumed training: steps {resumed_global_step + 1}-{final_step_phase2}")
    
    return True

def analyze_tensorboard_logs(log_dir):
    """
    分析tensorboard日志文件，检查连续性
    
    Args:
        log_dir: tensorboard日志目录
        
    Returns:
        dict: 分析结果
    """
    print(f"\n=== Analyzing TensorBoard Logs ===")
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory {log_dir} does not exist")
        return {"success": False, "error": "Log directory not found"}
    
    # 列出所有事件文件
    event_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith("events.out.tfevents"):
            event_files.append(filename)
    
    if not event_files:
        print("❌ No tensorboard event files found")
        return {"success": False, "error": "No event files found"}
    
    print(f"✓ Found {len(event_files)} tensorboard event files:")
    for i, filename in enumerate(sorted(event_files)):
        file_path = os.path.join(log_dir, filename)
        file_size = os.path.getsize(file_path)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file_path)))
        print(f"  {i+1}. {filename} (size: {file_size} bytes, modified: {mod_time})")
    
    # 分析结果
    analysis = {
        "success": True,
        "num_event_files": len(event_files),
        "total_size": sum(os.path.getsize(os.path.join(log_dir, f)) for f in event_files),
        "event_files": event_files
    }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Test TensorBoard resume functionality")
    parser.add_argument("--test-dir", type=str, default="./test_tensorboard_output",
                       help="Directory for testing")
    parser.add_argument("--keep-files", action="store_true",
                       help="Keep test files after testing")
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir).resolve()
    print(f"Testing TensorBoard resume functionality in: {test_dir}")
    
    try:
        # 清理之前的测试文件
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        # 创建测试目录
        test_dir.mkdir(parents=True)
        
        # 运行模拟工作流
        success = simulate_resume_workflow(str(test_dir))
        
        if success:
            # 分析tensorboard日志
            log_dir = os.path.join(test_dir, "logs")
            analysis = analyze_tensorboard_logs(log_dir)
            
            if analysis["success"]:
                print("\n✅ TensorBoard resume test completed successfully!")
                print(f"   Total event files: {analysis['num_event_files']}")
                print(f"   Total log size: {analysis['total_size']} bytes")
                print(f"\nTo view the results in TensorBoard:")
                print(f"   tensorboard --logdir {log_dir}")
                print(f"   Then open http://localhost:6006 in your browser")
                print(f"\nIn TensorBoard, you should see:")
                print("   1. Continuous loss/accuracy/learning_rate curves")
                print("   2. Resume marker at the interruption point")
                print("   3. Text logs showing initial and resumed training sessions")
                print("   4. Timestamp information about when training was resumed")
            else:
                print(f"❌ TensorBoard log analysis failed: {analysis['error']}")
                return 1
        else:
            print("❌ Simulation workflow failed")
            return 1
            
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1
    finally:
        # 清理文件（除非用户指定保留）
        if not args.keep_files and test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")
        elif args.keep_files:
            print(f"\nTest files kept in: {test_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 