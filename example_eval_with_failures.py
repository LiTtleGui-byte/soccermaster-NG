#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例脚本：如何运行评估并保存VideoCaption失败例子

使用方法:
python example_eval_with_failures.py --config path/to/config.yaml --checkpoint path/to/checkpoint
"""

import subprocess
import os
import sys

def run_eval_with_failures(config_path, checkpoint_path, output_dir="failure_analysis"):
    """
    运行评估并保存VideoCaption失败例子
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径
        output_dir: 输出目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成失败例子保存路径
    checkpoint_name = os.path.basename(checkpoint_path).replace('/', '_')
    failure_save_path = os.path.join(output_dir, f"video_caption_failures_{checkpoint_name}.txt")
    log_dir = os.path.join(output_dir, f"eval_logs_{checkpoint_name}")
    
    # 构建命令
    cmd = [
        sys.executable, "eval.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--log_dir", log_dir,
        "--save_video_caption_failures",
        "--failure_save_path", failure_save_path
    ]
    
    print("运行评估命令:")
    print(" ".join(cmd))
    print()
    
    # 运行评估
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 评估成功完成!")
        print(f"📝 失败例子保存在: {failure_save_path}")
        print(f"📊 评估日志保存在: {log_dir}")
        
        # 打印部分输出
        if result.stdout:
            print("\n--- 评估输出 ---")
            print(result.stdout[-1000:])  # 显示最后1000个字符
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        sys.exit(1)
    
    # 检查并显示失败例子文件
    if os.path.exists(failure_save_path):
        with open(failure_save_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"\n📄 失败例子文件内容预览 ({failure_save_path}):")
        print("=" * 50)
        
        # 显示前几个失败例子
        lines = content.split('\n')
        preview_lines = 0
        for line in lines:
            print(line)
            preview_lines += 1
            if preview_lines >= 50:  # 只显示前50行
                if len(lines) > 50:
                    print(f"\n... (还有 {len(lines) - 50} 行，请查看完整文件)")
                break
                
        # 统计失败例子数量
        failure_count = content.count("=== Failure Case ===")
        print(f"\n📊 总共发现 {failure_count} 个失败例子")
    else:
        print(f"\n⚠️  失败例子文件不存在: {failure_save_path}")
        print("这可能意味着没有发现失败例子，或者评估过程中出现了问题。")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行VideoCaption评估并保存失败例子")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--output_dir", type=str, default="failure_analysis", help="输出目录")
    
    args = parser.parse_args()
    
    run_eval_with_failures(args.config, args.checkpoint, args.output_dir)
