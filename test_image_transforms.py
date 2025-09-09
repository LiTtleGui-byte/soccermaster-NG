#!/usr/bin/env python3
"""
测试图片数据增强效果的脚本
加载指定图片，应用transforms，并保存结果进行对比
"""

import sys
import os
sys.path.append('/remote-home/haolinyang/sports/Soccer-Backbone')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import yaml
from data.utils import ColorJitter, RandomHorizontalFlip, GaussianNoise, GaussianBlur, ClearAugmentationMetas, Compose, ToTensor, RandomResize, Normalize
from data.soccernet_gsr_detection import build_transforms, BoxXYWHtoCXCYWH
from configs.util import yaml_to_dict
from torchvision.transforms import v2

def load_config():
    """加载默认配置"""
    config_path = '/remote-home/haolinyang/sports/Soccer-Backbone/configs/default.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 启用训练时的数据增强
    config["AUG_ENABLE_TRAINING_AUGMENTATION"] = True
    config["AUG_COLOR_JITTER_V2"] = True
    
    return config

def load_image(image_path):
    """加载图片并转换为tensor格式"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 使用PIL加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 转换为numpy数组，然后转为torch tensor
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()  # HWC -> CHW
    
    return image_tensor, image

def tensor_to_pil(tensor):
    """将tensor转换回PIL图片用于保存"""
    # 确保tensor在正确的范围内
    if tensor.max() <= 1.0:
        # 如果是归一化的tensor [0,1]，转换到[0,255]
        tensor = (tensor * 255).clamp(0, 255)
    else:
        # 如果已经是[0,255]范围，只需要clamp
        tensor = tensor.clamp(0, 255)
    
    # 转换为uint8并重新排列维度 CHW -> HWC
    tensor = tensor.byte().permute(1, 2, 0).cpu().numpy()
    
    return Image.fromarray(tensor)

def apply_transforms_and_save(image_path, output_dir):
    """应用不同的数据增强并保存结果"""
    print(f"正在处理图片: {image_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置和图片
    config = load_config()
    image_tensor, original_pil = load_image(image_path)
    
    print(f"原始图片尺寸: {image_tensor.shape}")
    print(f"原始图片数据范围: [{image_tensor.min()}, {image_tensor.max()}]")
    
    # 保存原始图片
    original_pil.save(os.path.join(output_dir, "00_original.jpg"))
    print("✓ 保存原始图片")
    
    # 创建虚拟的标注数据（因为transforms需要）
    dummy_annotation = {
        "bbox": torch.tensor([[100.0, 100.0, 50.0, 50.0]]),  # x, y, w, h
        "category": torch.tensor([1]),
        "lines": {}
    }
    dummy_metas = {}
    
    # 1. 测试单个数据增强效果
    print("\n=== 测试单个数据增强效果 ===")
    
    # Color Jitter
    color_jitter = ColorJitter(
        brightness=config["AUG_BRIGHTNESS"],
        contrast=config["AUG_CONTRAST"], 
        saturation=config["AUG_SATURATION"],
        hue=config["AUG_HUE"],
        p=1.0  # 100%应用
    )
    
    # 将图片转换为[0,1]范围用于Color Jitter
    image_normalized = image_tensor.float() / 255.0
    # 将图片resize到224x224
    image_normalized = torch.clamp(v2.functional.resize(image_normalized, [224, 224], interpolation=v2.InterpolationMode.BICUBIC), 0, 1)
    print(f"调整后图片尺寸: {image_normalized.shape}")
    
    image_cj, _, _ = color_jitter(image_normalized, dummy_annotation, dummy_metas.copy())
    
    # 保存Color Jitter结果
    cj_pil = tensor_to_pil(image_cj)
    cj_pil.save(os.path.join(output_dir, "01_color_jitter.jpg"))
    print("✓ 保存Color Jitter效果")
    
    # Random Horizontal Flip
    h_flip = RandomHorizontalFlip(p=1.0)  # 100%翻转
    image_hf, _, _ = h_flip(image_normalized, dummy_annotation, dummy_metas.copy())
    
    hf_pil = tensor_to_pil(image_hf.float())
    hf_pil.save(os.path.join(output_dir, "02_horizontal_flip.jpg"))
    print("✓ 保存水平翻转效果")
    
    # Gaussian Noise
    gauss_noise = GaussianNoise(
        mean=0.0,
        std=config["AUG_GAUSSIAN_NOISE_STD"],
        p=1.0
    )
    image_gn, _, _ = gauss_noise(image_normalized, dummy_annotation, dummy_metas.copy())
    
    gn_pil = tensor_to_pil(image_gn)
    gn_pil.save(os.path.join(output_dir, "03_gaussian_noise.jpg"))
    print("✓ 保存高斯噪声效果")
    
    # Gaussian Blur
    gauss_blur = GaussianBlur(
        kernel_size_range=config["AUG_GAUSSIAN_BLUR_KERNEL_SIZE_RANGE"],
        sigma_range=config["AUG_GAUSSIAN_BLUR_SIGMA_RANGE"],
        p=1.0
    )
    image_gb, _, _ = gauss_blur(image_normalized, dummy_annotation, dummy_metas.copy())
    
    gb_pil = tensor_to_pil(image_gb.float())
    gb_pil.save(os.path.join(output_dir, "04_gaussian_blur.jpg"))
    print("✓ 保存高斯模糊效果")
    
    # 2. 测试完整的训练transforms
    print("\n=== 测试完整的训练transforms ===")
    
    train_transforms = build_transforms(config, split="train")
    print(f"训练transforms包含 {len(train_transforms.transforms)} 个步骤:")
    for i, transform in enumerate(train_transforms.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    # 应用完整的transforms
    # 注意：build_transforms会包含ToTensor，所以我们传入PIL图片
    image_full, ann_full, metas_full = train_transforms(original_pil, dummy_annotation, dummy_metas.copy())
    
    # 由于完整transforms包含Normalize，我们需要反归一化来保存图片
    mean = torch.tensor(config["AUG_MEAN"]).view(3, 1, 1)
    std = torch.tensor(config["AUG_STD"]).view(3, 1, 1)
    
    # 反归一化: x = x * std + mean
    image_denorm = image_full * std + mean
    
    full_pil = tensor_to_pil(image_denorm)
    full_pil.save(os.path.join(output_dir, "05_full_train_transforms.jpg"))
    print("✓ 保存完整训练transforms效果")
    
    # 3. 测试测试时的transforms（应该没有数据增强）
    print("\n=== 测试测试时transforms ===")
    
    test_transforms = build_transforms(config, split="test")
    print(f"测试transforms包含 {len(test_transforms.transforms)} 个步骤:")
    for i, transform in enumerate(test_transforms.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    image_test, ann_test, metas_test = test_transforms(original_pil, dummy_annotation, dummy_metas.copy())
    
    # 反归一化
    image_test_denorm = image_test * std + mean
    
    test_pil = tensor_to_pil(image_test_denorm)
    test_pil.save(os.path.join(output_dir, "06_test_transforms.jpg"))
    print("✓ 保存测试transforms效果")
    
    print(f"\n🎉 所有处理完成！结果保存在: {output_dir}")
    print("\n📁 输出文件说明:")
    print("  00_original.jpg           - 原始图片")
    print("  01_color_jitter.jpg       - 颜色抖动效果")
    print("  02_horizontal_flip.jpg    - 水平翻转效果")
    print("  03_gaussian_noise.jpg     - 高斯噪声效果")
    print("  04_gaussian_blur.jpg      - 高斯模糊效果")
    print("  05_full_train_transforms.jpg - 完整训练变换效果")
    print("  06_test_transforms.jpg    - 测试变换效果（无增强）")

def main():
    """主函数"""
    image_path = "/remote-home/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS/train/SNGS-060/img1/000001.jpg"
    output_dir = "/remote-home/haolinyang/sports/Soccer-Backbone/transform_results"
    
    try:
        apply_transforms_and_save(image_path, output_dir)
        return True
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
