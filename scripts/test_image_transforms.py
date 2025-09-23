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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import yaml
import copy
from data.utils import ColorJitter, RandomHorizontalFlip, GaussianNoise, GaussianBlur, ClearAugmentationMetas, Compose, ToTensor, RandomResize, Normalize, RandomCrop, RandomAffine, RandomPerspective
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
    config["AUG_ENABLE_RANDOM_CROP"] = True
    config["AUG_ENABLE_RANDOM_AFFINE"] = True
    config["AUG_ENABLE_RANDOM_PERSPECTIVE"] = True
    config["AUG_ENABLE_RANDOM_HORIZONTAL_FLIP"] = True
    config["AUG_ENABLE_GAUSSIAN_NOISE"] = True
    config["AUG_ENABLE_GAUSSIAN_BLUR"] = True
    
    return config

def load_image_and_annotation(image_path, config):
    """加载图片和硬编码的annotation"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 使用PIL加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 转换为numpy数组，然后转为torch tensor
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()  # HWC -> CHW
    
    # 硬编码的annotation数据 - 基于真实GT数据
    annotation = {
        # bbox格式: [x, y, w, h] - 从GT数据中选择的几个球员
        "bbox": torch.tensor([
            [914.0, 855.0, 55.0, 172.0],    # track_id=1, 10号球员 (left team)
            [917.0, 575.0, 32.0, 122.0],    # track_id=2, 30号球员 (left team)
            [956.0, 557.0, 53.0, 133.0],    # track_id=3, 27号球员 (right team)
            [1257.0, 673.0, 44.0, 141.0],   # track_id=4, 10号球员 (right team)
            [799.0, 353.0, 55.0, 78.0],     # track_id=14, 裁判
            [1520.0, 478.0, 50.0, 82.0],    # track_id=7, 8号球员 (right team)
        ], dtype=torch.float32),
        
        # category: 0=person, 1=ball (所有都是人)
        "category": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64),
        
        # id: 目标的唯一标识符 (使用track_id)
        "id": torch.tensor([1, 2, 3, 4, 14, 7], dtype=torch.int64),
        
        # visibility: 可见性得分 (0-1) - 假设都可见
        "visibility": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        
        # role: 0=ball, 1=goalkeeper, 2=other, 3=player, 4=referee, 5=unknown
        "role": torch.tensor([3, 3, 3, 3, 4, 3], dtype=torch.int64),  # player, player, player, player, referee, player
        
        # jersey: 球衣号码 (0表示无号码或不可见)
        "jersey": torch.tensor([10, 30, 27, 10, 0, 8], dtype=torch.int64),  # 10号, 30号, 27号, 10号, 裁判(无号码), 8号
        
        # digit_head: 球衣号码十位数 (0表示无)
        "digit_head": torch.tensor([1, 3, 2, 1, 0, 0], dtype=torch.int64),  # 1(10号), 3(30号), 2(27号), 1(10号), 0(裁判), 0(8号)
        
        # digit_tail: 球衣号码个位数  
        "digit_tail": torch.tensor([0, 0, 7, 0, 0, 8], dtype=torch.int64),  # 0(10号), 0(30号), 7(27号), 0(10号), 0(裁判), 8(8号)
        
        # legibility_score: 号码清晰度得分 (0-1) - 基于有无jersey推测
        "legibility_score": torch.tensor([0.95, 0.90, 0.88, 0.92, 0.0, 0.85], dtype=torch.float32),
        
        # lines: 球场线条信息 (硬编码的真实lines数据)
        "lines": {
            "Side line top": [
                {
                    "x": 0.0,
                    "y": 0.2940777777777778
                },
                {
                    "x": 0.4890421875,
                    "y": 0.2992666666666666
                },
                {
                    "x": 0.77439375,
                    "y": 0.29635370370370373
                },
                {
                    "x": 1.0,
                    "y": 0.29099166666666665
                }
            ],
            "Middle line": [
                {
                    "x": 0.5051244791666667,
                    "y": 0.9998490740740741
                },
                {
                    "x": 0.5036705729166667,
                    "y": 0.7501189814814815
                },
                {
                    "x": 0.5022166666666666,
                    "y": 0.5003888888888889
                },
                {
                    "x": 0.5008604166666667,
                    "y": 0.2995601851851852
                }
            ]
        },
    }
    
    return image_tensor, image, annotation

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

def draw_bboxes_on_image(image, annotation, config, bbox_format="xywh"):
    """在图片上绘制bbox和lines
    
    Args:
        image: PIL图片
        annotation: 包含bbox和lines信息的annotation字典
        config: 配置字典
        bbox_format: bbox格式，"xywh"表示[x,y,w,h]，"cxcywh"表示[cx,cy,w,h]
    """
    if image is None:
        return None
    
    # 创建图片副本进行绘制
    img_with_annotations = image.copy()
    draw = ImageDraw.Draw(img_with_annotations)
    
    # 获取图片尺寸
    img_width, img_height = img_with_annotations.size
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 定义类别颜色
    category_colors = {
        0: "red",      # person
        1: "blue",     # ball
    }
    
    # 定义lines颜色
    line_colors = {
        "Side line top": "yellow",
        "Side line bottom": "yellow", 
        "Side line left": "yellow",
        "Side line right": "yellow",
        "Middle line": "green",
        "Goal left crossbar": "orange",
        "Goal right crossbar": "orange",
        "Big rect. left main": "cyan",
        "Big rect. right main": "cyan",
        "Small rect. left main": "magenta", 
        "Small rect. right main": "magenta",
    }
    
    # 角色映射 (与soccernet_gsr_reid.py中的role_mapping保持一致)
    role_names = {
        0: "ball",
        1: "goalkeeper", 
        2: "other",
        3: "player",
        4: "referee",
        5: "unknown",
    }
    
    if len(annotation["bbox"]) > 0:
        bboxes = annotation["bbox"]
        categories = annotation["category"]
        roles = annotation.get("role", torch.zeros_like(categories))
        jerseys = annotation.get("jersey", torch.zeros_like(categories))
        
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            category = int(categories[i])
            role = int(roles[i]) if i < len(roles) else 0
            jersey = int(jerseys[i]) if i < len(jerseys) else 0
            
            # 根据bbox格式进行转换
            if bbox_format == "cxcywh":
                # bbox格式: [cx, cy, w, h] -> [x1, y1, x2, y2]
                cx, cy, w, h = bbox
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
            else:  # "xywh"
                # bbox格式: [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 确保坐标在合理范围内
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            
            # 选择颜色
            color = category_colors.get(category, "green")
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            
            # 准备标签文本
            label_parts = []
            if category == 1:  # ball
                label_parts.append("ball")
            else:  # person
                role_name = role_names.get(role, f"role{role}")
                label_parts.append(role_name)
                if jersey > 0:
                    label_parts.append(f"#{jersey}")
            
            label = " ".join(label_parts)
            
            # 绘制标签背景
            bbox_label = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle(bbox_label, fill=color)
            
            # 绘制标签文字
            draw.text((x1, y1-20), label, fill="white", font=font)
    
    # 绘制lines
    if "lines" in annotation and annotation["lines"]:
        for line_name, points in annotation["lines"].items():
            if len(points) >= 2:
                # 获取线条颜色，如果没有预定义则使用白色
                line_color = line_colors.get(line_name, "white")
                
                # 转换归一化坐标到绝对坐标并绘制线条
                abs_points = []
                for point in points:
                    abs_x = point["x"] * img_width
                    abs_y = point["y"] * img_height
                    abs_points.append((abs_x, abs_y))
                
                # 绘制连接所有点的线条
                if len(abs_points) >= 2:
                    for i in range(len(abs_points) - 1):
                        draw.line([abs_points[i], abs_points[i + 1]], fill=line_color, width=2)
                    
                    # 在线条起点绘制标签
                    if abs_points:
                        start_x, start_y = abs_points[0]
                        # 绘制标签背景
                        bbox_label = draw.textbbox((start_x, start_y), line_name, font=font)
                        draw.rectangle(bbox_label, fill=line_color, outline=line_color)
                        # 绘制标签文字
                        draw.text((start_x, start_y), line_name, fill="black", font=font)
    
    return img_with_annotations

def apply_transforms_and_save(image_path, output_dir):
    """应用不同的数据增强并保存结果"""
    print(f"正在处理图片: {image_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置、图片和annotation
    config = load_config()
    image_tensor, original_pil, original_annotation = load_image_and_annotation(image_path, config)
    
    print(f"原始图片尺寸: {image_tensor.shape}")
    print(f"原始图片数据范围: [{image_tensor.min()}, {image_tensor.max()}]")
    print(f"找到 {len(original_annotation['bbox'])} 个目标")
    
    # 保存原始图片（带bbox和lines）
    original_with_annotations = draw_bboxes_on_image(original_pil, original_annotation, config)
    if original_with_annotations:
        original_with_annotations.save(os.path.join(output_dir, "00_original_with_annotations.jpg"))
        print("✓ 保存原始图片（带bbox和lines）")
    
    # 也保存不带bbox的原始图片
    original_pil.save(os.path.join(output_dir, "00_original.jpg"))
    print("✓ 保存原始图片")
    
    # 使用真实的annotation数据
    annotation = copy.deepcopy(original_annotation)
    metas = {}
    
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
    
    # 调整annotation的bbox坐标以适应resize
    original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]
    scale_h, scale_w = 224 / original_h, 224 / original_w
    annotation_resized = copy.deepcopy(annotation)
    if len(annotation_resized["bbox"]) > 0:
        annotation_resized["bbox"][:, 0] *= scale_w  # x
        annotation_resized["bbox"][:, 1] *= scale_h  # y  
        annotation_resized["bbox"][:, 2] *= scale_w  # w
        annotation_resized["bbox"][:, 3] *= scale_h  # h
    
    image_cj, annotation_cj, _ = color_jitter(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    # 保存Color Jitter结果
    cj_pil = tensor_to_pil(image_cj)
    cj_pil.save(os.path.join(output_dir, "01_color_jitter.jpg"))
    
    # 保存带bbox和lines的Color Jitter结果
    cj_with_annotations = draw_bboxes_on_image(cj_pil, annotation_cj, config)
    if cj_with_annotations:
        cj_with_annotations.save(os.path.join(output_dir, "01_color_jitter_with_annotations.jpg"))
    
    print("✓ 保存Color Jitter效果")
    
    # Random Horizontal Flip
    h_flip = RandomHorizontalFlip(p=1.0)  # 100%翻转
    image_hf, annotation_hf, _ = h_flip(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    hf_pil = tensor_to_pil(image_hf.float())
    hf_pil.save(os.path.join(output_dir, "02_horizontal_flip.jpg"))
    
    # 保存带bbox和lines的水平翻转结果
    hf_with_annotations = draw_bboxes_on_image(hf_pil, annotation_hf, config)
    if hf_with_annotations:
        hf_with_annotations.save(os.path.join(output_dir, "02_horizontal_flip_with_annotations.jpg"))
    
    print("✓ 保存水平翻转效果")
    
    # Gaussian Noise
    gauss_noise = GaussianNoise(
        mean=0.0,
        std=config["AUG_GAUSSIAN_NOISE_STD"],
        p=1.0
    )
    image_gn, annotation_gn, _ = gauss_noise(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    gn_pil = tensor_to_pil(image_gn)
    gn_pil.save(os.path.join(output_dir, "03_gaussian_noise.jpg"))
    
    # 保存带bbox和lines的高斯噪声结果
    gn_with_annotations = draw_bboxes_on_image(gn_pil, annotation_gn, config)
    if gn_with_annotations:
        gn_with_annotations.save(os.path.join(output_dir, "03_gaussian_noise_with_annotations.jpg"))
    
    print("✓ 保存高斯噪声效果")
    
    # Gaussian Blur
    gauss_blur = GaussianBlur(
        kernel_size_range=config["AUG_GAUSSIAN_BLUR_KERNEL_SIZE_RANGE"],
        sigma_range=config["AUG_GAUSSIAN_BLUR_SIGMA_RANGE"],
        p=1.0
    )
    image_gb, annotation_gb, _ = gauss_blur(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    gb_pil = tensor_to_pil(image_gb.float())
    gb_pil.save(os.path.join(output_dir, "04_gaussian_blur.jpg"))
    
    # 保存带bbox和lines的高斯模糊结果
    gb_with_annotations = draw_bboxes_on_image(gb_pil, annotation_gb, config)
    if gb_with_annotations:
        gb_with_annotations.save(os.path.join(output_dir, "04_gaussian_blur_with_annotations.jpg"))
    
    print("✓ 保存高斯模糊效果")
    
    # Random Crop
    random_crop = RandomCrop(
        crop_size_ratio_range=config["AUG_RANDOM_CROP_SIZE_RATIO_RANGE"],
        p=1.0
    )
    image_rc, annotation_rc, metas_rc = random_crop(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    rc_pil = tensor_to_pil(image_rc.float())
    rc_pil.save(os.path.join(output_dir, "05_random_crop.jpg"))
    
    # 保存带bbox和lines的随机裁剪结果
    rc_with_annotations = draw_bboxes_on_image(rc_pil, annotation_rc, config)
    if rc_with_annotations:
        rc_with_annotations.save(os.path.join(output_dir, "05_random_crop_with_annotations.jpg"))
    
    print("✓ 保存随机裁剪效果")
    
    # Random Affine
    random_affine = RandomAffine(
        degrees=config["AUG_AFFINE_DEGREES"],
        translate=config["AUG_AFFINE_TRANSLATE"],
        scale=config["AUG_AFFINE_SCALE"],
        shear=config["AUG_AFFINE_SHEAR"],
        # degrees=0,
        # translate=[0,0],
        # scale=[1.0,1.0],
        # shear=15,
        p=1.0
    )
    image_ra, annotation_ra, _ = random_affine(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    ra_pil = tensor_to_pil(image_ra.float())
    ra_pil.save(os.path.join(output_dir, "06_random_affine.jpg"))
    
    # 保存带bbox和lines的随机仿射变换结果
    ra_with_annotations = draw_bboxes_on_image(ra_pil, annotation_ra, config)
    if ra_with_annotations:
        ra_with_annotations.save(os.path.join(output_dir, "06_random_affine_with_annotations.jpg"))
    
    print("✓ 保存随机仿射变换效果")
    
    # Random Perspective
    random_perspective = RandomPerspective(
        distortion_scale=config["AUG_PERSPECTIVE_DISTORTION_SCALE"],
        p=1.0
    )
    image_rp, annotation_rp, _ = random_perspective(image_normalized, copy.deepcopy(annotation_resized), metas.copy())
    
    rp_pil = tensor_to_pil(image_rp.float())
    rp_pil.save(os.path.join(output_dir, "07_random_perspective.jpg"))
    
    # 保存带bbox和lines的随机透视变换结果
    rp_with_annotations = draw_bboxes_on_image(rp_pil, annotation_rp, config)
    if rp_with_annotations:
        rp_with_annotations.save(os.path.join(output_dir, "07_random_perspective_with_annotations.jpg"))
    
    print("✓ 保存随机透视变换效果")
    
    # 2. 测试完整的训练transforms
    print("\n=== 测试完整的训练transforms ===")
    
    train_transforms = build_transforms(config, split="train")
    print(f"训练transforms包含 {len(train_transforms.transforms)} 个步骤:")
    for i, transform in enumerate(train_transforms.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    # 应用完整的transforms
    # 注意：build_transforms会包含ToTensor，所以我们传入PIL图片
    image_full, ann_full, metas_full = train_transforms(original_pil, copy.deepcopy(original_annotation), metas.copy())
    
    # 由于完整transforms包含Normalize，我们需要反归一化来保存图片
    mean = torch.tensor(config["AUG_MEAN"]).view(3, 1, 1)
    std = torch.tensor(config["AUG_STD"]).view(3, 1, 1)
    H, W = image_full.shape[1], image_full.shape[2]
    ann_full["bbox"][:, [0, 2]] *= W  # x, width
    ann_full["bbox"][:, [1, 3]] *= H  # y, height
    
    # 反归一化: x = x * std + mean
    image_denorm = image_full * std + mean
    
    full_pil = tensor_to_pil(image_denorm)
    full_pil.save(os.path.join(output_dir, "08_full_train_transforms.jpg"))
    
    # 保存带bbox和lines的完整训练transforms结果 (使用cxcywh格式，因为经过了BoxXYWHtoCXCYWH变换)
    full_with_annotations = draw_bboxes_on_image(full_pil, ann_full, config, bbox_format="cxcywh")
    if full_with_annotations:
        full_with_annotations.save(os.path.join(output_dir, "08_full_train_transforms_with_annotations.jpg"))
    
    print("✓ 保存完整训练transforms效果")
    
    # 3. 测试测试时的transforms（应该没有数据增强）
    print("\n=== 测试测试时transforms ===")
    
    test_transforms = build_transforms(config, split="test")
    print(f"测试transforms包含 {len(test_transforms.transforms)} 个步骤:")
    for i, transform in enumerate(test_transforms.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    image_test, ann_test, metas_test = test_transforms(original_pil, copy.deepcopy(original_annotation), metas.copy())
    
    # 反归一化
    H, W = image_test.shape[1], image_test.shape[2]
    ann_test["bbox"][:, [0, 2]] *= W  # x, width
    ann_test["bbox"][:, [1, 3]] *= H  # y, height
    image_test_denorm = image_test * std + mean
    
    test_pil = tensor_to_pil(image_test_denorm)
    test_pil.save(os.path.join(output_dir, "09_test_transforms.jpg"))
    
    # 保存带bbox和lines的测试transforms结果 (使用cxcywh格式，因为经过了BoxXYWHtoCXCYWH变换)
    test_with_annotations = draw_bboxes_on_image(test_pil, ann_test, config, bbox_format="cxcywh")
    if test_with_annotations:
        test_with_annotations.save(os.path.join(output_dir, "09_test_transforms_with_annotations.jpg"))
    
    print("✓ 保存测试transforms效果")
    
    print(f"\n🎉 所有处理完成！结果保存在: {output_dir}")
    print("\n📁 输出文件说明:")
    print("  00_original.jpg                              - 原始图片")
    print("  00_original_with_annotations.jpg             - 原始图片（带bbox和lines标注）")
    print("  01_color_jitter.jpg                          - 颜色抖动效果")
    print("  01_color_jitter_with_annotations.jpg         - 颜色抖动效果（带bbox和lines）")
    print("  02_horizontal_flip.jpg                       - 水平翻转效果")
    print("  02_horizontal_flip_with_annotations.jpg      - 水平翻转效果（带bbox和lines）")
    print("  03_gaussian_noise.jpg                        - 高斯噪声效果")
    print("  03_gaussian_noise_with_annotations.jpg       - 高斯噪声效果（带bbox和lines）")
    print("  04_gaussian_blur.jpg                         - 高斯模糊效果")
    print("  04_gaussian_blur_with_annotations.jpg        - 高斯模糊效果（带bbox和lines）")
    print("  05_random_crop.jpg                           - 随机裁剪效果")
    print("  05_random_crop_with_annotations.jpg          - 随机裁剪效果（带bbox和lines）")
    print("  06_random_affine.jpg                         - 随机仿射变换效果")
    print("  06_random_affine_with_annotations.jpg        - 随机仿射变换效果（带bbox和lines）")
    print("  07_random_perspective.jpg                    - 随机透视变换效果")
    print("  07_random_perspective_with_annotations.jpg   - 随机透视变换效果（带bbox和lines）")
    print("  08_full_train_transforms.jpg                 - 完整训练变换效果")
    print("  08_full_train_transforms_with_annotations.jpg - 完整训练变换效果（带bbox和lines）")
    print("  09_test_transforms.jpg                       - 测试变换效果（无增强）")
    print("  09_test_transforms_with_annotations.jpg      - 测试变换效果（带bbox和lines）")

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
