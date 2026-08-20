"""
测试 PureDinoBackbone 的 forward 方法
验证 DINOv3 输出的形状和格式
"""
import torch
from soccermaster.models.pure_dino import PureDinoBackbone

def test_pure_dino_image_mode():
    """测试图像模式"""
    print("=" * 60)
    print("测试图像模式 (BACKBONE_TYPE='image')")
    print("=" * 60)
    
    # 初始化 backbone
    backbone = PureDinoBackbone(
        backbone_type='image',
        num_frames=1,
        ckpt_path='./pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        stage_1_ckpt_dir='./pretrained_models',
        text_encoder_ckpt_path='./pretrained_models/google/siglip2-base-patch16-512',
        use_lora=False,
        use_temporal_gate=False,
        freeze_vision_encoder=True,
        freeze_text_encoder=True,
        hidden_dim=1024  # DINOv3-Large 的输出维度
    )
    
    # 创建测试输入 [B, C, H, W]
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)
    
    print(f"\n输入形状: {images.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = backbone(images)
    
    # 检查输出
    print("\n输出字典的键:", outputs.keys())
    print(f"global_features 形状: {outputs['global_features'].shape}")
    print(f"local_features 形状: {outputs['local_features'].shape}")
    print(f"hidden_states 数量: {len(outputs['hidden_states'])}")
    print(f"hidden_states[0] 形状: {outputs['hidden_states'][0].shape}")
    print(f"hidden_states[-1] 形状: {outputs['hidden_states'][-1].shape}")
    print(f"text_features: {outputs['text_features']}")
    
    # 验证形状
    assert outputs['global_features'].shape == (batch_size, 1024), "global_features 形状错误"
    # local_features 应该是 [B, L, D]，其中 L = (H//16) * (W//16)
    expected_seq_len = (512 // 16) * (512 // 16)  # patch_size=16
    assert outputs['local_features'].shape[0] == batch_size, "local_features batch size 错误"
    assert outputs['local_features'].shape[2] == 1024, "local_features 维度错误"
    print(f"\n✓ 图像模式测试通过!")
    
def test_pure_dino_video_mode():
    """测试视频模式"""
    print("\n" + "=" * 60)
    print("测试视频模式 (BACKBONE_TYPE='video')")
    print("=" * 60)
    
    # 初始化 backbone
    backbone = PureDinoBackbone(
        backbone_type='video',
        num_frames=8,
        ckpt_path='./pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        stage_1_ckpt_dir='./pretrained_models',
        text_encoder_ckpt_path='./pretrained_models/google/siglip2-base-patch16-512',
        use_lora=False,
        use_temporal_gate=False,
        freeze_vision_encoder=True,
        freeze_text_encoder=True,
        hidden_dim=1024
    )
    
    # 创建测试输入 [B, T, C, H, W]
    batch_size = 2
    num_frames = 8
    images = torch.randn(batch_size, num_frames, 3, 512, 512)
    
    print(f"\n输入形状: {images.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = backbone(images)
    
    # 检查输出
    print("\n输出字典的键:", outputs.keys())
    print(f"global_features 形状: {outputs['global_features'].shape}")
    print(f"local_features 形状: {outputs['local_features'].shape}")
    print(f"hidden_states 数量: {len(outputs['hidden_states'])}")
    print(f"hidden_states[0] 形状: {outputs['hidden_states'][0].shape}")
    print(f"hidden_states[-1] 形状: {outputs['hidden_states'][-1].shape}")
    print(f"text_features: {outputs['text_features']}")
    
    # 验证形状
    assert outputs['global_features'].shape == (batch_size, num_frames, 1024), "global_features 形状错误"
    assert outputs['local_features'].shape[0] == batch_size, "local_features batch size 错误"
    assert outputs['local_features'].shape[1] == num_frames, "local_features num_frames 错误"
    assert outputs['local_features'].shape[3] == 1024, "local_features 维度错误"
    print(f"\n✓ 视频模式测试通过!")

def test_model_info():
    """打印模型信息"""
    print("\n" + "=" * 60)
    print("DINOv3 模型信息")
    print("=" * 60)
    
    backbone = PureDinoBackbone(
        backbone_type='image',
        num_frames=1,
        ckpt_path='./pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        stage_1_ckpt_dir='./pretrained_models',
        text_encoder_ckpt_path='./pretrained_models/google/siglip2-base-patch16-512',
        use_lora=False,
        use_temporal_gate=False,
        freeze_vision_encoder=True,
        freeze_text_encoder=True,
        hidden_dim=1024
    )
    
    print(f"\n模型嵌入维度: {backbone.vision_model.embed_dim}")
    print(f"模型层数: {backbone.vision_model.n_blocks}")
    print(f"注意力头数: {backbone.vision_model.num_heads}")
    print(f"Patch size: {backbone.vision_model.patch_size}")
    print(f"Storage tokens 数量: {backbone.vision_model.n_storage_tokens}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in backbone.vision_model.parameters())
    trainable_params = sum(p.numel() for p in backbone.vision_model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

if __name__ == '__main__':
    try:
        test_model_info()
        test_pure_dino_image_mode()
        test_pure_dino_video_mode()
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

