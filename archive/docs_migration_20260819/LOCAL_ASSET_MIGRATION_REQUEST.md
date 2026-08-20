# 本地大资产迁移申请（历史归档）

> 这份申请已经执行完成，仅作为迁移记录保留。当前资产位置和状态见`../../docs/ASSET_INVENTORY.md`。

状态：批次1已获用户批准并于2026-08-19完成。正式迁移退出码0，13项均经内容身份核对后从`.partial`原子改名为正式目标；结果为`.runtime/local_takeover/batch1/result.json`。本地目标配置位于`reproduction/configs/local_takeover/`，4份YAML和1份JSON均可解析且无`/remote-home`引用。

正式运行前目标文件系统可用163,237,179,392 bytes；完成后可用136,747,585,536 bytes（约127.4 GiB）。所有目标均位于Git忽略的`/home/tianlin/SoccerMaster/.local_assets/`，远端原件未被修改。

## 推荐批次1：主干与固定G10复现

SigLIP2已有用户受控NAS副本，因此未重复复制。本批清单apparent size总计26,492,433,131 bytes（约24.7 GiB）；正式结果记录的普通文件内容总量为26,492,396,267 bytes，差异来自目录元数据计量。

### SoccerMaster主干：7,089,887,126 bytes

| 来源 | 目标 | bytes |
|---|---|---:|
| `/remote-home/haolinyang/sports/Soccer-Backbone/outputs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19` | `.local_assets/checkpoints/soccermaster_epoch19` | 7,089,887,126 |

SigLIP2直接使用`/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817/models/siglip2-large-patch16-512`；该目录属于现有用户NAS bundle，已通过CPU身份检查。

### SoccerFactory权重：18,292,183,692 bytes

| 内容 | 来源 | 目标 | bytes |
|---|---|---|---:|
| YOLO人物检测 | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/yolo/yolo_v8x6_person_lr_default_best.pt` | `.local_assets/models/soccerfactory/pretrained_models/yolo/yolo_v8x6_person_lr_default_best.pt` | 195,209,883 |
| PRTReID | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar` | `.local_assets/models/soccerfactory/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar` | 396,287,605 |
| HRNet初始权重 | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth` | `.local_assets/models/soccerfactory/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth` | 165,587,602 |
| 球场关键点 | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/calibration/SV_kp` | `.local_assets/models/soccerfactory/pretrained_models/calibration/SV_kp` | 264,964,645 |
| 球场线 | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/calibration/SV_lines` | `.local_assets/models/soccerfactory/pretrained_models/calibration/SV_lines` | 264,857,893 |
| Legibility | `/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/legibility/legibility_resnet34_soccer_20240215.pth` | `.local_assets/models/soccerfactory/pretrained_models/legibility/legibility_resnet34_soccer_20240215.pth` | 85,289,629 |
| Qwen2.5-VL-7B | `/remote-home/haolinyang/huggingface_models/Qwen/Qwen2.5-VL-7B-Instruct` | `.local_assets/models/soccerfactory/pretrained_models/jn/Qwen2.5-VL-7B-Instruct` | 16,596,000,949 |
| Refiner | `/remote-home/haolinyang/sports/soccernet/Refiner/outputs/train_timesformer_100clip_coord_only_not_0init_l2_xyflip_seed42_20250328_224427/best_model.pth` | `.local_assets/models/soccerfactory/refiner/best_model.pth` | 323,985,486 |

### 固定复现数据：1,110,362,313 bytes

| 来源 | 目标 | bytes |
|---|---|---:|
| Chelsea–Burnley `1_720p.mkv` | `.local_assets/data/soccernet/raw/sngs10004/1_720p.mkv` | 979,439,471 |
| 对应`Labels-cameras.json` | `.local_assets/data/soccernet/cameras/sngs10004/Labels-cameras.json` | 106,194 |
| `sn_2_clip.json` | `.local_assets/data/SN-GSR-2024/SoccerNetGS/sn_2_clip.json` | 8,322,923 |
| NAS准备图片`SNGS-10004/img1` | `.local_assets/data/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1` | 122,493,725 |

## 已有受控资产：MatchTime自由生成解说

这部分与SoccerMaster五头主干不同，但已经存在于用户NAS：`/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817`，总目录58,218,439,162 bytes，CPU身份检查通过。运行时代码已经支持`SOCCERMASTER_COMMENTARY_ASSET_ROOT`，因此当前不申请再复制到本地文件系统。

| 资产 | bytes |
|---|---:|
| Meta-Llama-3-8B-Instruct | 32,132,599,005 |
| bert-base-uncased目录 | 3,454,169,471 |
| `model_save_11.pth` | 17,615,455,530 |
| 解说视觉`backbone.pt` | 1,435,281,181 |
| `match_time.pkl`限制词表 | 9,092 |

## 已执行边界

- 只复制了批准的批次和上述精确来源/目标。
- 正式执行使用了`SOCCERMASTER_LOCAL_ASSET_BATCH1_APPROVED=YES`批准守卫；没有该值时脚本拒绝复制。
- 目标必须为全新路径，不覆盖已有资产。
- 保留一个主复制日志；失败停止，不自动换来源或重跑。
- 复制后只检查目标存在、总大小和必要模型文件可读，不做无关逐文件hash。
- 本地复现配置已经指向`.local_assets/`并完成CPU静态解析；尚未用本地权重做GPU forward，任何GPU推理仍需新的`nvidia-smi`和当次授权。
