# 资产清单

更新：2026-08-19。本文只记录资产在哪里、由谁提供、目前证明到哪一步；不把“字段存在”写成“字段准确”。原始远端目录和共享环境保持只读。

## 状态标记

- **本地**：代码或派生结果已经在本工作区，可以修改。
- **远端只读**：运行时依赖的原始代码、数据或权重；目前不复制、不修改。
- **结构已通**：输入/输出格式和链路有运行证据。
- **质量未知**：没有独立真值，不能判断语义是否正确。
- **候选已定位**：找到了历史实现，但输入映射或副作用仍未完成复核。
- **未恢复**：当前还没有找到可安全重现的实现或资产。

## 当前接管结论

| 范围 | 当前结论 | 仍缺什么 |
|---|---|---|
| SoccerMaster 主干代码 | `research/src/soccermaster/models/`、`research/src/soccermaster/data/`、配置、训练和评估入口均已在本地，可修改；`epoch_19`已迁入`assets/`，SigLIP2使用用户受控NAS副本；本地五头GPU forward已通过 | 仅证明固定30帧接口，不代表完整训练或总体质量 |
| SoccerMaster 五头推理 | 固定SNGS-10004真实30帧已完成五头forward和5张输出图 | 只证明单片段接口；VideoCaption是固定短语检索，不是自由生成解说 |
| SoccerMaster 训练 | G1–G8通过；G9六卡稳定运行8,299/16,009步 | 单epoch未完成、无epoch checkpoint，完整训练未通过 |
| SoccerFactory 固定链 | 原始半场视频→本地255帧→Step 1→enrichment→Refiner→适配Step 3→PKL→DataLoader已连通；本地源码/权重动态复核完成 | 接口已通，但角色、球队、号码和坐标语义质量未通过 |
| 本地化 | 本地Python、源码、主干/SoccerFactory权重、固定输入、launcher、派生产物和报告均已进入用户控制路径；两条固定推理链均从本地/用户受控资产动态运行 | 完整数据集和完整训练没有迁移/复现，不在批次1目标内 |

## 运行环境

| 类型 | 位置 | 状态 |
|---|---|---|
| 本地 Gate Python | `/home/tianlin/SoccerMaster/.envs/SoccerMaster-repro/bin/python` | 本地；G1/G2 等价检查通过；旧 `.local_envs` 名称保留兼容链接 |
| 本地环境归档 | `/home/tianlin/SoccerMaster/.envs/SoccerMaster-repro.tar` | 本地；可恢复但未经批准不得清理 |
| 共享参考 Python | `/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python` | 远端只读；SoccerFactory 历史入口使用 |
| 本地工作区 | `/home/tianlin/SoccerMaster` | 本地；所有修改只放这里 |
| 原始 SoccerMaster/Soccer-Backbone | `/remote-home/haolinyang/sports/Soccer-Backbone` | 远端只读 |

## SoccerMaster 主干资产

| 资产 | 位置 | 用途/状态 |
|---|---|---|
| high-resolution 配置 | `research/configs/baseline/soccermaster_high_resolution.yaml` | 本地；当前主干配置，30 帧、512×512、`unisoccer_part_temporal` |
| SigLIP2 初始模型 | `/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817/models/siglip2-large-patch16-512` | 用户受控NAS；CPU身份检查通过；视觉/文本初始化 |
| 主干 `epoch_19` | `assets/checkpoints/official/soccermaster/epoch_19` | 本地规范目录；兼容旧路径的链接仍保留；包含backbone、text model和任务头，本地五头GPU forward已通过 |
| 检测/球场数据 | `/remote-home/haolinyang/datasets/SN-GSR-2024` 及 `.runtime/data_views/g5` | 原始数据只读；固定 G5 已消费 |
| MatchTime 标注 | `/remote-home/haolinyang/sports/UniSoccer/train_data/video_clip_json/MatchTime/` | 远端只读；解说分支输入 |
| MatchTime 视频 | `/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/SoccerNetv2/MatchTime/` | NAS 只读；训练/测试覆盖已盘点 |

主干可编辑代码位于 `research/src/soccermaster/`，当前配置位于 `research/configs/`；上游只读快照位于 `baseline/code/soccermaster/`，历史脚本位于 `archive/tools_20260819/`。

## SoccerMaster 解说分支资产

这一分支已经有一份属于当前用户的受控NAS bundle：`/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817`，总大小58,218,439,162 bytes。CPU身份和checkpoint元数据检查已通过，运行时可通过`SOCCERMASTER_COMMENTARY_ASSET_ROOT`直接切换，不必再从`/remote-home`复制一份。

| 资产 | 位置 | 用途/状态 |
|---|---|---|
| 生成模型运行模块 | `archive/experiments_20260819/commentary_generation/runtime/` | 本地 vendored 最小运行闭包 |
| 来源清单 | `archive/experiments_20260819/commentary_generation/sources.json` | 记录每个 vendored 文件的远端来源和版本身份 |
| 解说 checkpoint | `/remote-home/haolinyang/sports/dirty_code/UniSoccer/output/large_512_multitask_w_1_epoch_19_train_matchtime_eval_matchtime_half_lr_bf16/model_save_11.pth` | 远端只读；17,615,455,530 bytes；epoch 11 |
| 解说视觉 backbone | `/remote-home/haolinyang/sports/Soccer-Backbone/outputs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000/epoch_19/backbone.pt` | 远端只读；1,435,281,181 bytes；不是主干high-resolution checkpoint，不能静默替换 |
| Llama | `/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct` | 远端只读；32,132,599,005 bytes |
| Q-Former 配置 | `/remote-home/share/huggingface/bert-base-uncased` | 远端只读；3,454,169,471 bytes；当前 Q-Former 为2层，但迁移时需确认实际只读取哪些文件 |
| 解说生成限制词表 | `/remote-home/haolinyang/sports/UniSoccer/words_world/match_time.pkl` | 远端只读 |
| 解说实验报告 | `runs/reports_legacy_20260819/commentary_*/` | 本地派生证据；不作为新模型资产 |
| 用户NAS完整bundle | `/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817` | 用户受控；Llama、BERT、SigLIP2、视觉backbone、生成checkpoint、MatchTime标注和词表齐全；CPU预检通过 |

## SoccerFactory 资产

| 资产 | 位置 | 用途/状态 |
|---|---|---|
| TrackLab 源码 | 来源`/remote-home/haolinyang/sports/soccernet/tracklab`；本地`baseline/code/soccerfactory/tracklab` | 源码已复制；Step 1实现；整个来源目录约100,494,966 apparent bytes，但只vendored核心package、plugins和许可/构建元数据 |
| sn-gamestate 源码 | 来源`/remote-home/haolinyang/sports/soccernet/sn-gamestate`；本地`baseline/code/soccerfactory/sn-gamestate` | 源码已复制；enrichment/Step 3实现；TB级历史`runs/outputs_legacy_20260819/`未复制 |
| Refiner 源码 | 来源`/remote-home/haolinyang/sports/soccernet/Refiner`；本地`baseline/code/soccerfactory/refiner` | 推理入口、configs/model/dataset已复制；历史outputs/weights/visualization未复制 |
| Refiner checkpoint | `assets/checkpoints/official/soccerfactory/refiner/best_model.pth` | 本地；323,985,486 bytes |
| YOLO人物权重 | `assets/checkpoints/official/soccerfactory/pretrained_models/yolo/yolo_v8x6_person_lr_default_best.pt` | 本地；195,209,883 bytes |
| PRTReID任务权重 | `assets/checkpoints/official/soccerfactory/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar` | 本地；396,287,605 bytes |
| PRTReID HRNet初始权重 | `assets/checkpoints/official/soccerfactory/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth` | 本地；165,587,602 bytes |
| 球场关键点/线权重 | `assets/checkpoints/official/soccerfactory/pretrained_models/calibration/SV_kp`、`SV_lines` | 本地；264,964,645 + 264,857,893 bytes |
| Legibility权重 | `assets/checkpoints/official/soccerfactory/pretrained_models/legibility/legibility_resnet34_soccer_20240215.pth` | 本地；85,289,629 bytes |
| Qwen号码OCR模型 | `assets/checkpoints/official/soccerfactory/pretrained_models/jn/Qwen2.5-VL-7B-Instruct` | 本地；16,596,000,949 apparent bytes |
| Step 1/Enrichment/Refiner launcher | `archive/reproduction_20260819/gates/g10_*.py` | 本地；隔离、预检和运行边界 |
| Step 1/Enrichment 配置 | `archive/reproduction_20260819/configs/g10/` | 本地；固定 SNGS-10004 配置 |
| 固定原始视频 | `assets/data/soccernet/raw/sngs10004/1_720p.mkv` | 本地；979,439,471 bytes；来源保留在迁移清单 |
| 固定镜头标注 | `assets/data/soccernet/cameras/sngs10004/Labels-cameras.json` | 本地；106,194 bytes |
| 原始视频切片代码 | `archive/tools_20260819/segment_soccernet.py`，安全单片段逻辑由`archive/experiments_20260819/one_match_visualization/`调用 | 历史工具；固定`clip_id=4`已只写工作区并重建255帧；历史批处理入口仍因远端写副作用不能直接运行 |
| 视频片段映射 | `assets/data/SN-GSR-2024/SoccerNetGS/sn_2_clip.json` | 本地；8,322,923 bytes；`clip_id=4`对应`SNGS-10004` |
| 固定样本 raw-video lineage 候选清单 | `archive/reproduction_20260819/manifests/g10_raw_video_lineage_candidate_sngs10004.json` | 本地；记录修正后的 `clip_id=4` 候选，不是执行授权 |
| extracted_info 拆分工具 | `/remote-home/haolinyang/sports/Soccer-Backbone/tmp/split_extracted_info.py` | 远端只读；只把已有 PKL 按序列拆开，不负责原视频切帧 |
| 准备图片 | `assets/data/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1` | 本地；255张，122,493,725 apparent bytes |
| 当前训练 PKL | `.runtime/g10/sngs10004_current_pipeline_conversion/run1/SNGS-10004.pkl` | 本地派生；DataLoader smoke 已通过 |
| SoccerFactory 可视化 | `archive/experiments_20260819/soccerfactory_visualization/`、`archive/experiments_20260819/soccerfactory_step3/`、`archive/experiments_20260819/one_match_visualization/` | 本地；原视频、Step 1、标定、属性、Refiner和最终消费共8张图已生成 |

原始视频编号已经按实际命名规则纠正：`SNGS-10004`对应mapping `clip_id=4`（Chelsea–Burnley，帧`2278–2532`，255帧），不是旧manifest误写的`clip_id=10004`。2026-08-19的隔离单片段入口已从979,439,471-byte半场视频重建255张本地图片，并通过逐帧对应检查；输出为122,497,821 bytes。这里确认的是固定片段lineage和接口，不是历史批处理脚本可安全原样运行，也不代表其他比赛已复现。

这项编号矛盾是结构复核问题，不是角色、球队、号码或坐标质量结论；后者继续单独记录。

## 本地化批次建议

批次1已完成，完成后文件系统可用136,747,585,536 bytes（约127.4 GiB）。迁移没有按“整仓复制”执行，`sn-gamestate`和Refiner的历史输出未复制。

| 批次 | 内容 | 当前估算 | 决策 |
|---|---|---:|---|
| A：源码闭包 | TrackLab核心、sn-gamestate核心、Refiner实际入口/配置，以及来源说明 | 已落地约11.80MB | 源/目标核心963文件和10,871,371 bytes对应；9个关键模块本地import smoke通过，尚待正式配置切换 |
| B：主干权重 | high-resolution `epoch_19`；SigLIP2复用用户NAS bundle | 已完成7,089,887,126 bytes | 固定30帧本地五头GPU forward已通过 |
| C：固定复现数据 | SNGS-10004源视频、camera JSON、255张准备图片及必要mapping | 已完成1,110,362,313 bytes | 固定复现规范副本已落地 |
| D：SoccerFactory权重 | YOLO、ReID、标定、legibility、Qwen、Refiner | 已完成18,292,183,692 bytes | 全部完成固定样本本地GPU forward |
| E：生成解说模型 | 用户NAS bundle | 已存在58,218,439,162 bytes | 不建议再复制；通过环境变量使用受控NAS资产即可 |

源码闭包和批准的批次1数据/权重均已落地；原始资产继续只读，仅作为来源证明。未批准的完整数据集、历史训练输出和无关模型没有复制。

相关机器清单：

- 主干资产：`archive/experiments_20260819/commentary_generation/assets.json`
- 解说源码来源：`archive/experiments_20260819/commentary_generation/sources.json`
- SoccerFactory 固定资产：`archive/reproduction_20260819/manifests/g10_*` 和 `archive/reproduction_20260819/configs/g10/`
- 新实验统一资产入口：`research/configs/assets/local_assets.yaml`
- 本地目录与 checkpoint 规则：`docs/LOCAL_PROJECT_LAYOUT.md`
