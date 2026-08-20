# SoccerMaster 代码与 Pipeline 位置

当前代码有两个容易混淆的分支：主干多任务模型和 MatchTime 解说生成模型。它们共享 SigLIP2 思路和部分资产，但不是同一个输出头或 checkpoint。

## 主干多任务模型

```text
图片/视频帧
  → Dataset 与预处理
  → SigLIP2 + UniSoccer temporal backbone
  → 多任务 head
  → 指标/可视化/训练
```

| 阶段 | 主要代码 | 主要输出 | 当前状态 |
|---|---|---|---|
| Dataset/预处理 | `research/src/soccermaster/data/soccernet_gsr_detection.py`、`research/src/soccermaster/data/video_caption.py`、`research/src/soccermaster/data/video_utils_siglip.py` | `[B,T,3,H,W]`、标注和 `metas` | G4/G5/G10-C 固定样本结构已通 |
| 主模型组装 | `research/src/soccermaster/models/multi_task.py` | 选择 backbone 和 `multi_task_head` | 主干模型入口 |
| SigLIP2 标准/包装 | `research/src/soccermaster/models/siglip2.py`、`research/src/soccermaster/models/siglip2_unisoccer.py`、`research/src/soccermaster/models/pure_siglip.py` | 图像/文本特征 | 本地代码，权重远端只读 |
| 时间视觉表示 | `research/src/soccermaster/models/siglip2_unisoccer_part_temporal.py`、`research/src/soccermaster/models/modeling_timesformer_siglip.py` | video `global_features`、local features | 当前配置使用 `unisoccer_part_temporal`，`TEMPORAL_START_LAYER=16` |
| 检测头 | `research/src/soccermaster/models/deformable_detr/deformable_detr.py`，由 `build_deformable_detr_head()` 构造 | boxes、person/object logits、role、jersey、digits | G3/G5 forward 与固定评估已通 |
| 球场线头 | `research/src/soccermaster/models/lines_detection.py` | lines heatmap/目标 | G3/G5 结构和固定指标已通 |
| 球场关键点头 | `research/src/soccermaster/models/keypoints_detection.py` | keypoint heatmap/目标 | G3/G5 结构和固定指标已通 |
| VideoCaption 头 | `research/src/soccermaster/models/video_caption.py` | vision/text 对齐特征和 loss/metrics | G3/G4/G5 已通 |
| CaptionClassification 头 | `research/src/soccermaster/models/caption_classification.py` | 类别 logits/features | G3/G4/G5/G6 已通 |
| 可选头 | `research/src/soccermaster/models/soccernet_gsr_reid.py`、`research/src/soccermaster/models/camera.py`、`research/src/soccermaster/models/caption_classification_align.py` | ReID、相机回归、对齐分类 | 代码存在，但不属于当前默认五头组合 |
| 训练入口 | `train.py`、`archive/reproduction_20260819/gates/g7_single_task.py`、`archive/reproduction_20260819/g8_multitask_run.py` | optimizer、checkpoint、validation | G7/G8 小规模通过；G9 完整训练未通过 |
| 评估入口 | `eval.py`、`archive/reproduction_20260819/gates/g5_fixed_eval.py` | 五类固定评估指标 | 固定小规模通过，不代表全量指标 |
| 可视化 | `archive/experiments_20260819/task_head_visualization/generate_from_evidence.py`、`scripts/vis_*.py`、`runs/reports_legacy_20260819/g5/*overlay*.jpg` | 检测/线/关键点等图 | 已有固定样本证据 |

## 修改代码时从哪里进入

| 想修改什么 | 首先看哪里 | 关键类/函数 | 会影响什么 |
|---|---|---|---|
| 整体组装、启用哪些任务头 | `research/src/soccermaster/models/multi_task.py` | `MultiTaskingSigLIP`、`self.multi_task_head` | backbone选择、任务头注册、checkpoint加载和总forward |
| SigLIP2逐帧视觉编码 | `research/src/soccermaster/models/siglip2_unisoccer_part_temporal.py` | `SiglipBackbone`、`UniSoccerBackbone` | patch/frame特征与全局视觉表示 |
| 时间信息处理 | `research/src/soccermaster/models/siglip2_unisoccer_part_temporal.py`、`research/src/soccermaster/models/modeling_timesformer_siglip.py` | `Timesformer`、`ResidualAttentionBlock`、`SiglipTemporalAttention` | 30帧之间如何交换信息；当前配置从第16层开始时间建模 |
| 人物框、角色、号码 | `research/src/soccermaster/models/deformable_detr/deformable_detr.py` | `DeformableDetrHead`、`build_deformable_detr_head` | bbox、人物置信度、角色与球衣号码相关输出 |
| 球场线 | `research/src/soccermaster/models/lines_detection.py` | `LinesDetection`、`build_lines_detection_head` | 线热图、loss和metrics |
| 球场关键点 | `research/src/soccermaster/models/keypoints_detection.py` | `KeypointsDetection`、`build_keypoints_detection_head` | 关键点热图、loss和metrics |
| 视频—文本匹配 | `research/src/soccermaster/models/video_caption.py` | `VideoCaptionHead`、`build_video_caption_head` | 视频与文字的对齐特征；不是自由生成句子 |
| 23类事件分类 | `research/src/soccermaster/models/caption_classification.py` | `CaptionClassificationHead`、`build_caption_classification_head` | 每段视频的事件类别logits |
| Detection数据与标注转换 | `research/src/soccermaster/data/soccernet_gsr_detection.py` | `SoccerNetGSR_Detection`、`build_gsr_detection_dataloader` | 30帧图像、人物/线/点/相机标注如何进入模型 |
| Caption数据采样 | `research/src/soccermaster/data/video_caption.py` | `VideoCaptionDataset`、`get_frame_indices` | 视频如何抽成固定帧数及文字如何配对 |
| 训练调度 | `train.py` | 主训练循环 | 多任务batch、loss、backward、optimizer、scheduler与checkpoint |

`research/src/soccermaster/models/multi_task.py` 中的当前配置映射为：

```text
SoccerNetGSR_Detection → SoccerNetGSR_Detection + LinesDetection + KeypointsDetection
VideoCaption          → VideoCaption + CaptionClassification
```

## MatchTime 解说生成分支

```text
30 帧视频
  → `runtime/dataset/commentary.py`
  → `MatchVision_part_temporal.py`
  → 时间位置编码
  → `video_Qformer`（当前 2 层）
  → `llama_proj`
  → Llama-3 decoder
  → 文本解码
```

| 阶段 | 代码位置 | 说明 |
|---|---|---|
| 视频读取/采样 | `archive/experiments_20260819/commentary_generation/runtime/dataset/commentary.py`、`video_utils_siglip.py` | 30 帧 `middle` 采样和 SigLIP 预处理 |
| 视觉编码 | `archive/experiments_20260819/commentary_generation/runtime/model/MatchVision_part_temporal.py` | 解说 checkpoint 对应的视觉编码器 |
| Q-Former | `archive/experiments_20260819/commentary_generation/runtime/model/matchvoice_Qformer.py` | BERT 风格 Q-Former；层数由 `matchvoice_model_all_blocks.py` 初始化为 2 |
| 主 forward | `archive/experiments_20260819/commentary_generation/runtime/model/matchvoice_model_all_blocks.py` | 视觉、时间、Q-Former、projector 和 Llama 接口 |
| Projector | 同上文件中的 `self.llama_proj` | Q-Former hidden state → Llama hidden size |
| Llama 生成 | 同上文件 `generate_text()`；固定实验复用 `cached_prefix_experiments.py` | 当前 decoder 可重复对照，但 attention mask 风险仍记录在资产/实验说明中 |
| 固定入口 | `archive/experiments_20260819/commentary_generation/infer_one.py`、`decode_ablation_200.py` | 单样本和固定 200 条解码 |
| 深度诊断 | `run_stage2_layer_probe_cpu.py`、`render_qformer_depth_screen.py`、`run_qformer_layer_exit_gpu.py` | 只作候选筛查，不能替代匹配接口训练 |
| 解说可视证据 | `runs/reports_legacy_20260819/commentary_trace/`、`runs/reports_legacy_20260819/commentary_*review*` | 帧、接触图和人工审查包 |

这里最常改的三个位置是：

- Q-Former层数/query与cross-attention：`matchvoice_model_all_blocks.py`中的`init_video_Qformer()`以及`matchvoice_Qformer.py`中的`BertLayer/BertEncoder`。
- Projector：`matchvoice_model_all_blocks.py`中的`self.llama_proj`。
- Llama输入与解码参数：同文件`generate_text()`；限制词表逻辑位于`RestrictTokenGenerationLogitsProcessor`。

## 固定单场可视化入口

统一报告：`runs/reports_legacy_20260819/one_match/20260819_sngs10004_end_to_end/summary.md`。

| 子任务 | 实际图片 | 解释边界 |
|---|---|---|
| 人物检测 | `visuals/06_soccermaster_detection_overlay.jpg` | 一个30帧clip上的模型框，不是全数据指标 |
| 球场线 | `visuals/07_soccermaster_lines_heatmap_overlay.jpg` | 预测热图叠加 |
| 球场关键点 | `visuals/08_soccermaster_keypoints_heatmap_overlay.jpg` | 预测热图叠加 |
| 事件分类 | `visuals/09_soccermaster_caption_classification_top5.png` | 23类top-5，当前top-1为`clearance` |
| 视频—文本匹配 | `visuals/10_soccermaster_video_caption_retrieval_top5.png` | 对23条固定事件短语检索，当前top-1为`ball possession`；不是生成解说 |

这五张图证明固定真实输入完成了五头forward。G9完整单epoch因18小时timeout停在`8299/16009`且没有checkpoint，所以“推理链已通”不能写成“完整训练已通”。

解说分支的来源关系保存在 `archive/experiments_20260819/commentary_generation/sources.json`。其中 optimizer/metrics 仍只是远端来源记录，不应误写成已全部本地化。

## 代码归属建议

先保持当前路径和 checkpoint 不变，用本页作为导航；稳定后再把实际使用的源码闭包复制到本地 `vendor/` 或 `runtime/`。不要在未冻结基线前移动 `research/src/soccermaster/models/`、`research/src/soccermaster/data/` 或替换 high-resolution/解说 checkpoint。
