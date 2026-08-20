# SoccerFactory 代码与 Pipeline 位置

当前固定样本已经从原始Chelsea–Burnley半场视频本地重建为 **255 张准备图片**，并继续串到训练消费端。下面把“固定单片段接口已通”“历史批处理入口是否原样可用”和“输出质量是否准确”分开记录。

## 当前链路

```text
原始半场视频 + camera shot标注 + clip mapping
  → 本地隔离生成255张准备图片
  → Step 1: bbox detector → ReID → StrongSORT
  → enrichment: pitch/calibration/apply_camera_params
                 → legibility/OCR → tracklet aggregation → team → team_side
  → Refiner（coord-only）
  → 适配后的 Step 3 后处理
  → 训练 PKL
  → SoccerMaster DataLoader
```

## 阶段位置

| 阶段 | 本地入口/配置 | 实际实现来源 | 主要输入/输出 | 状态 |
|---|---|---|---|---|
| 原视频 → 准备图片 | `scripts/segment_soccernet.py`；隔离执行入口位于`archive/experiments_20260819/one_match_visualization/` | 本地安全逻辑来自远端同名脚本；依赖`dataset-720p`、`dataset-cameras`、`sn_2_clip.json` | `Labels-cameras.json` + 半场视频 → 主镜头区间 → JPEG `img1/000001.jpg...` | 固定`clip_id=4`已在工作区重建255帧并通过逐帧对应；历史批处理入口仍禁止直接运行 |
| 已有 PKL 拆分 | 无本地入口；远端 `tmp/split_extracted_info.py` | 远端只读脚本 | 总 `extracted_info.pkl` → 每个 `SNGS-*.pkl` | 不是原视频切帧步骤 |
| Step 1 detector | `scripts/run_local_takeover_step1.py`；`archive/reproduction_20260819/configs/local_takeover/` | `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/detect_multiple/yolov8_person_api.py` | 图片 → `bbox_ltwh`、置信度 | 本地源码/权重固定255帧已运行；语义质量未知 |
| Step 1 ReID | 同上 | 本地 vendor 的 `sn_gamestate.reid.prtreid_api.PRTReId` | 人物框 → embeddings、role 初始分数 | 本地运行已通过；语义质量未知 |
| Step 1 tracking | 同上 | 本地 vendor 的 `tracklab.wrappers.BPBReIDStrongSORT` | 检测/ReID → `track_id` | 本地运行已通过；ID Switch 未解决 |
| pitch/keypoints | `scripts/run_local_takeover_enrichment.py`；`archive/reproduction_20260819/configs/local_takeover/` | 本地 vendor 的 `NBJW_Calib_Keypoints` | 图片 → `keypoints`、`lines` | 本地字段结构已通 |
| camera calibration | 同上 | 本地 vendor 的 `NBJW_Calib_Decouped`、`ApplyParameters` | keypoints/lines/框 → `parameters`、`bbox_pitch` | 坐标结构已通，绝对质量未知 |
| legibility | 同上 | 本地 vendor 的 `Legibility` | 人物 crop → 可读性过滤 | 本地运行已通过；阈值为0.5 |
| jersey OCR | 同上 | 本地 vendor 的 `QWEN2_5VL_OCR_BATCH` | 清晰人物 crop → 号码候选 | 本地运行已通过；ID Switch污染风险未解决 |
| tracklet aggregation | 同上 | 本地 vendor 的 `MajorityVoteTrackletFilter2` | 逐检测 role/号码 → 轨迹 role/号码 | 本地运行已通过；同一track可能混人 |
| team clustering | 同上 | 本地 vendor 的 `TrackletTeamClustering` | embedding/role → `team_cluster` | 本地运行已通过；当前质量不可靠 |
| team side | 同上 | 本地 vendor 的 `TrackletTeamSideLabeling` | cluster/`bbox_pitch` → `team` | 本地结构已通，语义质量未知 |
| Refiner | `scripts/run_local_takeover_refiner.py`；`archive/reproduction_20260819/configs/local_takeover/` | `baseline/code/soccerfactory/refiner` | `bbox_pitch` + track序列 → refined `bbox_pitch` | 本地forward已运行；时间连续性出现风险 |
| Step 3 后处理 | `scripts/run_local_takeover_step3.py`、`archive/experiments_20260819/soccerfactory_step3/` | 本地 vendor 的 `sn_gamestate` 模块由项目脚本调用 | remove outside、聚合、号码合并、team/team_side | 本地候选链已运行；no-ReID仅为诊断候选 |
| PKL 转换 | 历史入口`archive/reproduction_20260819/gates/g10_current_step3_convert.py`；新接口`research/src/soccermaster/integrations/soccerfactory/step3_to_training.py` | 项目自有 adapter | Step 3状态 → `SNGS-10004.pkl` | 当前本地样本255帧、3,390人物结构已通 |
| SoccerMaster 消费 | `archive/reproduction_20260819/gates/g10_current_pipeline_dataloader_smoke.py` | 本地 `research/src/soccermaster/data/soccernet_gsr_detection.py` | PKL + 图片 → `[1,30,3,512,512]` | G10-C smoke 通过 |

## 当前可视化

| 内容 | 位置 | 状态 |
|---|---|---|
| Step 1 中间结果 | `archive/experiments_20260819/soccerfactory_visualization/render_step1_intermediates.py` | 可生成检测/轨迹等图 |
| 完成样本/阶段对照 | `render_completed_example.py`、`render_stage_separated_example.py` | 可生成阶段图 |
| 球队聚类和颜色回放 | `compare_color_team_assignment.py`、`diagnose_team_clustering.py` | 已有比较图和冲突图 |
| Refiner 前后坐标 | `audit_refiner_coordinate_effect.py`、`diagnose_refiner_clip_boundaries.py`、单场图`visuals/07_factory_refiner_before_after.png` | 已有统计、轨迹和直观前后图；没有二维真值 |
| Step 3 最终球场图 | `run_refiner_preserving_step3.py`、`run_step3_no_reid_ablation.py` | 已有 pitch 图和阶段变化图 |
| 固定单场总入口 | `runs/reports_legacy_20260819/one_match/20260819_sngs10004_end_to_end/summary.md` | 8张图串联原视频、各阶段和DataLoader |

## 质量问题的依赖顺序

```text
ID Switch / 轨迹切分
  → 角色（player / goalkeeper / referee）
  → 球队
  → 球衣号码
  → 球场坐标 / Refiner
  → PKL 与训练
```

因此不能先用最终 `team` 或号码投票来证明轨迹身份正确。当前已确认的是接口和字段链路，不是人物、球队或坐标真值。

## 原视频切片的编号核对

候选脚本的关键规则是：先从 `Labels-cameras.json` 取 `Main camera center` 且为 `real-time` 的区间，首尾各去约 10 帧，再按最多 750 帧切段；保存时用 `SNGS-{clip_id+10000:05d}`，帧名从 `000001.jpg` 重新编号。

固定样本的编号不能直接把 `SNGS-10004` 当成 mapping 的 `clip_id=10004`：

| 来源 | 记录 | 结果 |
|---|---|---|
| `tmp/sn_2_clip.json` + `segment_soccernet.py` 命名规则 | `clip_id=4`，Chelsea–Burnley，`2278–2532` | 得到 `SNGS-10004`，共 255 帧 |
| 当前 G10 manifest/旧报告 | `clip_id=10004`，Southampton–Liverpool，`14439–14645` | 只有 207 个包含首尾的索引 |
| 只读首帧像素核对 | 准备图片 vs 上面两个原视频候选 | 平均绝对差约 `0.85` vs `37.84`，支持第一条 |

因此当前准备图片来自mapping的第4个片段，而不是旧manifest误写的第10004条映射。隔离CPU单片段已经只写工作区并完成255帧重建与对应检查；可以说固定单片段从原视频到训练消费端接口已通。旧manifest的历史记录不覆盖，历史全量批处理入口也仍未恢复为安全本地入口。

## 版本与资产来源

- TrackLab、sn-gamestate、Refiner 的 revision、源码和 checkpoint 由 `archive/reproduction_20260819/manifests/g10_*` 固定。
- `baseline/code/soccerfactory/`已保存TrackLab、sn-gamestate和Refiner源码闭包及来源记录；当前正式G10运行证据仍来自复制前的远端import路径。
- 原始图片和模型权重继续只读；下一步先验证本地vendor import并修改配置搜索路径，再把获批的固定数据和权重放入`assets/`，最后用同一固定样本重跑结构smoke。
