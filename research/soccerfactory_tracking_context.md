# SoccerFactory tracking 一致性实验上下文调查

调查日期：2026-08-20

范围：只读检查代码、配置、已有运行产物和数据目录；没有修改 pipeline、阈值或模型，没有运行推理、评估或训练。

## 1. Pipeline 总览

### 1.1 先澄清实际执行顺序

用户给出的功能顺序是：

```text
人物检测 → 原始 tracking → refiner → 球队 → role → jersey → 二维坐标 → 导出
```

但当前固定样本 SNGS-10004 的实际顺序并不是这样：

```text
图片
  → YOLO 人物检测
  → PRTReID（逐检测 ReID 特征，同时给出逐检测 role）
  → StrongSORT 原始 tracking
  → 球场点/线检测
  → 相机标定
  → bbox 投影为二维球场坐标
  → 号码可读性
  → Qwen OCR
  → 按原始 track_id 聚合 role 和号码
  → 按轨迹 ReID 均值做两队 KMeans
  → 用球场平均位置把两个匿名簇命名为 left/right
  → 神经 Refiner（当前配置只修正二维坐标）
  → 当前 Step 3（移除场外、再次聚合、号码拼接、再次分队；不含 ReID 拼接）
  → SoccerFactory .pklz
  → SoccerMaster 训练 PKL 适配
```

入口由 TrackLab 的 `tracklab.main.main()` 读取 Hydra 配置、实例化 `Pipeline` 和 `OfflineTrackingEngine`：

- `baseline/code/soccerfactory/tracklab/tracklab/main.py:23-45`
- `baseline/code/soccerfactory/tracklab/tracklab/pipeline/module.py:65-85`
- `baseline/code/soccerfactory/tracklab/tracklab/engine/offline.py`

当前固定链路入口与配置：

- Step 1 launcher：`research/reproduction/smokes/soccerfactory/step1.py`
- Step 1 config：`research/reproduction/smokes/soccerfactory/configs/step1.yaml`
- enrichment launcher：`research/reproduction/smokes/soccerfactory/enrichment.py`
- enrichment config：`research/reproduction/smokes/soccerfactory/configs/enrichment.yaml`
- Refiner launcher：`research/reproduction/smokes/soccerfactory/refiner.py`
- Refiner config：`research/reproduction/smokes/soccerfactory/configs/refiner.json`
- 当前 Step 3：`research/reproduction/smokes/soccerfactory/step3.py`
- SoccerMaster 转换：`research/reproduction/smokes/soccerfactory/convert.py`

历史完整 Step 3 与当前固定 Step 3 不同。历史配置会依次运行号码拼接和 ReID 拼接；当前固定 Step 3 为了保留可检查的 1:1 轨迹关系，删除了 ReID 拼接：

- 历史配置：`baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/gsr_step_3_sn500_1000.yaml:35-51`
- 历史实验实现：`archive/experiments_20260819/soccerfactory_step3/run_refiner_preserving_step3.py:199-211`
- 当前实现：`research/reproduction/smokes/soccerfactory/step3.py:53-64`

### 1.2 各阶段输入、输出和连接字段

| 阶段 | 主要入口、类和函数 | 输入 | 输出 | 上下游连接字段 | checkpoint / 外部模型 |
|---|---|---|---|---|---|
| 人物检测 | `detect_multiple/yolov8_person_api.py`：`YOLOv8_Person.preprocess/process` | 一帧 RGB 图片和 image metadata | 每个人的 `bbox_ltwh`、`bbox_conf`、`image_id`、`video_id`、`category_id` | `image_id` 把检测连回帧；DataFrame 行索引是检测行身份 | `yolo_v8x6_person_lr_default_best.pt`；代码只保留 YOLO class 0 且 `bbox_conf >= 0.4`，见 `yolov8_person_api.py:52-74` |
| ReID 与当前逐帧 role | `reid/prtreid_api.py`：`PRTReId.preprocess/process` | `bbox_ltwh` 对应的人物 crop | 256 维全局 `embeddings`、`visibility_scores`、`body_masks`、`role_detection`、`role_confidence` | 检测行索引与 `bbox_ltwh` | `prtreid-soccernet-baseline.pth.tar`；HRNet32 backbone 使用 `hrnetv2_w32_imagenet_pretrained.pth`。role 是 PRTReID 同一 forward 的分类输出，不是当前链路中的 Qwen |
| 原始 tracking | `tracklab/wrappers/track/bpbreid_strong_sort_api.py`：`BPBReIDStrongSORT.prepare_next_frame/process` | 每帧的 `bbox_ltwh`、`bbox_conf`、`embeddings`、`visibility_scores` | `track_id`、Kalman bbox、匹配方式、cost、hits、age、state 等 | 检测行索引、逐帧 `image_id`、`track_id` | StrongSORT + Kalman，无独立 checkpoint；外观来自 PRTReID |
| 球场点/线 | `calibration/nbjw_calib.py`：`NBJW_Calib_Keypoints.process` | 完整帧 | image-level `keypoints`、`lines` | `image_id` / image DataFrame 索引 | `SV_kp`、`SV_lines`；heatmap 阈值分别为 0.1449、0.2983，见 `nbjw_calib.py:116-136` |
| 相机标定 | `calibration/nbjw_calib.py`：`NBJW_Calib_Decouped.process` | image-level `keypoints` | `parameters` 和 `h` | `image_id` | 几何求解 `FramebyFrameCalib`，RANSAC 参数 50；无额外 checkpoint |
| 二维球场坐标 | `calibration/nbjw_calib.py`：`ApplyParameters.process` | 每帧 `parameters/h` 和每个检测的 `bbox_ltwh` | 每个检测的 `bbox_pitch` 字典 | `image_id`、检测行索引 | 无模型；当前 `use_h=false`、`use_linalg=false`，用相机 K/R/T 投影；标定失败时允许沿用上一帧参数 |
| 号码可读性 | `legibility/legibility_api.py`：`LegibilityClassifier34`、`Legibility.process` | 人物 crop | 逐检测 `legibility_score` | 检测行索引 | `legibility_resnet34_soccer_20240215.pth` |
| jersey number | `jersey/qwen2_5vl_ocr_api.py`：`QWEN2_5VL_OCR_BATCH.process` | 人物 crop；当前还输入 `legibility_score` | `jersey_number_detection`、二值 `jersey_number_confidence`，当前还保存完整文本 | 检测行索引；之后由 `track_id` 聚合 | Qwen2.5-VL-7B-Instruct；当前只处理 `legibility_score >= 0.5` 的 crop |
| role 聚合 | `tracklet_agg/majority_vote_filter_api.py`：`MajorityVoteTrackletFilter2.process` | `role_detection/confidence` 和 `jersey_number_detection/confidence` | 整条轨迹统一的 `role`、`jersey_number` | `track_id` | 无模型；role 直接加权投票，号码还要求局部三项窗口中至少两项相同 |
| 历史 Qwen role 候选 | `role/qwen2_5vl_role_api.py`：`QWEN2_5VL_ROLE_BATCH.process` | 完整帧和人物 crop | `role_detection`、二值 `role_confidence` | 检测行索引；再按 `track_id` 聚合 | Qwen2.5-VL-7B/72B，取决于配置。它存在于历史 Step 3，但不在当前固定 enrichment 中 |
| 球队匿名聚类 | `team/tracklet_team_clustering_api.py`：`TrackletTeamClustering.process` | `track_id`、逐检测 PRTReID `embeddings`、聚合后的 `role` | `team_cluster` 0/1 | `track_id` | sklearn KMeans，无 checkpoint |
| 球队 left/right 命名 | `team/tracklet_team_side_labeling_api.py`：`TrackletTeamSideLabeling.process` | `team_cluster`、`bbox_pitch`、`role` | `team` 为 left/right | `track_id`、`bbox_pitch` | 无模型 |
| 神经 Refiner | `refiner/inference.py`：`inference_on_video`、`update_predictions`；`refiner/model/timesformer.py`：`SoccerTrackerTransformerTimeSformer` | 逐帧 ReID、bbox、初始坐标、role、team、jersey、track_id 和可见 mask | 当前只覆盖 `bbox_pitch.x/y_bottom_middle` | 检测行索引通过 `detection_mappings` 回写 | `assets/checkpoints/official/soccerfactory/refiner/best_model.pth` |
| 轨迹号码拼接 | `concat_tracklets_by_jn/concat_tracklets_by_jn_api.py`：`ConcatTrackletsByJN.process` | `track_id`、轨迹级 `jersey_number`、`image_id` | 重写后的 `track_id` | `track_id` | 无模型 |
| 轨迹 ReID 拼接（历史） | `concat_tracklets_by_reid/concat_tracklets_by_reid_api.py`：`ConcatTrackletsByReid.process` | `track_id`、逐检测 `embeddings`、`image_id` | 重写后的 `track_id` | `track_id` | 无新 checkpoint；复用 PRTReID embeddings |
| SoccerFactory 状态导出 | TrackLab state saver；本地检查入口见 `step1.py:102-126` | detection DataFrame、image DataFrame | `.pklz`，内部为 detection PKL、image PKL、通常还有 `summary.json` | detection `image_id` 对 image `id` | 无模型 |
| SoccerMaster 适配 | `research/src/soccermaster/integrations/soccerfactory/step3_to_training.py`：`convert_step3_to_training_frames` | Step 3 detection/image DataFrame | 按帧字典：`people`、`K/R/P`、`valid_cam_params` | `image_id` 对 image `id`；frame 从 0-based 转为 1-based | 无模型 |

重要接口损失：SoccerMaster 训练 PKL 中的人物只保留 `id`（检测行索引）、`bbox_ltwh`、`role`、`legibility_score`、`jersey_number`。`track_id`、`team` 和 `bbox_pitch` 没有被适配器带入，见 `step3_to_training.py:47-92` 和 `convert.py:136`。

## 2. Refiner 逻辑

### 2.1 当前神经 Refiner 不合并、也不拆分轨迹

这是由实际代码和固定输出共同确认的，不是根据名字推断：

1. 当前配置把 `track`、`role`、`team`、`jersey`、`missing` 全设为 false，只启用 `coord`：`research/reproduction/smokes/soccerfactory/configs/refiner.json:18-27`。
2. `update_predictions()` 只会根据任务开关回写 role、team、jersey 和坐标；函数中没有任何重写 `track_id` 的分支：`baseline/code/soccerfactory/refiner/inference.py:167-256`。
3. 模型在 `track=true` 时可以产生 `track_affinity`，但 `inference_on_video()` 只是收集 affinity，`update_predictions()` 没有消费它来生成新 ID：`inference.py:138-164`、`timesformer.py:283-303`。
4. 当前 SNGS-10004 输出在 Refiner 前后均为 3,390 行、48 条轨迹，逐行 `track_id` 相同；运行检查也明确断言 `track_ids_preserved=true`。

因此，当前神经 Refiner 的回答是：**既不合并，也不拆分**。即使以后把 track head 打开，现有 inference 代码也只输出 pairwise affinity，不会自动完成轨迹关联。

### 2.2 当前 Refiner 实际使用的信息

`SoccerTrackerTransformerTimeSformer.forward()` 将以下输入投影后拼接：

- PRTReID 特征；
- 初始二维坐标；
- role；
- team；
- jersey；
- 原始 track ID embedding；
- 时间位置 embedding 和空间位置 embedding。

代码：`baseline/code/soccerfactory/refiner/model/timesformer.py:161-197`。

当前坐标头采用 residual 模式：在归一化的输入坐标上加预测残差，再反归一化到球场坐标，见 `timesformer.py:243-278`。它只替换 `bbox_pitch` 的 bottom-middle 两个值，bottom-left/right 保持原值，见 `inference.py:237-254`。

### 2.3 当前 Refiner 的关键参数

| 参数 | 值 | 位置 | 作用 |
|---|---:|---|---|
| `d_model` | 512 | `refiner/configs/base.yaml:9` | Transformer 隐层宽度 |
| `nhead` | 8 | `base.yaml:10` | attention heads |
| `num_layers` | 6 | `base.yaml:11` | TimeSformer blocks 数 |
| `coord_regression_type` | residual | `base.yaml:12` | 预测初始坐标残差 |
| `max_clip_frames` | 100 | 当前 `refiner.json:14` | 255 帧被独立切成 0–99、100–199、200–254；没有重叠融合 |
| `max_detections_per_frame` | 30 | `base.yaml:33` | 每帧张量容量 |
| `max_affinity_frame_distance` | 25 | `base.yaml:16` | 只限制可选 track affinity 的帧对范围；当前 track head 关闭 |
| `track` task | false | 当前 `refiner.json:21` | 当前不产生 track affinity |

当前 Refiner **没有轨迹合并阈值**。`max_affinity_frame_distance=25` 不是合并阈值，而且当前未启用。

### 2.4 真正改动 track_id 的模块

此前被口头称为“Refiner 错误合并”的现象实际来自 Refiner 后面的 Step 3 拼接模块。

#### ReID 拼接

代码：`baseline/code/soccerfactory/sn-gamestate/sn_gamestate/concat_tracklets_by_reid/concat_tracklets_by_reid_api.py:19-90`。

合并条件：

- 先对每条轨迹的所有逐帧 `embeddings` 求均值；
- 两条当前轨迹的 cosine distance 必须 `<= threshold`；
- 两条轨迹占用的 `image_id` 集合必须完全不相交；
- 满足后将第二条轨迹的 ID 改成第一条轨迹的 ID；
- 合并后的 embedding 按两条轨迹各自占用的图像数加权平均，然后继续迭代。

历史固定实验使用 `threshold=0.1`，位置在：

- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/gsr_step_3_sn500_1000.yaml:79`
- `archive/experiments_20260819/soccerfactory_step3/run_refiner_preserving_step3.py:205`

它使用外观/ReID 和“是否同帧出现”，但**不使用**：

- 最大时间间隔；
- 二维或图像位置连续性；
- 运动速度；
- team；
- role；
- jersey number。

迭代合并允许传递链：A 接近 B、B 接近 C 时，最终组内 A 与 C 可以超过 0.1。已有 9 个合并组中，new track 3、5、12、22 都出现最终组内 pair distance 超过 0.1 的情况，证据在 `runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/merge_review_summary.json`。

#### 号码拼接

代码：`baseline/code/soccerfactory/sn-gamestate/sn_gamestate/concat_tracklets_by_jn/concat_tracklets_by_jn_api.py:23-91`。

它对每条轨迹取最常见的非空号码。如果同一号码对应的多条轨迹两两没有任何 `image_id` 重叠，就给它们同一个新 ID。它没有 ReID、时间间隔、位置、球队或置信度门槛。无论是否真正合并，它都会把所有轨迹重新编号，所以“ID 数值变化”本身不代表轨迹合并。

### 2.5 ID 变化和来源映射是否保存

- 神经 Refiner：`track_id` 不变，没有 `source_track_id` 字段。
- 标准 `ConcatTrackletsByJN` / `ConcatTrackletsByReid`：直接覆盖 `track_id`，没有输出来源映射。
- 2026-08-19 的诊断 runner 临时增加了 `__source_row_id` 和 `__original_track_id`，并把合并组写入 `result.json`；但正式 `.pklz` 保存前显式删除这两列：`run_refiner_preserving_step3.py:74-92`、`126-133`。

结论：已有诊断报告能恢复那一次实验的合并来源，但标准最终 SoccerFactory 标注中不能恢复 `new track_id ← source track_ids`。

## 3. 球队识别逻辑

### 3.1 当前球队是整条轨迹判断，不是逐帧独立判断

`TrackletTeamClustering.process()` 只选择 `role == "player"` 的检测，按 `track_id` 对逐帧 embedding 求均值，然后对“每条轨迹一个 embedding”做 KMeans：`tracklet_team_clustering_api.py:25-53`。

具体行为：

- ReID 特征：由 PRTReID 对每个检测 crop 提取，当前使用 256 维 global embedding；
- KMeans 输入：形状为 `[player_track_count, 256]` 的轨迹均值 embedding；
- 聚类数：有至少两条轨迹时固定 `n_clusters=2`；只有一条时直接标为 0；
- seed：`random_state=0`；
- 后处理：通过 `track_id` merge，把同一个 `team_cluster` 写回该轨迹的每一行；
- 置信度：没有保存到 detection DataFrame。KMeans distance、margin 或概率都没有输出。

匿名簇 0/1 由 `TrackletTeamSideLabeling` 命名为 left/right：比较两个簇所有检测的 `bbox_pitch.x_bottom_middle` 平均值，均值较大的簇为 right，见 `tracklet_team_side_labeling_api.py:29-42`。

### 3.2 门将和裁判

- 门将不进入 player KMeans。`TrackletTeamSideLabeling` 对每个门将检测单独看当前 `x_bottom_middle`：`x > 0` 为 right，否则 left。因此门将的 `team` 理论上可能在同一轨迹内逐帧变化。
- 裁判不进入 KMeans，也没有专门球队赋值，应保持 team 缺失。
- 当前 SNGS-10004 历史 enrichment 的聚合 role 全部变成 `player`，所以门将和裁判的上述特殊逻辑实际上没有可靠触发；它们会被当作普通球员参与 KMeans。

### 3.3 role 与置信度的实际含义

当前链路中的逐检测 role 来自 PRTReID 的 role head，再按 `track_id` 投票成整轨迹 `role`。保存的 `role_confidence` 是 PRTReID 最大分类 score；真实样例中可大于 1（如 3.73），因此不能直接解释成校准概率。

历史 Qwen role 会对“完整帧 + crop”输出一个词，合法词就把 confidence 设为 1，否则为 0；它也不是概率校准值。当前固定 enrichment 没有调用该 Qwen role 模块。

### 3.4 是否保存逐帧、轨迹级结果和置信度

| 内容 | 是否保存 | 说明 |
|---|---|---|
| 逐检测 ReID embedding | 是 | `embeddings`，随后用于 tracking 和 team KMeans |
| 逐检测 role | 是 | `role_detection`、`role_confidence` |
| 轨迹级 role | 是 | `role` 被重复写到该轨迹每一行；没有单独的聚合置信度 |
| 逐检测 OCR | 是 | `jersey_number_detection`、二值 confidence；当前还保存完整文本 |
| 轨迹级号码 | 是 | `jersey_number` 被重复写到该轨迹每一行；没有单独聚合置信度 |
| 逐检测 team 预测 | 否 | 没有逐帧颜色/队伍分类器输出 |
| 轨迹级匿名球队 | 是 | `team_cluster` 重复写入各行 |
| 轨迹级球队置信度 | 否 | 未保存 KMeans distance/margin |
| left/right team | 是 | 普通 player 通常整轨迹一致；门将按逐检测位置决定 |

### 3.5 球衣颜色实现和实验

仓库中已有一个**实验候选**，但没有接入正式 pipeline：

- 比较脚本：`archive/experiments_20260819/soccerfactory_step3/compare_color_team_assignment.py`
- 特征提取来源：该脚本调用历史 `replay_team_assignment_candidate.extract()`，对多帧人物 crop 提取上身颜色特征，再对轨迹颜色特征做 `KMeans(n_clusters=2, random_state=0)`。
- 结果：`runs/reports_legacy_20260819/g10/20260819_step3_color_team_replay/result.json`
- 人工目测记录：`runs/reports_legacy_20260819/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json`

在 39 条人工目测为外场球员且有队伍标签的轨迹上，ReID KMeans 为 32/39，颜色候选为 38/39。这个结果只来自一个固定视频；颜色结果没有写回 `.pklz`，也没有可靠 role gate 排除门将和裁判，因此不能写成正式球队模块已经替换。

## 4. 数据格式

### 4.1 同一个真实视频和三个文件

本节统一使用 `video_id=10004`、255 帧的 SNGS-10004 当前本地运行：

1. Refiner 前：`.runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz`
2. Refiner 后：`.runtime/local_takeover/g10/sngs10004_refiner/output/refined_sn-gamestate.pklz`
3. 最终 SoccerFactory Step 3：`.runtime/local_takeover/g10/sngs10004_step3/states/sn-gamestate.pklz`

三个阶段分别为 3,390 行 / 48 tracks、3,390 行 / 48 tracks、3,390 行 / 48 tracks。

`.pklz` 内 detection 表与 image 表分开：

- detection：`10004.pkl`
- image：`10004_image.pkl`
- Step 1/enrichment/Step 3 另有 `summary.json`；当前 Refiner 输出没有 summary。

`frame_id` 不在 detection 行中。必须用 detection.`image_id` 连接 image.`id`，再读取 image.`frame`；这里的 frame 是 0-based。

### 4.2 Refiner 前的真实记录

路径：`.runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz`

```text
frame=0, image_id=610004000001, track_id=1,
bbox_ltwh=[1524.418,445.318,50.855,94.542], bbox_conf=0.89437,
team_cluster=1, team=right, role=player,
role_detection=player, role_confidence=3.72779,
jersey_number_detection=4, jersey_number_confidence=1.0, jersey_number=4,
pitch_xy=(-26.30196,-10.23131)

frame=0, image_id=610004000001, track_id=2,
bbox_ltwh=[1349.163,395.019,41.068,72.969], bbox_conf=0.85527,
team_cluster=0, team=left, role=player,
role_detection=player, role_confidence=3.61147,
jersey_number_detection=null, jersey_number_confidence=0.0, jersey_number=21,
pitch_xy=(-33.37812,-17.52633)
```

第二条说明逐帧 OCR 可以是 null，但轨迹投票后的 `jersey_number` 仍可由该轨迹其他帧得到 21。

### 4.3 Refiner 后的真实记录

路径：`.runtime/local_takeover/g10/sngs10004_refiner/output/refined_sn-gamestate.pklz`

```text
frame=0, image_id=610004000001, track_id=1,
bbox_ltwh=[1524.418,445.318,50.855,94.542], bbox_conf=0.89437,
team_cluster=1, team=right, role=player, jersey_number=4,
pitch_xy=(-26.59469,-9.90790)

frame=0, image_id=610004000001, track_id=2,
bbox_ltwh=[1349.163,395.019,41.068,72.969], bbox_conf=0.85527,
team_cluster=0, team=left, role=player, jersey_number=21,
pitch_xy=(-33.57553,-17.13497)
```

相对于 Refiner 前，同一行的 track、bbox、team、role、jersey 不变，仅 bottom-middle pitch coordinate 改变。

### 4.4 最终 SoccerFactory Step 3 的真实记录

路径：`.runtime/local_takeover/g10/sngs10004_step3/states/sn-gamestate.pklz`

```text
frame=0, image_id=610004000001, track_id=1,
bbox_ltwh=[1524.418,445.318,50.855,94.542], bbox_conf=0.89437,
team_cluster=0, team=left, role=player, jersey_number=4,
pitch_xy=(-26.59469,-9.90790)

frame=0, image_id=610004000001, track_id=2,
bbox_ltwh=[1349.163,395.019,41.068,72.969], bbox_conf=0.85527,
team_cluster=0, team=left, role=player, jersey_number=21,
pitch_xy=(-33.57553,-17.13497)
```

track 1 的 team 从 Refiner 前的 right 变成最终的 left，但它的检测行和 track ID 没变。这是 Step 3 重新 KMeans/命名的结果，不能直接解释为 ID switch。

### 4.5 要求字段的存在情况

| 字段 | Refiner 前 | Refiner 后 | 最终 Step 3 | 备注 |
|---|---|---|---|---|
| `frame_id` | 间接存在 | 间接存在 | 间接存在 | detection.`image_id` → image.`id` → image.`frame` |
| `track_id` | 是 | 是且不变 | 是，号码拼接可能重编号 | 当前固定 Step 3 没有 ReID merge |
| `bbox` | `bbox_ltwh` | `bbox_ltwh` | `bbox_ltwh` | 同时保留 Kalman bbox 字段 |
| `team` | 是 | 是 | 是 | 没有 team confidence |
| `role` | 是 | 是 | 是 | 另有逐检测 role 和 score |
| `jersey_number` | 是 | 是 | 是 | 另有逐检测 OCR 和二值 confidence |
| `pitch_position` | `bbox_pitch` | `bbox_pitch` | `bbox_pitch` | 字典含 bottom-left/middle/right 共 6 个数 |
| `confidence` | 部分 | 部分 | 部分 | 有 bbox/role/jersey/legibility；没有统一 confidence、team confidence 或聚合置信度 |
| `source_track_id` | 否 | 否 | 否 | 标准输出不保留来源轨迹 |

## 5. 可用真值和评估代码

### 5.1 真正的人工真值

服务器上存在官方 SoccerNet Game State Recognition 标注：

- 根目录：`/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS`
- 代码仓中的原路径是只读链接：`/remote-home/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024`
- 找到 `Labels-GameState.json`：train 57 个视频、valid 59 个、test 49 个；challenge 0 个。
- 真实例子：`/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/valid/SNGS-095/Labels-GameState.json`

这些 JSON 的 object annotations 包含逐帧 `track_id`、`bbox_image`、`bbox_pitch`、`attributes.role`、`attributes.team`、`attributes.jersey`。加载实现见：

- `baseline/code/soccerfactory/tracklab/tracklab/wrappers/datasets/soccernet/soccernet_game_state.py:189-212`
- `soccernet_game_state.py:247-309`

固定开发视频 SNGS-10004 位于 `sn500`，只有 255 张 JPEG，**没有** `Labels-GameState.json`，因此不能在这个视频上计算官方 ID switch、HOTA 或坐标误差真值。

### 5.2 模型生成标签

以下是模型/规则生成标签，不是真值：

- `.runtime/local_takeover/g10/sngs10004_step1/states/sn-gamestate.pklz`
- `.runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz`
- `.runtime/local_takeover/g10/sngs10004_refiner/output/refined_sn-gamestate.pklz`
- `.runtime/local_takeover/g10/sngs10004_step3/states/sn-gamestate.pklz`
- `.runtime/local_takeover/g10/sngs10004_training_pkl/SNGS-10004.pkl`

其中 SoccerMaster 训练 PKL 还是由 SoccerFactory 伪标签转换而来，不能反过来充当 SoccerFactory 的评估真值。

### 5.3 人工目测记录

39 条球队记录来自对单场 SNGS-10004 轨迹 crop 的人工目测：

- `runs/reports_legacy_20260819/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json`
- 评估汇总：`runs/reports_legacy_20260819/g10/20260818_team_color_diagnostic/annotation_evaluation.json`

它们可作为球队方法开发集，但不是官方逐帧 tracking 真值：没有逐帧人物身份边界、ID switch 时刻或 bbox GT；部分轨迹明确标为 uncertain/unknown，track 20 还有“画面重叠、较难判断”的备注。

### 5.4 可用评估代码

1. `TrackEvalEvaluator`：`baseline/code/soccerfactory/tracklab/tracklab/wrappers/eval/trackeval_evaluator.py:13-105`。它把预测保存为 TrackEval 格式，然后可运行 CLEAR、HOTA、Identity。
2. SoccerNetGS 数据集配置：`baseline/code/soccerfactory/tracklab/tracklab/configs/dataset/soccernet_gs.yaml`。
3. GS-HOTA 配置：`baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/eval/gs_hota.yaml`。
4. 当前本地环境中的 SoccerNetGS TrackEval 实现：`.envs/SoccerMaster-repro/lib/python3.10/site-packages/trackeval/datasets/soccernet_gs.py`。它可在 image space 用 bbox IoU，或在 pitch space 用 bottom-middle 的 Gaussian similarity；启用 role/team/jersey 时，任一属性不匹配会把 pair similarity 置零，见该文件 `:455-489`。
5. `tracklab/wrappers/eval/soccernet/soccer_accuracy.py` 当前只是空壳，不能提供独立球队/role/jersey 指标。

历史 Step 3 配置列出了 `YOLOmAP`、`AttributesACC`、`PitchDistance`、`RolesACC`，但当前安装的 `trackeval.metrics` 没有暴露这些同名 metric class；`TrackEvalEvaluator` 对未知名字只会 warning 并跳过。因此，不能仅根据 YAML 中出现名字就认为这些独立指标已经可运行。

### 5.5 已有 Refiner 前后指标

没有找到“同一官方 GT 视频上，神经 Refiner 前后”的 HOTA、Identity、ID switch 或绝对坐标误差比较。

已有的是无真值的单视频平滑性诊断：

- `runs/reports_legacy_20260819/g10/20260819_refiner_coordinate_quality/result.json`
- `runs/reports_legacy_20260819/g10/20260819_refiner_coordinate_quality/README.md`

它确认 3,176 行和 49 个 ID 保持不变，但连续帧二阶差分中位数由 0.213 m 增到 0.401 m。该结果只能说明输出没有更平滑，不能说明绝对坐标更准或更差。

## 6. 典型失败案例

本节的 `frame` 均为 image table 的 0-based frame。SNGS-10004 没有官方 identity/coordinate GT，所以除代码结构和逐行字段变化外，视觉身份结论均标为“疑似”或“强视觉证据”，不写成正式真值。

### 6.1 原始 tracker 自身疑似 ID switch：2 个

| video_id | 原始 track_id | frame range | Refiner 前后 ID | 证据与路径 |
|---|---:|---|---|---|
| 10004 | 3 | frame 0 为酒红球员；16–17、204、207 为蓝队球员 | 3 → 3，不变 | 强视觉证据，且中间有长 gap。数据：`.runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz`；已有 contact sheet 也展示 old ID 3 的 frame 0/17/207：`runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/reid_merge_groups/new_track_03.png` |
| 10004 | 28 | frame 91–92 为酒红球员；139–190、219–254 为蓝队球员 | 28 → 28，不变 | 强视觉证据，92→139 有 46 帧缺口。数据同上；人工球队记录把整轨迹记为 blue，因此该记录本身不能标出早期 switch |

这两个案例都说明长时间消失后重新使用旧 ID 是主要风险。球队标签变化可以提供线索，但这里的判断还额外检查了人物 crop；不能把 team 字段变化单独当作 ID switch。

### 6.2 “Refiner 疑似错误合并”：无法找到，因为当前 Refiner 不改 ID

当前神经 Refiner 不存在合并轨迹的代码路径，因此无法给出两个“神经 Refiner 错误合并”案例。此前观察到的两个典型错误实际发生在历史 Step 3 的 `ConcatTrackletsByReid`：

| video_id | Step 3 新 ID | 原始 IDs 与 frame range | Refiner 前后 ID | 证据与路径 |
|---|---:|---|---|---|
| 10004 | 13 | old 13：0–135；old 31：96–254 | 神经 Refiner 保持 13/31；Step 3 后都变 13 | 疑似错误合并，crop 中可见不同号码（6 与 14），cosine distance 0.04046。`runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/reid_merge_groups/new_track_13.png` |
| 10004 | 16 | old 16：16–127；old 43：171–254 | 神经 Refiner 保持 16/43；Step 3 后都变 16 | 疑似错误合并，old 43 清楚为 14，old 16 外观/号码不同；distance 0.09527，仍低于 0.1。`runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/reid_merge_groups/new_track_16.png` |

来源映射和距离：`runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/merge_review_summary.json`。

### 6.3 同一行谱系/同一轨迹的球队标签在两个阶段变化：2 个

以下案例证明“球队模块输出不稳定”，不证明人物身份真值：

| video_id | pre track → final track | frame range | team 变化 | 路径 |
|---|---|---|---|---|
| 10004 | 1 → 1 | 0–180 | enrichment `right` → final Step 3 `left` | `.runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz`；`.runtime/local_takeover/g10/sngs10004_step3/states/sn-gamestate.pklz` |
| 10004 | 7 → 7 | 0–254 | enrichment `right` → final Step 3 `left` | 同上 |

两个文件的 DataFrame 行索引完全对齐，且这两组 old/new ID 相同。变化来自 Step 3 重新聚类/命名，不应解释为 ID switch。

### 6.4 人物不同但球队颜色相同：2 个

最明确的现有例子就是 6.2 的两个错误拼接候选：

- new ID 13：old 13 与 old 31 都穿酒红色，但可见号码分别为 6 和 14；球队颜色无法阻止同队不同人被合并。
- new ID 16：old 16 与 old 43 都穿酒红色，old 43 为 14，另一轨迹不是同一号码/人物；同样无法靠二队颜色发现。

因此颜色适合做 team 分类，但不能替代人物级 ReID 或 identity consistency 检查。

### 6.5 二维坐标明显跳跃：2 个

| video_id | track_id | frame range | Refiner 前 | Refiner 后 | 前后 ID | 路径 |
|---|---:|---|---:|---:|---|---|
| 10004 | 32 | 121→122 | 2.757 m | 2.786 m | 32 → 32 | `.runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz`；`.runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz` |
| 10004 | 15 | 202→203 | 2.744 m | 2.591 m | 15 → 15 | 同上 |

这是同一 track 的相邻帧 bottom-middle 欧氏距离。它们是明显的时间跳跃，但在没有官方 pitch GT 和明确帧率解释时，不能断言哪一帧坐标错误。

另有两个更能显示 Refiner 新增不连续性的案例：track 16 的 126→127 从 0.537 m 增至 1.553 m；track 17 的 199→200 从 0.081 m 增至 0.901 m。后者正好跨 100 帧 clip 边界，但已有整体统计说明 clip 边界不能解释全部抖动增加。

## 7. 当前仍缺少的信息

1. **SNGS-10004 的官方 track ID 和 pitch coordinate GT**：当前不存在，不能把该视频上的目测案例转成正式指标。
2. **同一官方 GT 视频的完整缓存**：服务器有官方 train/valid/test 标签，但尚未找到同一视频对应的 raw tracker、Refiner 后和每个 Step 3 子阶段缓存。
3. **正式输出中的轨迹 lineage**：标准 Step 3 丢弃 source track IDs，后续很难判断一个新 ID 由哪些旧 ID 组成。
4. **轨迹拆分实现**：当前神经 Refiner inference、号码拼接和 ReID 拼接都没有真正的 split 操作。
5. **track affinity 到最终 ID 的解码器和阈值**：模型可选输出 affinity，但仓库当前 inference 没有把它变成轨迹。
6. **球队置信度**：正式 KMeans 没有保存距离、margin 或 unknown；颜色实验的 margin 只在诊断 JSON 中。
7. **逐帧人工 identity 审核**：39 条人工记录只给整轨迹 team/role，无法评价 ID switch 时刻。
8. **独立属性评估实现**：当前可用的是 GS-HOTA 的联合惩罚；历史 YAML 中的独立 AttributesACC/RolesACC/PitchDistance 名称与当前安装代码不匹配。
9. **当前 3,390 行本地链路与历史 3,176 行诊断链路的质量等价性**：两者是同一视频但不同运行产物，失败案例不能直接当作当前输出的总体发生率。

## 8. 建议下一步应该先评估什么

第一项应做：**在一个有官方 `Labels-GameState.json` 的固定 valid 视频上，保持 detection 行不变，只比较 raw StrongSORT、号码拼接后、ReID 拼接后和两者都启用后的 identity consistency。**

最小实验表：

```text
A. raw StrongSORT
B. A + ConcatTrackletsByJN
C. A + ConcatTrackletsByReid
D. A + JN + ReID
```

每个分支必须保存 `output_track_id ← source_track_ids`，并报告 image-space HOTA 中的 association 部分、Identity/IDF1、ID switch、fragmentation、同帧 ID collision 和轨迹数。这样可以先回答“后处理到底修复了碎片，还是制造了错误合并”，而不会把坐标 Refiner、球队或检测 recall 混进同一个结论。

坐标 Refiner 应作为第二个独立实验：固定同一检测匹配和 track ID，用官方 `bbox_pitch` 比较 Refiner 前后的绝对位置误差与时间连续性。不要再用“是否更平滑”代替“是否更准确”。

## 附录 A：相关代码文件

### Pipeline 与数据

- `baseline/code/soccerfactory/tracklab/tracklab/main.py`
- `baseline/code/soccerfactory/tracklab/tracklab/pipeline/module.py`
- `baseline/code/soccerfactory/tracklab/tracklab/engine/offline.py`
- `baseline/code/soccerfactory/tracklab/tracklab/wrappers/datasets/soccernet/soccernet_game_state.py`
- `baseline/code/soccerfactory/tracklab/tracklab/datastruct/tracker_state.py`
- `research/reproduction/smokes/soccerfactory/step1.py`
- `research/reproduction/smokes/soccerfactory/enrichment.py`
- `research/reproduction/smokes/soccerfactory/refiner.py`
- `research/reproduction/smokes/soccerfactory/step3.py`
- `research/reproduction/smokes/soccerfactory/convert.py`

### 检测、ReID 和 tracking

- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/detect_multiple/yolov8_person_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/reid/prtreid_api.py`
- `baseline/code/soccerfactory/tracklab/tracklab/wrappers/track/bpbreid_strong_sort_api.py`
- `baseline/code/soccerfactory/tracklab/plugins/track/bpbreid_strong_sort/strong_sort.py`
- `baseline/code/soccerfactory/tracklab/plugins/track/bpbreid_strong_sort/sort/tracker.py`

### role、球队和号码

- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/tracklet_agg/majority_vote_filter_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/team/tracklet_team_clustering_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/team/tracklet_team_side_labeling_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/legibility/legibility_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/jersey/qwen2_5vl_ocr_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/role/qwen2_5vl_role_api.py`

### 标定、坐标与 Refiner

- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/calibration/nbjw_calib.py`
- `baseline/code/soccerfactory/refiner/inference.py`
- `baseline/code/soccerfactory/refiner/model/timesformer.py`
- `baseline/code/soccerfactory/refiner/model/utils.py`
- `baseline/code/soccerfactory/refiner/dataset/dataset_utils.py`

### 轨迹拼接、导出和评估

- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/concat_tracklets_by_jn/concat_tracklets_by_jn_api.py`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/concat_tracklets_by_reid/concat_tracklets_by_reid_api.py`
- `archive/experiments_20260819/soccerfactory_step3/run_refiner_preserving_step3.py`
- `archive/experiments_20260819/soccerfactory_step3/render_reid_merge_review.py`
- `archive/experiments_20260819/soccerfactory_step3/compare_color_team_assignment.py`
- `archive/experiments_20260819/soccerfactory_visualization/diagnose_track_switch_candidates.py`
- `research/src/soccermaster/integrations/soccerfactory/step3_to_training.py`
- `baseline/code/soccerfactory/tracklab/tracklab/wrappers/eval/trackeval_evaluator.py`
- `.envs/SoccerMaster-repro/lib/python3.10/site-packages/trackeval/datasets/soccernet_gs.py`

## 附录 B：相关配置文件

- `research/reproduction/smokes/soccerfactory/configs/step1.yaml`
- `research/reproduction/smokes/soccerfactory/configs/enrichment.yaml`
- `research/reproduction/smokes/soccerfactory/configs/refiner.json`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/gsr_step_3_sn500_1000.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/bbox_detector/yolov8_person.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/reid/prtreid.yaml`
- `baseline/code/soccerfactory/tracklab/tracklab/configs/modules/track/bpbreid_strong_sort.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/pitch/nbjw_calib.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/calibration/nbjw_calib_decouped.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/apply_camera_params/nbjw_calib_apply_params.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/legibility/legibility.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/jersey_number_detect/qwen2_5vl_ocr_batch.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/role/qwen2_5vl_role_batch.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/concat_tracklets_by_jn/concat_tracklets_by_jn.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/concat_tracklets_by_reid/concat_tracklets_by_reid.yaml`
- `baseline/code/soccerfactory/refiner/configs/base.yaml`
- `baseline/code/soccerfactory/refiner/configs/train_timesformer_100clip_coord_only_not_0init_l2_xyflip.yaml`
- `baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/eval/gs_hota.yaml`
- `baseline/code/soccerfactory/tracklab/tracklab/configs/dataset/soccernet_gs.yaml`

## 附录 C：可复现本次检查的只读命令

以下命令都只读；不启动模型、GPU、评估或训练。

```bash
cd /home/tianlin/SoccerMaster

nl -ba baseline/code/soccerfactory/refiner/inference.py | sed -n '94,256p'
nl -ba baseline/code/soccerfactory/refiner/model/timesformer.py | sed -n '161,303p'
nl -ba baseline/code/soccerfactory/sn-gamestate/sn_gamestate/concat_tracklets_by_reid/concat_tracklets_by_reid_api.py | sed -n '19,90p'
nl -ba baseline/code/soccerfactory/sn-gamestate/sn_gamestate/team/tracklet_team_clustering_api.py | sed -n '14,55p'
nl -ba baseline/code/soccerfactory/sn-gamestate/sn_gamestate/team/tracklet_team_side_labeling_api.py | sed -n '11,49p'

unzip -l .runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz
unzip -l .runtime/local_takeover/g10/sngs10004_refiner/output/refined_sn-gamestate.pklz
unzip -l .runtime/local_takeover/g10/sngs10004_step3/states/sn-gamestate.pklz

env CUDA_VISIBLE_DEVICES='' PYTHONPATH='' LD_LIBRARY_PATH='/home/tianlin/SoccerMaster/.envs/SoccerMaster-repro/lib:/usr/local/cuda/lib64' \
  /home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python -c \
  "import zipfile,pandas as pd; p='.runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz'; z=zipfile.ZipFile(p); d=pd.read_pickle(z.open('10004.pkl')); im=pd.read_pickle(z.open('10004_image.pkl')); print(d.columns.tolist()); print(im.columns.tolist()); print(d[['image_id','track_id','bbox_ltwh','team','role','jersey_number','bbox_pitch']].head(3))"

find /remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/valid \
  -mindepth 2 -maxdepth 2 -type f -name 'Labels-GameState.json' | head

python3 -c "import json; p='runs/reports_legacy_20260819/g10/20260819_step3_refiner_preserving_cpu/merge_review_summary.json'; d=json.load(open(p)); print([(x['new_track_id'],x['source_track_ids'],x['source_frame_ranges']) for x in d['groups']])"
```

## 附录 D：本次新增或修改文件

- 新增：`research/soccerfactory_tracking_context.md`
- 修改现有 pipeline、配置、模型或数据：无
