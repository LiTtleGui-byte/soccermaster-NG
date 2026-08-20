# SNGS-10004 单场端到端可视化 Goal

请立即调用 `create_goal` 创建并持续执行下面的目标，不要只给计划，也不要在中途把工作交回用户：

> 在 `/home/tianlin/SoccerMaster` 内，以固定比赛片段 `SNGS-10004` 为唯一样本，完成 SoccerFactory 与 SoccerMaster 主干五任务头的机器结果和可视化交付。先读必要代码/论文与现有证据，写极短研究计划，然后尽快实现和运行。只有形成最终 `summary.md`、`result.json` 和可打开的图片清单后才结束 Goal。

## 固定边界和授权

- 完整遵守根目录 `AGENTS.md`、`REPRODUCTION_STATUS.md` 和 `docs/HARNESS.md`；新会话接管顺序不能省略。
- `/remote-home/haolinyang/sports/Soccer-Backbone`、共享环境、NAS/GPFS 数据和权重永久只读。所有新文件只能写入本地工作区。
- 保留 dirty worktree，不回滚、不覆盖无关改动。
- 本任务属于单样本 inference-only 研究交付，不改变正式 Gate 结论。
- 禁止训练、backward、optimizer、scheduler、checkpoint 写入/覆盖、自动重跑、终止或改变其他用户进程。
- 不重新运行已有且证据完整的 SoccerFactory 重模型阶段；优先复用固定 `SNGS-10004` 的 Step 1、enrichment、Refiner、Step 3 和 PKL 产物，再生成统一可视化。只有确实缺失且 CPU 可安全补出的阶段才补跑。
- 用户已经对本任务一次性明确授权：CPU 预检通过后，由本地 launcher 用 `nvidia-smi` 检查 GPU；只有同一张 H800 连续两次、间隔至少 60 秒都满足空闲显存不少于 70,000 MiB、利用率不超过 5%、无任何 compute process，才选择编号最小的合格卡运行一次 SoccerMaster 单样本 inference-only worker。条件不满足就持续等待并留心跳；worker 失败立即停止并保留日志，不换卡、不修复后重跑。
- GPU worker 启动前必须在日志和终端报告全部 0–7 卡的总/已用/空闲显存、利用率和 compute process。运行命令必须显式记录 `CUDA_VISIBLE_DEVICES`、Python、`PYTHONPATH`、`LD_LIBRARY_PATH`、dtype、checkpoint、输入、timeout、30 秒心跳和输出目录。
- 不使用 `sudo`、`danger-full-access`、`--yolo` 或绕过审批模式。

## 要回答的问题

以同一个 `SNGS-10004` 具体片段为例，用户能直接看到：

1. 原始比赛视频如何得到这 255 张准备图片；必须区分序列名 `SNGS-10004` 和 mapping `clip_id=4`。将历史切片逻辑改成只写本地全新目录的 CPU 单片段复现，并与已有 255 帧逐帧核对。不得运行会写远端的历史入口。
2. SoccerFactory 每个主要阶段输出了什么：准备帧、人物检测框、track ID、ReID/轨迹、球场关键点/线、相机/球场坐标、号码可读性与 OCR、角色、球队、Refiner 前后、Step 3、最终训练 PKL 与 SoccerMaster DataLoader。需要机器结果，也需要少量代表性可视化；不能把结构跑通写成语义正确。
3. SoccerMaster 主干对同一片段的五个当前任务头输出了什么：`SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption`、`CaptionClassification`。必须是模型预测或明确标注为输入/真值，不得把标注伪装成预测。检测、线和关键点至少有 overlay；两个 caption 头至少有可读文本/类别及置信度或 top-k（如果接口只输出 embedding，则如实展示可计算的现有相似度/分类结果）。
4. 每个结果对应哪段代码、哪个输入、哪个模型/checkpoint、哪个产物路径，以及当前可信范围。

## 实施原则

- 先复用并理解：`docs/PIPELINE_MAP_SOCCERMASTER.md`、`docs/PIPELINE_MAP_SOCCERFACTORY.md`、`docs/ASSET_INVENTORY.md`、`experiments/task_head_visualization/`、`experiments/soccerfactory_visualization/`、`reproduction/gates/g5_fixed_eval.py` 和所有 `g10_*sngs10004*` 固定清单/结果。
- 先检查 `g9_run6_six_gpu_one_epoch` 及其他 tmux/GPU 作业是否仍在运行；不得占用其卡或干扰它。
- 建一个本地分阶段 launcher。CPU 阶段先生成/核对原视频切片、聚合已有 SoccerFactory 产物并生成图片；GPU 阶段只做一次 SoccerMaster 五头 forward；最后 CPU 汇总。
- 长命令必须有 timeout、30 秒心跳和明确退出码。一次正式运行只保留一个主日志和一个主结果文件，不制造多余 hash/manifest/audit 文件。
- 优先快速产出代码、定量结果和图；仅做会影响结论的 sanity checks。不要把精力花在 CI、格式化、覆盖率或契约表演。
- 如果现有主干一次 forward 无法让五头共享同一 batch，应说明接口原因并用最少的合法 forward 覆盖五头；不得静默跳头。
- 不自动训练或修改模型来改善效果；本 Goal 是跑通与可视化，不是修复质量问题。

## 固定交付目录和完成标准

新产物统一写入：

`reports/one_match/20260819_sngs10004_end_to_end/`

至少包含：

- `summary.md`：一页式中文说明，按 SoccerFactory / SoccerMaster pipeline 顺序列出阶段、结果图、代码位置、已确认和未知。
- `result.json`：每阶段 `status`、输入、代码、输出、是否新运行/复用、设备、checkpoint、关键数值、限制；顶层明确总状态。
- `run.log`：launcher 主日志，带阶段、心跳、GPU 两次快照和退出码。
- `visuals/`：代表性 PNG/JPG；文件名带阶段编号，至少覆盖原视频/准备帧、Step 1 检测+track、球场线/关键点、角色/球队/号码、Refiner 前后、Step 3/PKL、SoccerMaster detection/lines/keypoints、两个 caption 头。
- 如适合，额外生成一个短 MP4/GIF 或联系表帮助理解时间变化，但不能替代上述静态图。
- `launcher/` 或 `experiments/one_match_visualization/` 中保留可重复入口，默认不得训练，默认不得覆盖既有正式结果。

完成前做一次针对性检查：所有 `result.json` 中标为完成的图片实际存在且可打开；五个任务头没有遗漏；原始视频 lineage 不再误写为 `clip_id=10004`；没有训练调用或远端写入。然后更新根目录 `research_log.md` 和当天 UTC `lab_notes/2026-08/2026-08-19.md`。只有这些完成后才把 Goal 标为 complete，并返回最终产物路径、运行结果、失败/未验证范围。
