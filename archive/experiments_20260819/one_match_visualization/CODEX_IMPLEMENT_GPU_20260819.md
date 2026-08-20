在 `/home/tianlin/SoccerMaster` 内立即实现，不要重新规划或扩大审读。

当前主 Goal 已完成必要接口审查，并生成 `experiments/one_match_visualization/prepare_sngs10004_cpu.py`。你的唯一任务是用 `apply_patch` 新增以下两个最小文件，然后做一次 `py_compile`/`bash -n` 针对性检查并退出；不要运行 CPU 正式阶段或 GPU：

1. `experiments/one_match_visualization/run_soccermaster_five_heads_gpu.py`
   - 复用 `reproduction/gates/g5_fixed_eval.py` 的 high-resolution 配置、真实 `epoch_19` 加载、DataLoader/模型构建和后处理接口。
   - 输入是 CPU 脚本创建的 `.runtime/one_match/sngs10004_data_view` 中 `SNGS-10004` 的首个合法 30 帧 clip。
   - inference-only、`torch.no_grad()`/`inference_mode()`、模型 `eval()`；源代码中不得出现 backward、optimizer、scheduler、训练或 checkpoint 保存。
   - 最少两次 forward：Detection dataset branch 同时取得 `SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`；Caption branch 同时取得 `VideoCaption`、`CaptionClassification`。
   - 生成真实预测：检测框 overlay、lines heatmap/overlay、keypoints heatmap/overlay；CaptionClassification 输出 23 类 top-5；VideoCaption 与同一 23 个事件短语的真实模型相似度 top-5，明确标为 retrieval 而非生成解说。
   - 只写 `reports/one_match/20260819_sngs10004_end_to_end/soccermaster_gpu_result.json` 和 `visuals/06...10...` 图片；拒绝覆盖。记录 checkpoint、dtype、输入 shape、设备、五头结果、峰值 GPU 显存、wall time、training=false。
   - 可从准备脚本导入常量，但 GPU worker 不得创建或修改数据视图。

2. `experiments/one_match_visualization/launch_sngs10004_end_to_end.py`
   - 分阶段本地 supervisor：新鲜输出下运行一次 CPU prepare；CPU 通过后用宿主 `nvidia-smi` 轮询。
   - GPU 合格条件固定为同一 H800 连续两次、间隔至少 60 秒：free>=70000 MiB、util<=5%、无 compute process；从合格卡选最小 index。GPU 0–5 若有 g9 进程自然不合格，绝不能终止/改变进程。
   - 每次快照打印全部 0–7 卡总/已用/空闲、利用率和 compute process；等待期间每 30 秒心跳。两次确认后只启动一次 worker，失败即停，不换卡、不修复重跑。
   - worker 命令显式设置 `CUDA_VISIBLE_DEVICES`、本地 Python、`PYTHONPATH=/home/tianlin/SoccerMaster`、本地 `LD_LIBRARY_PATH`、offline 环境；timeout 合理且 30 秒心跳。
   - CPU 和 GPU 都成功后聚合生成最终 `result.json`、`summary.md`，检查标为完成的图片可打开且五个头齐全，并更新 `research_log.md` 与 `lab_notes/2026-08/2026-08-19.md`（只写实际结果）。
   - 主日志固定 `reports/one_match/20260819_sngs10004_end_to_end/run.log`，不得制造额外 manifest/hash/audit 文件。

完整遵守 `AGENTS.md`：远端/NAS/共享环境只读，工作区已有 dirty 文件不能覆盖。本任务禁止训练和自动重跑。本轮只实现与语法检查，不运行正式 launcher，也不查询 GPU。
