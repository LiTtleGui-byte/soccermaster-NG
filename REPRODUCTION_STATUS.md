# SoccerMaster 复现进度

更新日期：2026-08-12

## Gate 总览

| Gate | 状态 | 结论 |
| --- | --- | --- |
| G0 资产定位 | 基本完成 | 真实代码、high-resolution 配置、SigLIP2 和 `epoch_19` 已定位 |
| G1 Python/依赖/CUDA 环境 | 通过 | 共享环境已验证；本地高速环境已建立，NumPy 已恢复为单一 1.26.4 并通过核心导入检查 |
| G2 完整 checkpoint 加载 | 通过 | 共享环境和本地高速环境均完成 CPU-only 七组件加载，所有 missing/unexpected keys 为空 |
| G3 随机张量 forward | 通过 | 单张 H800 上以 float32 完成两个 dataset 分支、五个任务头的随机张量 forward，shape/device/有限性断言全部通过 |
| G4 单个真实视频 | 未开始 | 未执行 |
| G5 固定小规模评估 | 未开始 | 未执行 |
| G6 tiny overfit | 未开始 | 未执行 |
| G7 单任务训练 | 未开始 | 未执行 |
| G8 小规模多任务训练 | 未开始 | 未执行 |
| G9 完整训练 | 未开始 | 未执行 |
| G10 SoccerFactory 分支 | 未开始 | 未执行 |

## 已确认

### G0 资产定位

- 本地代码：`/home/tianlin/SoccerMaster`
- 原始真实代码、数据和权重：`/remote-home/haolinyang/sports/Soccer-Backbone`，永久只读。
- 目标配置：`configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml`
- SigLIP2：`/remote-home/haolinyang/sports/Soccer-Backbone/pretrained_models/google/siglip2-large-patch16-512`
- checkpoint：`/remote-home/haolinyang/sports/Soccer-Backbone/outputs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19`
- checkpoint 包含 backbone、text model、五个任务头和训练状态。

### G1 Python/依赖/CUDA 环境

- 共享参考 Python：`/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`
- 共享环境已验证版本：Python 3.10.16、torch 2.4.1、CUDA build 12.1、transformers 4.51.3、accelerate 1.8.1、cv2 4.11.0、yaml 6.0.2、einops 0.8.1。
- 历史 CUDA 可见性检查为 True，但本轮本地环境生成和导入验证没有使用 GPU。
- 本地候选环境：`/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro`
- 本地候选 Python：`/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`
- 本地环境由原共享环境打包并重定位；最终 `conda-pack` 耗时 12,061 秒，退出码 0；本地解包耗时 30 秒，退出码 0。
- 本地环境大小约 9.31 GB；可恢复 tar 归档约 9.24 GB，仍保留在 `.local_envs/SoccerMaster-repro.tar`。
- `SentencePiece 0.2.0` 已离线安装到本地环境。
- 本地环境导入验证退出码为 0；日志：`reports/local_env_import_validation_20260812.log`。
- 核心导入耗时：torch 1.004 秒、transformers 0.633 秒、accelerate 0.116 秒、cv2 0.025 秒、SentencePiece 0.008 秒、`MultiScaleDeformableAttention` 0.001 秒。
- 全部导入脚本内部耗时 1.790 秒，命令墙钟 2.11 秒，峰值 RSS 440,804 KiB。
- `decord 0.6.0` 实际导入成功，退出码 0。
- 本地环境的 NumPy 混装已修复：实际运行版本和 Python 元数据版本均为 1.26.4，只保留 `numpy-1.26.4.dist-info`。
- NumPy RECORD 专项检查无缺失、无哈希错误、无重复路径；NumPy/SciPy/OpenCV ABI 冒烟检查退出码为 0。
- 修复前的混合 NumPy 已保存在 `.local_envs/SoccerMaster-repro.numpy_backup_20260812_before_fix`，约 95 MiB，未经批准不得删除。

### G2 完整 checkpoint 加载

- 验证脚本：`scripts/verify_epoch19_load.py`
- 安全条件：CPU-only、offline、`ckpt_type="soccer_master"`、`load_heads=True`；未创建 dataset/DataLoader/optimizer/scheduler，未执行 forward、eval、inference 或 train。
- 首次失败：缺少 `MultiScaleDeformableAttention` 导入路径；未构建模型、未读取 checkpoint。
- CPU-only 扩展导入测试通过，耗时 238.1 秒，退出码 0；日志：`reports/g2_extension_import_20260811.log`。
- retry1 失败：`GemmaTokenizer` 缺少 SentencePiece；未进入 `model.load_checkpoint()`。
- retry2 在共享参考 Python 中通过，退出码 0；日志：`reports/g2_epoch19_load_retry2_20260811.log`。
- retry2 墙钟耗时 15:32.90，峰值 RSS 8,274,084 KiB（约 7.89 GiB）。
- backbone、text model、`SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption` 和 `CaptionClassification` 全部加载成功。
- 七个组件的 missing keys 和 unexpected keys 全部为空，error 全部为 `null`。
- G2 完成后没有进入 G3，没有执行 forward、推理、训练或 GPU 任务。
- 本地高速环境等价 G2 于 2026-08-12 通过；日志：`reports/g2_epoch19_load_local_env_20260812.log`。
- 本地等价 G2 使用 `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`，Python 3.10.16、torch 2.4.1、CUDA build 12.1、transformers 4.51.3、accelerate 1.8.1、NumPy 1.26.4、SentencePiece 0.2.0。
- 本地等价 G2 明确设置 `CUDA_VISIBLE_DEVICES=""`、CPU-only 和 offline；`PYTHONPATH` 只包含本地 ops 构建目录与本地仓库，`LD_LIBRARY_PATH` 只包含本地环境的 torch/lib 与 lib。
- 本地等价 G2 的脚本退出码、pipeline 退出码均为 0；机器结果断言再次解析日志并通过。
- 本地等价 G2 墙钟耗时 4:07.07，user time 17.32 秒，system time 7.86 秒，峰值 RSS 8,412,036 KiB（约 8.02 GiB），swap 为 0。
- 本地等价 G2 中 backbone、text model、`SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption` 和 `CaptionClassification` 全部加载成功；七个组件的 missing/unexpected keys 均为空，error 均为 `null`。
- 本地等价 G2 没有发生 fallback，没有创建 dataset/DataLoader/optimizer/scheduler，没有执行 forward、eval、inference、train，没有使用 GPU，也没有进入 G3。

### G3 随机张量 forward

- 验证脚本：`scripts/verify_g3_random_forward.py`；日志：`reports/g3_random_forward_gpu7_20260812.log`。
- 运行清单记录的 Git commit 为 `adf7e3f57cd6823009436f3918cb97072bf92834`，运行时工作区为 dirty，完整文件列表保存在日志中。
- 使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 7（逻辑 `cuda:0`）、NVIDIA H800、compute capability 9.0、torch 2.4.1、CUDA build 12.1。
- 输入为固定 seed 42 生成的 float32 随机张量，shape `[1, 30, 3, 512, 512]`，数值范围 `[-1, 1]`；模型使用 `eval()` 和 `torch.inference_mode()`。
- checkpoint 仍使用真实 high-resolution `epoch_19`、`ckpt_type="soccer_master"`、`load_heads=True`；CPU 加载后完整移动到单张 GPU。
- `SoccerNetGSR_Detection` 分支完成一次 forward，并同时覆盖检测、球场线和关键点三个头。
- `VideoCaption` 分支使用一条固定非空文本完成一次 forward，并同时覆盖视频文本对齐和字幕分类两个头。
- 检测关键输出 shape：boxes `[1,30,300,4]`、logits `[1,30,300,1]`、roles `[1,30,300,6]`、jersey number `[1,30,300,101]`、digit head `[1,30,300,10]`、digit tail `[1,30,300,11]`、query features `[1,30,300,256]`。
- 球场线热图 shape `[1,30,24,256,256]`；关键点热图 shape `[1,30,58,256,256]`。
- VideoCaption 的 vision/text features 均为 `[1,1024]`，相似度矩阵为 `[1,1]`；CaptionClassification logits 为 `[1,23]`、features 为 `[1,1024]`。
- 所有输出都位于 `cuda:0`；浮点输出均为 float32；全部数值有限，没有 NaN 或 Inf；机器结果二次解析断言通过。
- 脚本内部总耗时 26.907 秒；外层墙钟 27.98 秒。分阶段耗时：导入 4.747 秒、CPU 构建 8.549 秒、CPU checkpoint 加载 8.670 秒、移动 GPU 1.378 秒、检测 forward 2.250 秒、字幕 forward 0.829 秒。
- 峰值 GPU allocated 6,328,850,432 bytes（约 5.89 GiB），峰值 GPU reserved 8,531,214,336 bytes（约 7.95 GiB）；峰值 CPU RSS 8,536,492 KiB（约 8.14 GiB），swap 为 0。
- 脚本退出码和 pipeline 退出码均为 0；没有 fallback、没有 dataset/DataLoader/loss/backward/optimizer/train、没有真实视频，也没有进入 G4。
- G3 前 GPU 7 上已有用户明确知晓并批准共存的外部训练进程；G3 结束后显存恢复到运行前约 35,811 MiB 占用，没有遗留本次验证进程的 GPU 显存。

### 已知工作区状态

- `data/video_caption.py` 存在复制前已有的一行注释删除，不得回滚或覆盖。
- `AGENTS.md`、`REPRODUCTION_STATUS.md`、`reports/`、`scripts/verify_epoch19_load.py`、`scripts/verify_g3_random_forward.py`、`.local_deps/`、`.local_envs/`、`.conda_pkgs/` 均是本地资产；其中部分路径受 `.gitignore` 影响，具体状态以每次运行清单为准。
- `.conda_pkgs:/` 是首次 Conda 缓存参数分隔符错误留下的约 20 KB 异常目录，内含零字节 partial 文件；尚未删除。
- 当前本地文件系统剩余空间约 189 GiB。

## 推断

- 本地环境本次 G2 墙钟 4:07.07，历史共享环境 G2 为 15:32.90；本次观测明显更短，但受文件缓存、GPFS 当时负载等因素影响，不能仅凭两次运行把全部差异归因于环境迁移。
- 由于 SigLIP2 和 `epoch_19` 仍在 GPFS，后续首次或冷缓存 checkpoint 读取仍可能受共享存储速度影响。

## 未知

- 本地环境 G2 等价性已经确认；不同缓存与 GPFS 负载条件下的稳定加速幅度仍未知。
- G4 及以后所有真实样本、评估和训练行为均未知。

## 风险

- `pip check` 报告 `decord 0.6.0 is not supported on this platform`，但实际 `import decord` 已成功。
- 本地 NumPy 1.26.4 目前由 Python 元数据管理，不再保留错误的 Conda 2.0.1 记录；以后不得使用 Conda 操作 NumPy，除非先制定并批准一致性迁移方案。
- 本地环境中 `tracklab.pth` 和 `sn_gamestate.pth` 仍指向只读 GPFS 路径；相关功能仍可能受 GPFS 速度影响。
- `.local_envs/`、环境 tar 归档和 Conda 缓存不得提交到 Git；清理前需要用户批准。
- 后续长时间命令仍必须使用 timeout、心跳和明确退出码。

## 下一步

- G3 已通过；当前停在 G3 完成点，不自动进入 G4。
- 唯一建议的最小下一步：只读定位一个固定真实视频及其解码、归一化和元数据契约，再由用户决定是否授权 G4。
- 未经用户明确批准，不读取真实视频内容、不进入 G4。
- 任何 GPU 操作前必须重新执行 `nvidia-smi`、报告显存占用并等待用户明确批准。
