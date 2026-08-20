# SoccerMaster 复现进度

更新日期：2026-08-20

## 2026-08-20目录迁移说明

- 冻结原版代码现位于`baseline/`，可修改研究代码位于`research/src/soccermaster/`，官方checkpoint和固定数据位于`assets/`。
- 重构前的`reproduction/`、`experiments/`、`reports/`、`outputs/`和`exports/`均未删除，分别归档到`archive/`和`runs/`；完整映射见`archive/PATH_MAP.md`。
- 下文2026-08-19及以前记录中的旧绝对路径表示运行当时的位置；读取当前文件时应按路径映射定位，不能把目录移动解释为Gate证据丢失或重新通过。
- `.local_assets`、`.local_envs`、`.local_deps`和`.conda_pkgs`保留为隐藏兼容链接。新工作只使用`assets/`、`.envs/`和`research/`规范路径。
- 目录迁移本身只做CPU文件与导入调整，不改变G0-G10任何结论。
- 迁移后经用户单次授权，在物理GPU 7完成固定SNGS-10004前30帧的五头inference-only路径smoke：新`research/src`源码、`research/configs`配置和`assets/checkpoints/official/soccermaster/epoch_19`均被真实运行消费；退出码0、五头和5张图齐全，峰值allocated/reserved为6,422,995,456/8,529,117,184 bytes。证据位于`runs/path_smoke_20260819/soccermaster/`。这是目录迁移诊断，不重新判定G4，也不验证训练或总体质量。
- 随后经新的单次授权，在物理GPU 7完成固定SNGS-10004全部255帧的SoccerFactory路径smoke：`baseline/code/soccerfactory`、`assets/checkpoints/official/soccerfactory`和`assets/data`被真实消费，Step 1、enrichment、coord-only Refiner、CPU Step 3、训练PKL转换和真实DataLoader均退出0。链路保持255帧、3,390人物和48轨迹，最终batch为`[1,30,3,512,512]`；峰值reserved为19,639,828,480 bytes。证据位于`runs/path_smoke_20260819/soccerfactory/`。该诊断不改变G10“固定链已通、语义质量未通过”的结论。

## Gate 总览

| Gate | 状态 | 结论 |
| --- | --- | --- |
| G0 资产定位 | 基本完成 | 真实代码、high-resolution 配置、SigLIP2 和 `epoch_19` 已定位 |
| G1 Python/依赖/CUDA 环境 | 通过 | 共享环境已验证；本地高速环境已建立，NumPy 已恢复为单一 1.26.4 并通过核心导入检查 |
| G2 完整 checkpoint 加载 | 通过 | 共享环境和本地高速环境均完成 CPU-only 七组件加载，所有 missing/unexpected keys 为空 |
| G3 随机张量 forward | 通过 | 单张 H800 上以 float32 完成两个 dataset 分支、五个任务头的随机张量 forward，shape/device/有限性断言全部通过 |
| G4 单个真实视频 | 通过 | 固定 SoccerReplay-1988 视频完成解码、测试预处理和 Caption 两任务头推理，并保存机器结果与联系图 |
| G5 固定小规模评估 | 通过 | retry1 完成固定 2 个检测 clip + 23 个 Caption 视频的两遍评估，五类指标结构完整，重复性最大差异为 0，退出码 0 |
| G6 tiny overfit | 通过 | 4 个固定真实视频的 CaptionClassification 分类器在 110 steps 内从 0% 达到 100% accuracy，loss 从 3.87645 降至 0.01003，梯度/参数范围断言全部通过 |
| G7 单任务训练 | 通过 | retry2 完整运行 2 epochs、184 optimizer steps 和每 epoch train/valid 评估；8 项机器断言全部通过，退出码 0 |
| G8 小规模多任务训练 | 通过（功能性恢复） | run5 完成两卡、两 epoch、8 个全局 optimizer steps；全部功能断言通过，但不具备位级 exact resume |
| G9 完整训练 | 未通过（run6超时） | GPU 0–5上的单epoch长跑稳定到日志step 8,299/16,009，18小时timeout后退出124；未完成epoch且未生成checkpoint |
| G10 SoccerFactory 分支 | 进行中（固定单片段接口链已跑通） | SNGS-10004已从原始半场视频本地复现255帧，并串到Step 1、enrichment、Refiner、隔离Step 3、PKL、真实DataLoader和可视化；语义质量与上游原版入口等价性仍未通过 |

## 已确认

### G0 资产定位

- 本地代码：`/home/tianlin/SoccerMaster`
- 原始真实代码、数据和权重：`/remote-home/haolinyang/sports/Soccer-Backbone`，永久只读。
- 2026-08-17 在 `gpu200` 只读复核确认 `/remote-home` 已恢复挂载：来源为 `gpfsdata`、文件系统为 GPFS、挂载选项包含 `ro`。`Soccer-Backbone`、Refiner 目录和共享参考 Python 均可访问；Refiner `best_model.pth` 可见且大小为 323,985,486 bytes，与冻结 manifest 记录一致。本次未重算大文件哈希、未反序列化 checkpoint、未执行 GPU 查询。
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
- 本地环境导入验证退出码为 0；日志：`reports/g1/local_env_import_validation_20260812.log`。
- 核心导入耗时：torch 1.004 秒、transformers 0.633 秒、accelerate 0.116 秒、cv2 0.025 秒、SentencePiece 0.008 秒、`MultiScaleDeformableAttention` 0.001 秒。
- 全部导入脚本内部耗时 1.790 秒，命令墙钟 2.11 秒，峰值 RSS 440,804 KiB。
- `decord 0.6.0` 实际导入成功，退出码 0。
- 本地环境的 NumPy 混装已修复：实际运行版本和 Python 元数据版本均为 1.26.4，只保留 `numpy-1.26.4.dist-info`。
- NumPy RECORD 专项检查无缺失、无哈希错误、无重复路径；NumPy/SciPy/OpenCV ABI 冒烟检查退出码为 0。
- 修复前的混合 NumPy 已保存在 `.local_envs/SoccerMaster-repro.numpy_backup_20260812_before_fix`，约 95 MiB，未经批准不得删除。

### G2 完整 checkpoint 加载

- 验证脚本：`reproduction/gates/g2_checkpoint_load.py`
- 安全条件：CPU-only、offline、`ckpt_type="soccer_master"`、`load_heads=True`；未创建 dataset/DataLoader/optimizer/scheduler，未执行 forward、eval、inference 或 train。
- 首次失败：缺少 `MultiScaleDeformableAttention` 导入路径；未构建模型、未读取 checkpoint。
- CPU-only 扩展导入测试通过，耗时 238.1 秒，退出码 0；日志：`reports/g2/g2_extension_import_20260811.log`。
- retry1 失败：`GemmaTokenizer` 缺少 SentencePiece；未进入 `model.load_checkpoint()`。
- retry2 在共享参考 Python 中通过，退出码 0；日志：`reports/g2/g2_epoch19_load_retry2_20260811.log`。
- retry2 墙钟耗时 15:32.90，峰值 RSS 8,274,084 KiB（约 7.89 GiB）。
- backbone、text model、`SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption` 和 `CaptionClassification` 全部加载成功。
- 七个组件的 missing keys 和 unexpected keys 全部为空，error 全部为 `null`。
- G2 完成后没有进入 G3，没有执行 forward、推理、训练或 GPU 任务。
- 本地高速环境等价 G2 于 2026-08-12 通过；日志：`reports/g2/g2_epoch19_load_local_env_20260812.log`。
- 本地等价 G2 使用 `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`，Python 3.10.16、torch 2.4.1、CUDA build 12.1、transformers 4.51.3、accelerate 1.8.1、NumPy 1.26.4、SentencePiece 0.2.0。
- 本地等价 G2 明确设置 `CUDA_VISIBLE_DEVICES=""`、CPU-only 和 offline；`PYTHONPATH` 只包含本地 ops 构建目录与本地仓库，`LD_LIBRARY_PATH` 只包含本地环境的 torch/lib 与 lib。
- 本地等价 G2 的脚本退出码、pipeline 退出码均为 0；机器结果断言再次解析日志并通过。
- 本地等价 G2 墙钟耗时 4:07.07，user time 17.32 秒，system time 7.86 秒，峰值 RSS 8,412,036 KiB（约 8.02 GiB），swap 为 0。
- 本地等价 G2 中 backbone、text model、`SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption` 和 `CaptionClassification` 全部加载成功；七个组件的 missing/unexpected keys 均为空，error 均为 `null`。
- 本地等价 G2 没有发生 fallback，没有创建 dataset/DataLoader/optimizer/scheduler，没有执行 forward、eval、inference、train，没有使用 GPU，也没有进入 G3。

### G3 随机张量 forward

- 验证脚本：`reproduction/gates/g3_random_forward.py`；日志：`reports/g3/g3_random_forward_gpu7_20260812.log`。
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

### G4 单个真实视频

- 验证脚本：`reproduction/gates/g4_real_video.py`；日志：`reports/g4/g4_real_video_gpu7_20260812.log`；机器结果：`reports/g4/g4_real_video_result_20260812.json`；可视联系图：`reports/g4/g4_real_video_contact_sheet_20260812.jpg`。
- 运行使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 7（逻辑 `cuda:0`）、float32、seed 42、`model.eval()` 和 `torch.inference_mode()`。
- 输入固定为 SoccerReplay-1988 `classification_test.json` 第 0 条；视频大小 6,144,860 bytes，SHA256 为 `2408eeb2eb11c0fce43067b8eb7a392ded4d4efc78cc7d91aecb3147ccf289b3`，标签为 `end of half game`。
- 项目现有 Decord 路径成功解码 30 帧，原始 tensor shape `[30,3,572,1024]`、uint8；`middle` 采样帧号为 12、37、62、87、112、137、162、187、212、237、262、287、312、337、362、387、412、437、462、487、512、537、562、587、612、637、662、687、712、737。
- `split="test"` 预处理成功产生 `[30,3,512,512]` float32 CPU tensor；数值有限，范围 `[-1,1]`，均值约 -0.40254。
- 完整 high-resolution `epoch_19` 仍以 `ckpt_type="soccer_master"`、`load_heads=True` 加载；单次真实视频 forward 只进入 `VideoCaption` dataset 分支，同时验证 `VideoCaption` 与 `CaptionClassification` 两个头。
- VideoCaption vision/text features 均为 `[1,1024]`，两个 similarity matrix 均为 `[1,1]`；CaptionClassification logits 为 `[1,23]`、features 为 `[1,1024]`。全部输出位于 `cuda:0`、数值有限，文本有效 mask 为真。
- 本样本分类 top-1 为真实标签 `end of half game`，概率约 0.94771；该结果只作单样本可检查输出，不代表总体评估指标。
- 脚本内部耗时 23.557 秒，外层墙钟 24.62 秒；解码 1.193 秒、预处理 0.493 秒、CPU 模型构建 8.514 秒、checkpoint 加载 4.136 秒、移动 GPU 1.583 秒、forward 1.075 秒。
- 峰值 CPU RSS 8,651,656 KiB（约 8.25 GiB）；forward 后记录的峰值 GPU allocated 5,764,115,456 bytes（约 5.37 GiB）、reserved 6,228,541,440 bytes（约 5.80 GiB）。
- 脚本退出码、pipeline 退出码和机器结果二次解析均为 0；没有 fallback，没有创建 Dataset/DataLoader/loss/optimizer/scheduler，没有 backward 或训练，也没有进入 G5。
- G4 前 GPU 7 已有用户明确批准共存的外部 PID 1506769；结束后 GPU 7 恢复到约 35,821 MiB 占用，仅剩该外部进程，没有本次验证遗留 GPU 进程。
- 本次 G4 的证明范围是固定真实视频的 Caption 完整输入与推理路径；没有在真实视频上执行检测、球场线或关键点头，也没有证明总体指标。

### G5 固定小规模评估

- G5 于 2026-08-12 按用户授权在物理 GPU 7 上只运行一次；退出码为 1，结论为失败，没有自动修复或重跑。
- 原始日志：`reports/g5/g5_fixed_small_eval_gpu7_20260812.log`；机器结果：`reports/g5/g5_fixed_small_eval_result_20260812.json`。
- 日志 SHA256 为 `4e3699af30ffa6ebeeb8a8431121062ed7ddcb657dbaa8e863736115e5cfc081`；结果 JSON SHA256 为 `813ebf483b50301c0828a3a2471b70ebf33bb210433e038ae6aafff0a283b486`。
- 直接错误为 `FileNotFoundError: [Errno 2] No such file or directory: '/home/tianlin/SoccerMaster/reports/g5_fixed_data_view/SN-GSR-2024/camera_params/test/SNGS-116.json'`。
- 已确认原始只读目录中对应文件 `/remote-home/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/camera_params/test/SNGS-116.json` 存在，大小 2,756,920 bytes，权限 664。
- 已确认直接原因：本地数据视图只建立了 `SoccerNetGS/test/SNGS-116` 和 `legibility_jn/test.json` 两个只读资产链接，而当前数据集构造代码还会读取 `camera_params/test/SNGS-116.json`。
- 失败发生在 `build_fixed_detection_dataset`。本次只完成 manifest/资产校验、本地视图准备、框架导入、CUDA 初始化、配置读取和 Caption 记录校验。
- 脚本内总耗时 11.789 秒，外层墙钟 12.85 秒；峰值 CPU RSS 1,263,260 KiB，swap 为 0。
- CUDA 初始化后采样为 allocated/reserved 均为 0，当时空闲 46,920,564,736 bytes；模型未构建、checkpoint 未读取、未执行 forward 或任何指标计算，因此两次评估和重复性均未开始。
- 本次没有训练、backward、optimizer、scheduler 或 G6。结束后 GPU 7 恢复为 35,821 MiB 占用，只剩授权前已存在的外部 PID 1506769，本次验证没有遗留 GPU 进程。
- 固定输入清单：`reproduction/manifests/g5_fixed_eval.json`。
- 独立验证脚本：`reproduction/gates/g5_fixed_eval.py`；运行前通过 `ast.parse`、manifest 契约、无训练调用和 `git diff --check` 静态检查。
- 检测部分固定为 `SNGS-116` 的两个不重叠 30 帧 clip，起始帧分别为 0 和 360；共 60 帧，静态盘点确认图片存在并具有球场线标注。
- Caption 部分固定为 SoccerReplay-1988 test 中每个类别一个可访问样本，共覆盖 23/23 类、23 个视频。
- 目标指标包括 detection AP/mAP、precision/recall/F1、属性匹配、lines/keypoints 指标、CaptionClassification 指标，以及在统一 23×23 相似度矩阵上计算的 VideoCaption retrieval top-1/3/5。
- 已确认根目录 `eval.py` 不适合作为该 Gate 的直接入口：其 checkpoint 调用仍把 logger 作为第二个位置参数传入，且默认构造并遍历完整测试集。G5 需要独立的本地验证入口显式限定 manifest。
- 目标脚本设计为在单张可见 GPU 上加载模型一次，连续执行两遍相同固定清单，并比较检测输出摘要、Caption logits、23×23 相似度矩阵、帧索引和所有指标；容差预设为 `rtol=1e-5`、`atol=1e-6`。本次失败未到达该阶段。
- 脚本当时在本地 `reports/` 创建受限数据视图符号链接、JSON 和两张检测叠加图；整理后运行时数据视图位于 `.runtime/data_views/g5/`，远端资产保持只读，不复制数据。
- 本次已导入项目模块并初始化 CUDA，但没有读取视频帧、加载模型/checkpoint、执行 forward 或生成指标。
- 用户已批准针对该直接原因的最小本地修改。`reproduction/gates/g5_fixed_eval.py` 现已将远程只读 `camera_params/test/SNGS-116.json` 加入必需资产检查和本地受限数据视图。
- retry1 使用了新产物路径：`reports/g5/g5_fixed_small_eval_retry1_gpu7_20260812.log`、`reports/g5/g5_fixed_small_eval_retry1_result_20260812.json` 和两张名含 `retry1` 的检测叠加图，没有覆盖首次失败证据。
- 修改后 `ast.parse`、manifest 2+23 数量、camera-params 源文件存在性、新输出路径未占用和 `git diff --check` 均通过。
- G5 retry1 已于 2026-08-12 按用户对最新 GPU 状态的明确授权在物理 GPU 7 上运行一次；脚本退出码和 pipeline 退出码均为 0，没有 fallback，G5 结论为通过。
- retry1 日志：`reports/g5/g5_fixed_small_eval_retry1_gpu7_20260812.log`；机器结果：`reports/g5/g5_fixed_small_eval_retry1_result_20260812.json`。日志 SHA256 为 `5e625da2bb501df91cc5b1236613631c03c48fc94eb491bdb56f3c55c8290a9a`，JSON SHA256 为 `ed5ffb37228780d380d8e349acdccd27abf9af97e8ea0ee0bd47f9db1c972a85`。
- 固定输入为 `SNGS-116` 起始帧 0 和 360 的两个 30 帧 clip（共 60 帧，dataset indices 0 和 12），以及 23 个 Caption 类别各一个真实视频（总计 127,506,848 bytes）。
- high-resolution `epoch_19` 以 `ckpt_type="soccer_master"` 和 `load_heads=True` 完整加载；两个 dataset 分支和 `SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption`、`CaptionClassification` 五个任务头均进入评估。
- 两遍评估结果通过 `rtol=1e-5`、`atol=1e-6` 重复性断言；Caption 帧索引完全一致，classification logits、23×23 similarity 和所有数值叶子的最大绝对/相对差异均为 0。
- 固定小规模指标：Detection mAP 0.4944773、mAP@0.5 0.8626513、mAP@0.75 0.4987767、precision 0.9713656、recall 0.8655545、F1 0.9154125；role accuracy 0.9365079，jersey accuracy 0.7460317。
- Lines accuracy/F1 为 0.9666666/0.9525341；Keypoints accuracy/F1 为 0.9613095/0.9178377；两者均基于 60 个 valid samples。
- CaptionClassification accuracy 为 0.3478261，macro F1 为 0.2855072；VideoCaption retrieval top-1/3/5 为 0.3478261/0.5217391/0.6521739；均基于 23 个每类一样本的固定清单，不代表完整测试集或论文指标。
- 脚本内总耗时 569.495 秒，外层墙钟 9:30.87；checkpoint CPU 加载 255.304 秒，第一/二遍评估 155.481/123.663 秒。峰值 CPU RSS 8,596,992 KiB，swap 为 0。
- 峰值 GPU allocated 8,124,680,192 bytes（约 7.57 GiB），reserved 10,590,617,600 bytes（约 9.86 GiB）。结束后 GPU 7 恢复为 35,821 MiB 占用，只剩原外部 PID 1506769，没有本次验证遗留进程。
- 两张检测叠加图 `reports/g5/g5_detection_clip_start_000_retry1_overlay_20260812.jpg` 和 `reports/g5/g5_detection_clip_start_360_retry1_overlay_20260812.jpg` 均存在并已成功打开检查。
- G5 retry1 没有 loss、backward、optimizer、scheduler、DataLoader、分布式评估或训练，没有进入 G6。

### G6 tiny overfit

- G6 于 2026-08-12 按用户对最新 GPU 状态的明确授权，在物理 GPU 7 上只运行一次；脚本退出码和 pipeline 退出码均为 0，没有 fallback，G6 结论为通过。
- 固定清单：`reproduction/manifests/g6_tiny_overfit.json`；本地入口：`reproduction/gates/g6_tiny_overfit.py`。
- 固定数据是 G5 中被 `epoch_19` 初始误分类的 4 个 SoccerReplay-1988 真实视频，标签分别为 `second yellow card`、`throw in`、`show added time` 和 `start of half game`，共 21,947,461 bytes。
- 输入固定使用 30 帧 `middle` 采样和 deterministic test transform；不使用训练随机增强，因为本 Gate 只验证微型数据上的标签、loss、梯度与 optimizer 链路。
- 模型仍从真实 high-resolution `epoch_19` 加载，但 G6 范围只构建 `CaptionClassification` 头。backbone 对 4 个视频各执行一次无梯度特征提取，随后对固定特征执行 tiny-overfit。
- 唯一可训练范围是 `CaptionClassification.classifier` 的 weight 和 bias，共 23,575 个参数。backbone、text encoder、CaptionClassification transformer/layer-norm 和其他任务头均不更新。
- loss 使用项目原生 `CaptionClassificationLoss` 交叉熵；optimizer 为 AdamW，学习率、weight decay 和 gradient clip 分别从 `LR_CAPTION_CLASSIFICATION_CLASSIFIER`、`WEIGHT_DECAY` 和 `MAX_CLIP_NORM` 读取；不使用 scheduler。
- 最多 1,000 steps，每 10 steps 检查一次；成功必须同时满足：4/4 accuracy、final loss 不高于 0.2、final/initial loss ratio 不高于 0.25，并连续保持 5 次检查。
- 额外机器断言包括：每步梯度非零且有限、optimizer 只包含两个分类器参数、可训练参数实际变化、冻结参数无梯度且 version 不变。
- 运行没有保存模型、optimizer 或 checkpoint；只写入了 `reports/g6/g6_tiny_overfit_gpu7_20260812.log` 和 `reports/g6/g6_tiny_overfit_result_20260812.json`。
- 脚本和 manifest 已通过 `ast.parse`、JSON/样本契约、资产存在性、输出路径未占用、唯一 heartbeat thread、唯一 backward/optimizer step 代码路径、无 checkpoint save 调用和 `git diff --check` 静态检查。
- 日志 SHA256 为 `7b3e8edba77209d8bce244b372116cdc6aee06eabff8f122c29da964d60baa78`；结果 JSON SHA256 为 `012482bb1b2b65c8d25d0a96f35ecb451f591ac45763cfdcc0f9be3718aab7de`；机器结果二次解析断言通过。
- 初始 accuracy 为 0/4，loss 为 3.8764548；第 50 step 首次达到 4/4 accuracy，第 70 step 起同时满足 loss 和 loss-ratio 阈值，到第 110 step 连续 5 次满足全部成功条件并提前停止。
- 最终 accuracy 为 4/4，loss 为 0.0100294，final/initial loss ratio 为 0.0025873；四个预测标签均与目标一致，单样本最大类别概率为 0.9843–0.9948。
- AdamW 实际使用 lr 0.0002、weight decay 0.0001、gradient clip max-norm 0.1；110/110 steps 梯度均非零且有限，梯度 norm 范围为 0.0711568–16.4075069。
- 分类器参数最大绝对变化为 0.0345927；optimizer 只包含 classifier weight/bias。冻结的 891 个参数无梯度且 version 全部不变。
- 脚本内总耗时 31.230 秒，外层墙钟 32.39 秒；checkpoint CPU 加载 4.321 秒，4 个真实视频特征提取 10.256 秒，tiny-overfit 训练 1.347 秒。峰值 CPU RSS 8,433,976 KiB，swap 为 0。
- 整个运行的峰值 GPU allocated 至少 5,689,222,144 bytes（约 5.30 GiB），峰值 reserved 6,140,461,056 bytes（约 5.72 GiB）。结束后 GPU 7 恢复为 35,821 MiB 占用，只剩原外部 PID 1506769，没有本次训练遗留进程。
- 全部 9 项成功断言均为 true；本 Gate 没有 DataLoader、scheduler、mixed precision、分布式训练、checkpoint 保存/恢复或 G7。

### G7 单任务训练设计与静态准备（运行前记录）

- 2026-08-12 已按 `docs/HARNESS.md` 完成只读设计审查，并按用户批准创建现位于 `reproduction/manifests/g7_single_task.json` 和 `reproduction/gates/g7_single_task.py` 的固定清单与验证入口；静态准备当时没有导入 PyTorch、加载模型、创建 checkpoint、使用 GPU 或启动训练。
- 推荐任务为 `CaptionClassification`。它已经覆盖 G4 真实视频、G5 固定评估和 G6 分类器 tiny-overfit，继续使用它可以把新增变量限定为完整任务头训练、独立验证集、scheduler 和 checkpoint 恢复；检测和 VideoCaption 对比学习暂不作为 G7 首选。
- G7 必须继续使用 high-resolution `unisoccer_part_temporal` 架构和真实 `epoch_19`，不能直接使用仓库中的 `pure_siglip` downstream 配置。建议解析目标 high-resolution 配置后仅覆盖 `DATASETS_TO_HEADS={VideoCaption: [CaptionClassification]}`、冻结 vision/text encoder，并把数据根、基础模型、checkpoint 和输出目录解析为绝对路径。
- 唯一建议的数据源是只读 `SoccerReplay-1988`。其标注包含 train 104,080 条、valid 17,892 条、test 17,402 条，三个 split 都覆盖 23 类；相对视频名在三个 split 间的交集均为 0。
- train/valid/test 标注 SHA256 分别为 `7dc57c841217c1c5280bb1f5e70dd2e56454b94d6dcb2e7fd3e7e54078484aa1`、`0a4e5b3d49d6f2e5c0560b6e5efb475edb91776197fae6d262be1c58f4ba8340`、`fc65c3b5f061c57c8850452d1e5f1d11e584f1c022a5d4cecb3a8acf826ad123`。
- 已抽查 SoccerReplay-1988 train 的首、中、末三个 high-resolution 视频，均可读取；尚未逐个 stat 全部 121,972 条 train+valid 记录引用的视频。后续生成 manifest 时必须逐项验证选中资产，任何缺失立即失败，不能静默换样本。
- 原配置同时列出 `SoccerReplay-1988`、`MatchTime` 和 `SoccerNet-v2`。只读抽查确认后两者的视频软链接目标当前不存在，不能直接按原三数据源配置训练；其 JSON 标注存在不等于视频可用。
- 固定最小范围已经写入 manifest：每类 8 个 train 样本和 4 个 valid 样本，即 184 train + 92 valid，全部来自各自 split，共 276 个互不重复的视频，train/valid 视频交集为 0；test 不参与训练、调参或 Gate 成功判断。manifest 固定了选择策略、JSON index、标签文件 SHA256 和每个视频当前大小；逐视频内容 SHA256 未在本轮完整读取，实际运行会再次逐项检查路径与大小。
- 建议训练完整 `CaptionClassification` 头：两层 transformer encoder、两组 classifier layer norm 和最终 linear classifier，共 25,220,119 个 float32 参数；backbone 和 text encoder 全部冻结并保持 eval。该参数数由历史头 checkpoint 的 30 个 tensor storage 共 100,880,476 bytes 只读核对得到。
- 脚本已固定为单卡、float32、batch size 2、30 帧、512×512、`num_workers=0`、不使用 AMP；AdamW 的 head-other/classifier 学习率分别为 `1e-4`/`2e-4`，weight decay `1e-4`，gradient clip `0.1`，共 2 epochs，`CosineAnnealingLR` 每个完整 epoch 后 step 一次、最低学习率 `1e-8`。
- 每个样本的顺序、帧采样和训练增强必须由 seed、epoch、manifest index 确定；记录 Python、NumPy、PyTorch CPU 和目标 CUDA RNG。`num_workers=0` 可避免首轮 G7 引入 worker RNG 和 GPFS 多 worker 并发变量。
- G7 至少记录初始、每 epoch 和最终的 train/valid loss、accuracy、macro accuracy/precision/recall/F1、每类样本数、学习率、梯度范数、吞吐、分阶段耗时、CPU RSS 与 GPU allocated/reserved。成功要求包括：所有固定样本无跳过、loss/梯度/参数有限、完整头中非 classifier 参数实际更新、冻结 backbone/text 不变、训练 loss 相比初始基线下降、valid 指标结构完整。
- exact-resume 检查必须从一个中间保存点分叉：未中断分支与重新构建并恢复的分支读取相同下一批样本，并比较恢复前后的 batch IDs、loss、学习率、optimizer/scheduler 状态和一次参数更新，容差在脚本创建时明确固定。只证明“能够继续跑”不算通过。
- G7 checkpoint 设计为仅保存完整 CaptionClassification 头、optimizer、scheduler、进度、全部必要 RNG/采样状态和引用的 `epoch_19`/配置标识，不重复保存冻结 backbone/text。运行前估算单份约 303 MiB；retry1 的 step-3 checkpoint 后续实测为 302,695,666 bytes，详见下方运行记录。
- 新 checkpoint 必须写入本地 `outputs/` 下的同文件系统临时目录，生成并校验 manifest，最后写 `COMPLETE`，再原子 rename。当前 `train.py` 的 `save_training_state()` 会直接逐文件写正式 `epoch_*` 目录，不满足该事务式契约。
- 当前 `train.py`/数据入口还有四个阻塞点：本地 `datasets/VideoCaption` 不存在；`data/video_caption.py` 含远端仓库 `sys.path` 追加；`data/build.py` 为 VideoCaption epoch 评估固定构造 `test` 而不是 `valid`；现有随机状态代码把 PyTorch RNG 误标为 Python/NumPy RNG，且不保存 sampler/micro-step。G7 不应直接运行根目录 `train.py`。
- 当前本地文件系统剩余约 189 GiB，轻量 G7 checkpoint 空间充足。G5/G6 的实测峰值 reserved 分别约 9.86/5.72 GiB；retry1 后来完成了部分真实训练，但因中止没有采集完整两 epoch 的 GPU 峰值和总时长。
- 验证入口不导入或调用根目录 `train.py`，不使用 DataLoader，也不包含 `classification_test` 路径；它串行解码固定清单、使用独立 valid split，并把训练范围限定为完整 `CaptionClassification` 头。
- 脚本在第 3 个 optimizer step 后保存轻量 checkpoint，再用同一下一批分别执行未中断分支和新建 head/optimizer/scheduler 的恢复分支；机器断言比较 batch IDs、输入 SHA256、loss、head 参数、optimizer state 和 scheduler state。
- checkpoint 协议已静态实现为本地临时目录写入、逐文件大小/SHA256 manifest 校验、`COMPLETE` 标记和同文件系统 `os.replace()` 原子 rename；任何目标路径已存在时拒绝覆盖，失败时不自动清理证据。
- 创建后的静态检查结果：`ast.parse` 通过、manifest JSON 解析通过、184/92 样本和 23 类契约通过、276 个视频唯一且 train/valid 零重叠、无 DataLoader 调用、无 `train.py` import、无 `classification_test`、唯一 heartbeat thread、唯一 backward/optimizer-step 代码路径，`git diff --check` 通过。
- 静态检查首次发现 valid JSON 第 70 和 71 条跨标签引用同一视频；最终 manifest 已跳过重复的第 71 条，并把 `red card` 第四个样本固定为 JSON index 1049。该问题在训练前已消除。

#### G7 首次运行尝试（失败，未进入训练）

- 用户明确批准在物理 GPU 7 上运行一次 G7，并知晓与外部 PID 1506769 共存；约定失败不自动修复或重跑、不进入 G8。
- 授权前宿主机 `nvidia-smi` 显示 GPU 7 为 NVIDIA H800，总显存 81,559 MiB、已用 35,821 MiB、空闲 45,269 MiB；外部 PID 1506769 占用约 35,778 MiB，GPU 7 负载在相邻采样间波动。
- 运行命令使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、`CUDA_VISIBLE_DEVICES=7`、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、offline、float32、单线程 CPU、30 秒脚本心跳和 21,600 秒 timeout；日志现归档为 `reports/g7/20260812_first_attempt/run.log`。
- 本次进程退出码为 1；直接错误为 `RuntimeError: G7 requires exactly one visible CUDA device`，发生在 `import_framework` 完成后的 CUDA 可见性前置断言。
- 已确认直接原因是运行命令仍处于受限沙箱；同一沙箱内此前 `nvidia-smi` 也无法与宿主机驱动通信。该失败不表示宿主机 GPU 7 或项目 CUDA 环境损坏。
- 失败前只完成 manifest 加载、路径校验、固定记录校验和框架/项目导入；耗时分别为 0.000、0.229、0.329 和 5.082 秒。脚本总耗时 5.647 秒，外层墙钟 6.64 秒，峰值 RSS 810,384 KiB，swap 为 0。
- 模型未构建，`epoch_19` 未读取，没有视频解码、forward、loss、backward、optimizer、scheduler、评估或 checkpoint 保存；因此 G7 训练协议、exact resume 和 valid 指标仍未验证。
- 机器结果：`reports/g7/20260812_first_attempt/result.json`，状态为 `failed`；日志 SHA256 为 `0a0ba8ad9c61e1ea91bd9cceeab0b12c8e502eda80d3ce5693bce1e6cd006066`，结果 JSON SHA256 为 `5767a96f5fd96940f15a94aff3e84e95b998846f1a74c12395db00e0e4c0c8a1`。
- 预定正式 checkpoint 路径和临时路径均不存在；失败后宿主机 `nvidia-smi` 仍只显示外部 PID 1506769，GPU 7 已用 35,821 MiB、空闲 45,269 MiB，没有本次尝试遗留 GPU 进程或显存。
- 没有发生 fallback，没有自动修改脚本或重跑；G7 结论为失败且未进入训练，G8 未开始。

#### G7 retry1 静态准备（运行前记录）

- 2026-08-12 用户批准准备 G7 retry1。为保留首次失败证据，此次运行现归档到 `reports/g7/20260812_retry1_interrupted/`，事务式 checkpoint 现归档到 `outputs/g7/20260812_retry1_interrupted/step_000003`。
- 本次只改名本地预定产物，未导入 PyTorch/项目模块，未读取模型或 checkpoint，未使用 GPU，未进行 forward、backward、optimizer、scheduler 或训练。
- retry1 静态检查已通过：`ast.parse`、manifest 184 train + 92 valid/23 类/276 视频唯一/train-valid 零重叠、固定资产存在性与大小、`git diff --check` 均通过；新日志、结果、正式 checkpoint 和临时 checkpoint 路径均未被占用。
- 2026-08-12 06:02:45 宿主机只读 `nvidia-smi` 显示 GPU 7 总显存 81,559 MiB、已用 35,821 MiB、空闲 45,269 MiB、GPU/显存利用率 99%/44%、温度 70°C、P0；外部 PID 1506769 占用 35,778 MiB。空闲显存高于脚本 25 GiB 门槛，但当时计算负载很高，retry1 尚未获得针对该快照的共存运行批准。

#### G7 retry1 运行（用户中止，未通过）

- 用户在了解 GPU 7 与外部 PID 1506769 共存风险后，明确批准只运行一次 G7 retry1，失败不自动重跑、不进入 G8。
- 运行使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 7（逻辑 `cuda:0`）、float32、batch size 2、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30 秒心跳和 21,600 秒 timeout；日志现归档为 `reports/g7/20260812_retry1_interrupted/run.log`。
- 运行从 2026-08-12 06:04:35 开始。manifest、固定资产和 CUDA 前置检查通过；CPU 模型构建 14.855 秒，真实 high-resolution `epoch_19` 加载 111.810 秒，backbone 全键匹配，text encoder 和 CaptionClassification 头均加载。
- 训练前的 92 个 train batches 和 46 个 valid batches 全部完成；epoch 0 的 92 个 optimizer steps、完整 train/valid 评估和 scheduler step 完成；epoch 1 完成 global step 93–111，并开始第 20/92 batch。
- 脚本在 global step 3 之后生成事务式轻量 checkpoint，global step 4 的未中断与重新恢复分支之后继续运行到 step 111；根据脚本控制流，该次 next-batch/input SHA/loss/head/optimizer/scheduler 等价性断言未失败。但最终结果 JSON 未写入，因此不能将整个 G7 描述为 exact resume 验证通过。
- step-3 checkpoint 路径现为 `outputs/g7/20260812_retry1_interrupted/step_000003`，存在 `COMPLETE`、manifest、完整 CaptionClassification 头、optimizer、scheduler、RNG 和 training state；5 个 manifest 文件合计 302,695,666 bytes，中止后重新校验大小和 SHA256 全部通过，临时目录不存在。该 checkpoint 只代表 step 3，不包含后续 step 4–111。
- 用户随后明确要求不跑完全程。作业于 06:48:31 停止，墙钟约 43 分 56 秒，外层退出码为 141；日志 SHA256 为 `216fab50264c4904784df7b31c07a1b8ce561277ec19ca9382cb35090584d854`。
- 中止使进程没有进入脚本的最终结果写入；`reports/g7/20260812_retry1_interrupted/result.json` 不存在，`/usr/bin/time -v` 未留下完整末尾。因此最终 train/valid 数值、全部成功断言、峰值 CPU RSS 和实际训练峰值 GPU 显存均未采集；只能确认移入 GPU 后的早期 allocated/reserved 分别为 3,799,319,552/3,804,233,728 bytes，不能当作整段训练峰值。
- 停止后没有本次 G7 匹配进程；GPU 7 恢复为已用 35,821 MiB、空闲 45,269 MiB，只剩外部 PID 1506769。G7 因未完成规定的 2 epochs 和最终断言而未通过，G8 未开始。

#### G7 retry2 完整运行（通过）

- 用户在了解 GPU 7 与外部 PID 1506769 共存、GPU 利用率 99% 和性能风险后，明确批准只运行一次完整 2 epochs 的 G7 retry2；失败不自动重跑、不进入 G8。
- 运行前只修改 `reproduction/gates/g7_single_task.py` 的三个本地输出路径，改为 `reports/g7/20260812_retry2_full_2epochs/` 和 `outputs/g7/20260812_retry2_full_2epochs/`；训练协议、manifest、模型、checkpoint、设备、精度、batch size、数据范围和断言均未改变。`ast.parse`、manifest 契约、276 个固定视频大小、train/valid 零重叠、新路径未占用和 `git diff --check` 均通过。
- 运行使用 commit `9da39e171dc961ef1c054667741f3ad546f37324` 的本地工作区、本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 7（逻辑 `cuda:0`）、float32、batch size 2、`num_workers=0`、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30 秒心跳和 21,600 秒 timeout。
- 输入保持为 `reproduction/manifests/g7_single_task.json`：SoccerReplay-1988 的 184 个 train 与 92 个 valid 固定视频，23 类均衡，train/valid 视频零重叠；test split 未访问。vision backbone 和 text encoder 全部冻结，只训练完整 CaptionClassification 头。
- 运行于 2026-08-12 07:46:45 开始，脚本耗时 4,096.137 秒，外层墙钟 1:08:17；CPU 模型构建 22.527 秒，真实 high-resolution `epoch_19` CPU 加载 179.916 秒。backbone 全键匹配，text encoder 和 CaptionClassification 头均成功加载。
- 两个 epoch 的 184 个 optimizer steps 全部完成，共处理 368 个训练样本位置；每个 epoch 后 scheduler、184 样本 train 评估和 92 样本 valid 评估均完成。epoch 0/1 的 online mean train loss 分别为 1.7529373/1.1921330。
- 固定 train 的初始、epoch 0、epoch 1 loss 为 1.6711464、1.0111239、0.8424046；accuracy 为 0.5000000、0.6739130、0.7228261；macro F1 为 0.4714747、0.6616970、0.7194353。
- 固定 valid 的初始、epoch 0、epoch 1 loss 为 1.9712630、1.6194372、1.7600775；accuracy 为 0.4673913、0.5978261、0.5869565；macro F1 为 0.4426846、0.5902971、0.5728967。该固定小规模 valid 在第二个 epoch 略低于第一个 epoch，不能据此宣称总体泛化持续改善。
- step 3 的事务式 checkpoint 为 `outputs/g7/20260812_retry2_full_2epochs/step_000003`；5 个 manifest 文件总计 302,695,666 bytes，`COMPLETE`、逐文件大小和 SHA256 二次核验均通过。它仍只表示 step 3 的恢复探针，不是 epoch 2 最终模型 checkpoint。
- exact-resume 探针通过：恢复分支与未中断分支的下一批 IDs、输入 SHA256 和 loss 一致，head 最大参数差及 optimizer state 最大差均为 0，scheduler state 一致。
- 8 项机器断言全部为 true：每 epoch 所有 train 样本均处理、梯度有限且为正、非 classifier 头参数发生变化、冻结 backbone/text 不变、最终 train loss 低于初始值、valid 覆盖全部 23 类、exact-resume 探针通过、test split 未使用。梯度 norm 范围为 0.1334345–34.0916977，非 classifier 头参数最大变化为 0.00600642。
- 峰值 CPU RSS 为 8,433,136 KiB，swap 为 0；峰值 GPU allocated/reserved 分别为 8,183,918,592/9,481,224,192 bytes。运行结束后 GPU 7 恢复为已用 35,821 MiB、空闲 45,269 MiB，只剩外部 PID 1506769，没有本次 G7 残留进程。
- 日志：`reports/g7/20260812_retry2_full_2epochs/run.log`，SHA256 `b32c3b938657d0b806806a094dcc1a0e6b5acbfebabc9fec0d3f15302bee852b`；机器结果：`reports/g7/20260812_retry2_full_2epochs/result.json`，SHA256 `038e2809ea98560800f89ef4cd053ba192c5db84e43c84f8c3ed6c2fbb060982`。脚本、timeout 和 pipeline 退出码均为 0，`status="passed"`、`error=null`，没有 fallback。
- G7 达到当前 Harness 的最小证明范围并标记为通过；这只证明固定小规模单任务协议、恢复和评估链路可信，不代表完整数据集训练或论文总体指标。G8 未开始。

### G8 小规模多任务只读设计审查

- 2026-08-12 已完整读取 `AGENTS.md`、`docs/HARNESS.md`、`REPRODUCTION_STATUS.md`、原始多任务配置、`train.py`、两个数据入口、五个任务头的 loss/分布式路径和原始训练日志。本次只读审查没有修改代码、创建 G8 入口、使用 GPU 或启动训练。
- 原始一个 optimizer step 依次处理 `SoccerNetGSR_Detection` 和 `VideoCaption` 两个 dataset 分支，分别 backward，最后统一 optimizer step。两个分支共覆盖 `SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption` 和 `CaptionClassification` 五个头。
- 真实训练配置为每 rank detection batch 1、caption batch 2、30 帧、512×512、`VIDEO_CAPTION_SIGLIP_LOSS_WEIGHT=4.0`；vision backbone 和全部五头可训练，text encoder 冻结。
- 原始日志记录 967.79M 总参数、402.22M 可训练参数，其中 vision backbone 358.77M、五头 43.45M；原 H200 运行的每-rank 最大显存记录约 123,115 MiB。这超过当前 80 GB H800 的单卡容量，不得原样直接运行。
- 已确认不应直接使用根目录 `train.py` 作为 G8 Harness：它会按最长 DataLoader 循环较短数据集、通过 `batch.values()` 依赖字典顺序、默认用 `test` 作 epoch 评估，且当前 checkpoint/RNG/任务调度状态不足以证明多任务 exact resume。
- high-resolution `UniSoccerBackbone` 的现有 gradient-checkpointing 开关不能作为未验证 fallback：`train.py` 调用的 `gradient_checkpointing_enable()` 不在该自定义 backbone 上，内部 checkpoint 分支的 block 调用也缺少 `B,T` 参数。
- 推荐首个 G8 最小协议使用两张 GPU/两个 DDP rank、float32、`num_workers=0`，冻结 vision backbone 和 text encoder，只训练全部五头。每 rank 每步固定处理 detection batch 1 和 caption batch 2，两次 backward 后只做一次 optimizer step。
- 建议固定检测 train 为 `SNGS-060` 和 `SNGS-062` 各4个30帧 clip；独立 held-out 为 `SNGS-061` 的4个30帧 clip。候选起点已只读核对，每个 clip 的30帧均有人物和球场标注。Caption 建议复用 G7 manifest 前8类的16个 train 和8个 valid 视频。
- 建议训练2 epochs，每 epoch 4 个全局 optimizer steps，共8步。必须机器断言：两 rank 任务/collective 顺序一致、固定样本无循环或跳过、五头 raw/weighted loss 有限、五头均获得有限非零梯度并更新、冻结参数不变、两 rank 更新后参数一致、VideoCaption gather 尺寸一致、step 3 多任务 exact-resume 一致、五头 valid 指标结构完整、不访问 test 且无 fallback。
- 冻结 backbone 后的每卡显存估算为 12–25 GiB，但尚未实测。运行前建议每张目标 GPU 至少有30 GiB 空闲显存，且仍必须按 `AGENTS.md` 重新执行 `nvidia-smi` 并获得针对当次设备和命令的明确授权。
- 该最小协议只能证明五头多任务调度、梯度、DDP 同步和指标记录；不验证共享 backbone 的多任务梯度冲突，不能等同于原始完整多任务训练，也不自动授权 G9。
- 已创建 `reproduction/manifests/g8_multitask.json` 和 `reproduction/gates/g8_multitask.py`。AST 和 JSON 解析、G8 数据契约、两 rank/五头/8 步设置、无 test split 和静态副作用检查均通过；没有导入项目、使用 GPU、读取 checkpoint 或启动训练。
- 静态复核发现并修正了两个 manifest 契约错误：训练检测样本由4个补足为两 rank × 4 steps 所需的8个；`SNGS-061` 明确记录为从上游 `train` 资产划出的 held-out 序列，不再错误指向不存在的官方 `valid` 资产。Caption train/valid 标签 SHA256、24个固定视频大小、360张固定检测帧和必需 checkpoint 路径均通过只读检查。
- 已建立实际两 rank 入口 `reproduction/gates/g8_multitask_run.py`。它的目标协议是物理 GPU 0、1、float32、2 epochs × 4 global steps、每 rank 每步 detection batch 1 + caption batch 2、冻结完整 backbone/text、只训练五个头，并包含跨 rank 梯度同步、五头更新、固定样本计数、validation loss 结构和 step-3 exact-resume 断言。该入口在 run1 前通过 `ast.parse` 和 `git diff --check`；运行正确性尚未通过。

#### G8 run1（失败，未进入模型构建或训练）

- 用户在获知 G8 目标协议后明确批准多卡验证。2026-08-12 启动前宿主机 `nvidia-smi` 显示8张 H800 均无计算进程、利用率0；GPU 0、1 各有约81,090 MiB空闲。
- run1 使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`，物理 GPU 0、1，`torch.distributed.run --standalone --nproc_per_node=2`，offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30秒心跳和14,400秒 timeout。
- run1 外层退出码为1；脚本 rank 0 记录耗时7.162秒，外层墙钟9.54秒，峰值 RSS 925,056 KiB，swap为0。日志：`reports/g8/20260812_run1/run.log`；机器结果：`reports/g8/20260812_run1/result.json`。
- 两个 rank 均在 `import_framework` 之后、`dist.init_process_group(backend="nccl")` 处失败，完整直接错误为 `ValueError: trying to initialize the default process group twice!`。
- 直接原因已由静态导入链确认：`models/deformable_detr/deformable_detr.py` 导入 `utils.misc`，而 `utils/misc.py` 在模块级创建 `Accelerator(...).state` 并已经初始化默认进程组；G8 Harness 随后无条件再次调用 `init_process_group()`。
- 失败前没有建立数据集、没有构建模型、没有读取 SigLIP2 或 `epoch_19`、没有 forward/loss/backward/optimizer/scheduler/checkpoint；因此五头、DDP 同步、显存和 exact-resume 均未验证。`outputs/g8/20260812_run1/` 未产生 checkpoint。
- 没有 fallback、没有自动修复或重跑。退出后再次执行宿主机 `nvidia-smi`，8张卡均为0 MiB进程占用，GPU 0、1 各约81,090 MiB空闲，没有本次运行残留进程。

#### G8 run2 静态准备（运行前记录）

- 按用户批准，只修改了本地 `reproduction/gates/g8_multitask_run.py`：输出标识从 `20260812_run1` 切换到全新的 `20260812_run2`，并在项目导入后先检查 `dist.is_initialized()`。
- 如果项目导入已通过 Accelerate 建立进程组，入口会复用它并严格断言 backend 为 NCCL、rank/world size 与 torchrun 环境一致；只有进程组尚不存在时才调用 `init_process_group()`。没有修改上游 `utils/misc.py` 或任何远端文件。
- run2 的 AST、manifest/资产静态契约和分布式复用条件检查全部通过；`reports/g8/20260812_run2/result.json`、正式/临时 step-3 checkpoint 和 `.runtime/data_views/g8/20260812_run2` 均未被占用，`git diff --check` 退出码为0。本次准备没有导入 Torch/项目、读取 checkpoint、使用 GPU 或启动训练。

#### G8 run2（失败，dataset 构建未完成）

- 用户在查看 GPU 0、1 实时资源、目标协议、显存预算、timeout 和单次重跑策略后，明确批准运行 G8 run2。启动前两卡各约81,090 MiB空闲、利用率0、无计算进程。
- run2 使用本地 Python、物理 GPU 0、1、两个 NCCL rank、float32、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30秒心跳和14,400秒 timeout；日志为 `reports/g8/20260812_run2/run.log`，机器结果为 `reports/g8/20260812_run2/result.json`。
- run2 外层退出码为1；rank 0 脚本耗时5.871秒，外层墙钟8.29秒，峰值 RSS 1,082,076 KiB，swap为0。
- run1 的阻塞已确认解除：两个 rank 均输出 `reusing existing NCCL process group`，backend/rank/world size 断言通过。
- 两个 rank 随后在 `build_datasets` 阶段失败。完整直接错误为 `FileNotFoundError: .../rank_N/SN-GSR-2024/SoccerNetGS/train`。
- 原因已由代码和实际本地链接确认：`configs/default.yaml` 设置 `SoccerNetGSR_SUB_DIR: SN-GSR-2024`，dataset 会在 `DATA_ROOT/SN-GSR-2024/` 下查找资产；run2 的 `prepare_data_view()` 却把链接创建在 `DATA_ROOT/SoccerNetGS/`、`DATA_ROOT/camera_params/` 和 `DATA_ROOT/legibility_jn/`，少了一层 `SN-GSR-2024/`。远端固定资产存在且未修改。
- dataset 对象未完成构建；模型未构建，SigLIP2 和 `epoch_19` 未读取，没有 forward/loss/backward/optimizer/scheduler/checkpoint。`outputs/g8/20260812_run2/` 没有生成 checkpoint。
- run2 没有 fallback、没有自动修复或重跑。失败后再次检查，8张 GPU 均为0 MiB进程占用、GPU 0和1各约81,090 MiB空闲，没有残留进程。

#### G8 run3 静态准备（运行前记录）

- 按用户批准，只修改本地 `reproduction/gates/g8_multitask_run.py`：运行标识切换为全新的 `20260812_run3`；每个 rank 的固定检测视图改为在 `DATA_ROOT/SN-GSR-2024/` 下创建 `SoccerNetGS`、`camera_params` 和 `legibility_jn`。
- 该布局与 `configs/default.yaml` 的 `SoccerNetGSR_SUB_DIR: SN-GSR-2024` 及 dataset 的 `self.data_dir = os.path.join(data_root, sub_dir)` 一致。远端链接目标、manifest、训练语义、GPU、dtype、batch size、五头、冻结范围和断言没有改变。
- run3 的两个 G8 脚本均通过 AST；`DATA_ROOT/SN-GSR-2024/` 静态路径推导和固定资产契约通过；run3 的 report、正式/临时 step-3 checkpoint 和 data-view 路径均未占用，`git diff --check` 退出码为0。本次准备没有导入 Torch/项目、读取 checkpoint、使用 GPU 或启动训练。

#### G8 run3（失败，首个 detection forward 未完成）

- 用户在查看实时 GPU 状态和完整运行范围后明确批准单次 G8 run3。启动前 GPU 0、1 各约81,090 MiB空闲、利用率0、无计算进程。
- run3 使用本地 Python、物理 GPU 0、1、两个 NCCL rank、float32、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30秒心跳和14,400秒 timeout。日志：`reports/g8/20260812_run3/run.log`；机器结果：`reports/g8/20260812_run3/result.json`。
- run3 外层退出码为1；rank 0 脚本耗时167.816秒，外层墙钟2:49.68，峰值 CPU RSS 8,718,584 KiB，swap为0。运行未到最终资源采集，因此实际峰值 GPU allocated/reserved 未知。
- run1/run2 的两个阻塞均已解除：两 rank 成功复用 NCCL 进程组，train/held-out dataset 构建完成；两 rank 的 backbone、text model 和五个任务头均从真实 high-resolution `epoch_19` 加载，并移入 GPU。
- 两个 rank 均在 epoch 0/global step 1 的首个 detection forward 中失败。完整直接错误为 `RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'`；触发点为 `models/deformable_detr/position_encoding.py:41`。
- 已确认原因是本地 Harness 开启严格确定性，而 PyTorch 2.4.1 的 CUDA `cumsum` 没有被标记为确定性实现。这不是 checkpoint、数据缺失、OOM 或 NCCL 错误。
- 首个 detection forward 未完成；没有产生任何完整模型输出、loss、backward、梯度同步、optimizer/scheduler step、validation 或 exact-resume checkpoint。机器结果明确记录 `steps=[]`、`validation=[]`，`outputs/g8/20260812_run3/` 没有 checkpoint。
- run3 没有 fallback、没有自动放宽确定性、修复或重跑。失败后宿主机检查显示8张 GPU 均为0 MiB进程占用、GPU 0和1各约81,090 MiB空闲，没有残留进程。
- 日志 SHA256 为 `35e3baf74ca3daf62b7b17134f1002782bbc85347e9ac7b08f31943c31a3b6e3`；机器结果 SHA256 为 `f86e64843bf0fcf1108fb351ce302c4e6a0164ba9e88ae73031a221b2bb198bd`。

#### G8 run4 静态准备（运行前记录）

- 用户明确批准仅限 G8 Harness 使用 `torch.use_deterministic_algorithms(True, warn_only=True)`。这不是对项目模型代码或其他 Gate 的全局修改。
- 本地运行标识切换为全新的 `20260812_run4`；manifest 新增 `determinism_policy=enabled_warn_only_for_cuda_ops_without_deterministic_implementation`，运行结果环境字段也必须记录 deterministic algorithms 已启用且 warn-only 为真。
- 该兼容例外允许 PyTorch 2.4.1 没有严格确定性 CUDA 实现的 `cumsum` 继续运行并输出警告。它不构成官方严格确定性保证；G8 仍必须依靠固定输入 SHA、跨 rank 参数一致和 step-3 exact-resume 机器断言验证本次运行的实际一致性。
- run4 的两个 G8 脚本均通过 AST；manifest 策略、`warn_only=True` 调用和机器结果记录字段检查通过，旧的严格调用已不存在；run4 的 report、正式/临时 step-3 checkpoint 和 data-view 路径均未占用，固定资产契约与 `git diff --check` 退出码均为0。本次准备没有导入 Torch/项目、读取 checkpoint、使用 GPU 或启动训练。

#### G8 run4（失败，exact-resume 机器断言不通过）

- 用户授权“重新检查，如果空闲直接批准”。启动前宿主机确认 GPU 0、1 各约81,090 MiB空闲、利用率0、无计算进程，因此按该条件授权直接运行单次 G8 run4。
- run4 使用本地 Python、物理 GPU 0、1、两个 NCCL rank、float32、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30秒心跳和14,400秒 timeout；确定性策略为已批准的 `torch.use_deterministic_algorithms(True, warn_only=True)`。
- 日志：`reports/g8/20260812_run4/run.log`；机器结果：`reports/g8/20260812_run4/result.json`。外层退出码为1；rank 0 脚本耗时173.721秒，外层墙钟2:58.00，峰值 CPU RSS 8,726,556 KiB，swap为0。运行未到最终资源汇总，峰值 GPU allocated/reserved 未采集，仍未知。
- 两 rank 成功复用 NCCL 进程组，dataset、模型、真实 high-resolution `epoch_19` 七组件加载和移入 GPU 均通过。run4 越过 run3 的 CUDA `cumsum` 阻塞并记录 warn-only 告警。
- epoch 0 的参考分支完成 global step 1–4；每步均依次处理 detection 和 Caption 分支，五个头记录的 raw/weighted loss 有限、同步梯度 norm 均为有限正数，跨 rank 头参数哈希在各已完成 step 后一致。该运行没有到达 epoch validation 或第二个 epoch。
- global step 3 后的事务式 checkpoint 位于 `outputs/g8/20260812_run4/step_000003`，包含五头、optimizer、scheduler、两 rank RNG 和 training state。`COMPLETE` 存在、临时目录不存在；5个 manifest 文件逐一重新校验大小与 SHA256 均通过，总目录大小521,590,658 bytes。
- 恢复探针成功重新载入 step 3 并重复 step 4。机器断言结果：样本 IDs 相同、输入 SHA256 相同、raw/weighted loss 相同、scheduler 状态相同；但五头参数哈希不同、optimizer 状态哈希不同，因此 exact-resume 断言失败并使 run4 退出1。
- 日志明确记录 memory-efficient attention backward 使用非确定性算法的 warn-only 告警。这与恢复后参数差异高度相关，但当前产物只记录哈希相等/不等，没有保存每个头、每个 optimizer tensor 的最大绝对/相对差异；因此差异的具体来源和数值幅度仍未知，不能把推断写成已确认根因。
- run4 没有削弱 exact-resume 断言、没有 fallback、没有自动修复或重跑，也没有进入 G9。失败后8张 GPU 均为0 MiB进程占用、GPU 0和1各约81,090 MiB空闲，没有残留进程。
- run4 日志 SHA256 为 `b564e2ef9ab93cc26e66989a62264afd9b319adff250cdfd5b9b20dbd5c8ad39`；机器结果 SHA256 为 `eccc00addd6f0103aaa19b3aedd42eb5f49dcce50335c91f80123b787a550a11`。

#### G8 run5 静态准备（运行前历史记录）

- 按用户指示，run5 将保留位级 `exact_resume` 结果，同时增加独立的 `functional_resume` 判定；只有差异低于预先写死的容差且其余恢复断言全部通过，才允许把 G8 写成“功能性恢复通过”，不得写成 exact resume。
- 本地入口已切换到全新的 `20260812_run5`。恢复探针会保存参考 step 4 的逐头参数、逐头梯度和逐 optimizer group/state 快照，并报告最大绝对/相对差异、首个差异 tensor/state、逐组结果和哈希结果。
- 功能性容差固定为：任务头参数 `atol=1e-7, rtol=1e-5`；optimizer state `atol=1e-7, rtol=1e-5`；梯度 `atol=1e-6, rtol=1e-4`。输入 ID、输入 SHA256、loss、scheduler、shape、dtype、整数状态和 optimizer 参数组设置不使用数值容差，仍必须精确一致。
- run5 的两个 Python 入口均通过 `ast.parse`，manifest 通过 JSON 解析和只读资产契约检查；`reports/g8/20260812_run5/result.json`、正式/临时 step-3 checkpoint 和 data-view 路径均未占用，`git diff --check` 退出码为0。
- 本次静态准备没有导入 Torch 或项目模块，没有读取 checkpoint，没有使用 GPU，也没有启动训练；在该准备阶段 G8 仍未通过。

#### G8 run5（功能性恢复通过）

- 用户在查看 GPU 0、1 实时资源后，明确批准只运行一次 G8 run5，失败不自动重跑、不进入 G9。启动前两卡各有约81,090 MiB空闲、利用率0、无计算进程。
- run5 使用 commit `90f7ce1ef2cfb848110c3b2ff212dc4cdfcf6a4b` 的 dirty 工作区、本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 0、1、两个 NCCL rank、float32、offline、本地 ops/repo `PYTHONPATH`、本地环境 `LD_LIBRARY_PATH`、30秒心跳和14,400秒 timeout。
- 两个 rank 均成功复用 NCCL process group，固定 train/held-out dataset 构建完成；真实 high-resolution `epoch_19` 的 backbone、text model 和五个任务头全部加载并移入 GPU。
- 两个 epoch、8个全局 optimizer steps 全部完成；机器结果包含16条 rank-step 记录。每个 optimizer step 均依次处理 detection 和 Caption 分支，五个任务头的 loss 有限、同步梯度 norm 为有限正数、参数实际更新，且每步后的五头参数在两个 rank 间一致。
- step 3 事务式 checkpoint 位于 `outputs/g8/20260812_run5/step_000003`。`COMPLETE` 存在、临时目录不存在；五个 manifest 文件合计521,585,637 bytes，逐文件大小和 SHA256 二次核验全部通过。
- 恢复后重复 step 4 的输入 ID、输入 SHA256、loss 和 scheduler state 精确一致。位级哈希仍不一致：`exact_resume.passed=false`；数值差异只出现在 `SoccerNetGSR_Detection`，其余四头和相应 optimizer groups 均位级一致。
- `SoccerNetGSR_Detection` 中有5个参数 tensor 非位级一致；最大绝对差为 `7.450580596923828e-09`、最大相对差为 `5.781730123999296e-06`，满足预设 `atol=1e-7, rtol=1e-5`。同组有12个 optimizer state tensor 非位级一致；最大绝对差为 `5.4569682106375694e-12`、最大相对差为 `0.004833337850868702`，满足预设 `atol=1e-7, rtol=1e-5`。optimizer 较大的相对值来自接近零的数值，绝对差远低于容差。
- 同一头中有6个梯度 tensor 非位级一致；最大绝对差为 `5.4569682106375694e-11`、最大相对差为 `0.1147540956735611`，满足预设 `atol=1e-6, rtol=1e-4`；较大的相对值同样来自接近零的梯度。首次差异位于 `SoccerNetGSR_Detection.input_proj.0.0.bias`。
- `functional_resume_passed=true`、`verdict="functional_resume"`。所有最终功能断言均通过：固定训练位置全覆盖、五头非零有限梯度、五头更新、冻结 backbone/text 不变、跨 rank 参数一致、VideoCaption gather shape 正确、两 epoch validation 指标结构完整、未访问 test、无 fallback。
- 两个 epoch 的 CaptionClassification held-out accuracy 均为0.25；五个任务头均产生4个 validation batches/epoch。该固定小规模指标只证明评估链路完整，不代表总体性能。
- 脚本耗时367.113秒，外层墙钟6:10.89，峰值 CPU RSS 8,718,276 KiB、swap为0。每个 rank 的峰值 GPU allocated 为13,221,223,936 bytes，峰值 reserved 为16,741,564,416 bytes。
- 日志：`reports/g8/20260812_run5/run.log`，SHA256 `6011d06b4ae6e79b020b4bb143607e1ef6e9fc69b002e2e49e1ac6b53481fe6d`；机器结果：`reports/g8/20260812_run5/result.json`，SHA256 `1eb57ec5a6c39d6a10bd72b66ad7eac9bdd7a5fce27f998a528935364d7d70bd`。脚本、timeout 和 pipeline 退出码均为0，`status="passed"`、`error=null`。
- 运行仍记录 CUDA `cumsum` 和 memory-efficient attention backward 的 warn-only 非确定性警告。因此本结论是“G8 功能性恢复通过”，不是位级 exact resume，也不是严格确定性保证。
- 运行结束后8张 GPU 均为0 MiB进程占用、利用率0，无本次运行残留进程。没有自动重跑，也没有进入 G9。

### G9 完整训练 run1（失败，首个 forward CUDA OOM）

- 用户明确授权在检查 GPU 可用性后直接启动一次 G9 完整训练，允许以 OOM 作为当前硬件可行性证据；失败后不自动改配置或重跑。启动前 8 张 NVIDIA H800 均为 0 MiB 进程占用、利用率 0，每卡约 81,090 MiB 空闲。
- run1 使用本地 Python `/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`、物理 GPU 0–7、8 个 NCCL rank、float32、`accelerate --mixed_precision=no`、offline、本地 ops/repo `PYTHONPATH` 和本地环境 `LD_LIBRARY_PATH`。命令带 30 秒心跳、14 天 timeout、明确 pipeline 退出码和 `/usr/bin/time -v` 资源记录。
- 运行配置为 `.runtime/g9_run1_config.yaml`，只把原始 high-resolution 配置的继承路径、SigLIP2、数据、额外 7,000 样本、输出目录和实验名解析到本地/只读绝对路径，并关闭 W&B；模型、30 帧、512×512、batch size、数据集列表、学习率、20 epochs 和冻结范围与目标配置保持一致。
- 运行日志为 `reports/g9/20260812_run1/run.log`，SHA256 `45a9a240532c7f2d6883d782698ea0727b628c5035950e4c261d81b4d650945c`。输出目录 `outputs/g9/20260812_run1/` 仅含 resolved config、文本日志和 88-byte TensorBoard 事件文件，共约 28 KiB；没有产生 checkpoint。
- 8 个 rank 均完成 7,000 个额外检测序列的 lazy 索引初始化；完整 train/test DataLoader、模型、optimizer 和 scheduler 均已构建。实际 dataloader 长度为 detection 12,007、VideoCaption 11,410；epoch 0 计划 12,007 iterations。
- 模型参数契约与原始记录一致：总参数 967.79M、可训练 402.22M；vision 358.77M 全部可训练、text 565.57M 全部冻结、五个任务头合计 43.45M 全部可训练。
- 失败发生在 epoch 0、首个 `SoccerNetGSR_Detection` batch 的视觉骨干 forward 内，具体触发点为 SigLIP MLP GELU。rank 0、2、6 均报告 `torch.OutOfMemoryError`：额外申请 960 MiB 失败；rank 0/6 当时各仅余 162.50 MiB，进程约占 79.02 GiB，PyTorch allocated 73.06 GiB、reserved-but-unallocated 约 2.52–2.55 GiB。心跳观测的最高整卡占用为 80,927 MiB。
- 首个 forward 未完成，没有 loss、backward、梯度同步、optimizer step、scheduler step、validation 或 checkpoint；因此完整训练完成性、指标和恢复仍未验证。该失败属于当前 80 GiB H800 上原样 FP32 协议的资源不足，不是数据缺失、checkpoint 加载错误或 NCCL 初始化错误。
- 外层训练退出码为 1；开始时间 `2026-08-12T20:43:28-07:00`，结束时间 `2026-08-12T21:24:51-07:00`，`/usr/bin/time` 墙钟 41:23.34。外层进程峰值 CPU RSS 为 7,517,044 KiB、swap 为 0；该 RSS 不是 8 个 rank 与 DataLoader workers 的聚合峰值。
- 没有 fallback、没有自动降精度、修改模型、启用不可用的 gradient checkpointing 或重跑。退出后再次检查，8 张 GPU 均为 0 MiB 占用、利用率 0，无本次运行残留进程。

#### G9 run2 BF16 单步探针（用户中止，未进入模型或训练）

- 用户先批准在保留 GPU 0、4 两个外部推理服务的前提下启动 8 卡 BF16 单步探针，随后于运行中明确要求“暂时停止训练”。本次停止优先执行，没有等待数据初始化结束。
- run2 使用 `.runtime/g9_run2_bf16_probe_config.yaml` 和 `.runtime/g9_run2_bf16_probe_launcher.sh`。相对 run1，运行配置只新增 `LOGGING_INTERVAL=1` 并使用新的实验名/输出目录；启动参数改为 `--mixed_precision=bf16`。模型、数据、30 帧、512×512、batch size、学习率、五头和解冻范围未改变。
- 启动时间为 `2026-08-12T22:04:34-07:00`；停止完成时间为 `2026-08-12T22:08:05-07:00`。训练子命令墙钟为 3:07.84，峰值外层 CPU RSS 为 2,826,220 KiB、swap 为 0。
- 停止时 8 个 rank 均位于 `SoccerNetGSR_Detection._init_extra_data_lazy()` 的 pickle 读取阶段。模型、optimizer 和 scheduler 尚未构建；没有 forward、loss、backward、optimizer step、evaluation 或 checkpoint。
- 退出由本会话向本次训练进程组发送 SIGINT 触发；各 rank 报告 `KeyboardInterrupt`，训练退出码与探针退出码均为 1，`g9_run2_target_reached=0`。这是预期的外部中止证据，不是 BF16 OOM、数值错误或数据错误。
- 日志为 `reports/g9/20260812_run2_bf16_probe/run.log`，SHA256 `b67f5d3661927aad94c202ca3ee61fcba7380c4be4146a6d9e4482e0b21b5d33`。输出目录 `outputs/g9/20260812_run2_bf16_probe/` 仅含 resolved config 和 88-byte TensorBoard 事件文件，共约 20 KiB；没有 checkpoint。
- 停止后宿主机确认没有本次 Accelerate/训练 rank 残留。GPU 1、2、3、5、6、7 均为 0 MiB 占用；GPU 0、4 只保留停止前已存在的外部 PID 3246998、3247000，各约 8.7 GiB，本次操作没有终止或改变它们。
- 同时终止了本会话此前误启动、仍在递归遍历 GPFS 的只读 `find -L` 进程组；该进程与训练无关，没有修改文件。
- 当前 BF16 完整 forward/backward/optimizer step 的显存和数值可行性仍未知。现有 run2 输出路径已被占用，运行时启动器的受控停止进程组逻辑也需在任何复用前重新静态检查；不得直接重跑。

#### G9 run3 BF16 单步探针（失败，首个 forward CUDA OOM）

- 用户在修复外部问题后明确要求继续训练。本次仍限定为 8 卡 BF16 的一个完整 optimizer-step 探针：捕获首条 epoch 0/iteration 0 指标后受控停止，不跑完整 epoch；失败不自动修改或重跑，不进入 G10。
- run3 使用 `.runtime/g9_run3_bf16_probe_config.yaml` 和 `.runtime/g9_run3_bf16_probe_launcher.sh`。相对 run2，训练配置只更换了全新的实验名和输出目录；模型、数据、30 帧、512×512、batch size、学习率、五头和解冻范围均未改变。启动器使用物理 GPU 0–7、8 个 NCCL rank、`--mixed_precision=bf16`、offline、本地 Python/ops、30 秒心跳和 7,200 秒 timeout。
- 启动前 GPU 0、4 分别保留另一位用户的 PID 3246998、3247000，各占约 8.7/8.8 GiB；其余 6 张卡为空。用户此前明确批准保留这些服务并共存，本次没有终止或修改它们。8 个训练 rank PID 3401838–3401845 均已启动；rank 0 在数据初始化期间尚未建立 CUDA context，因此早期 `nvidia-smi` 暂时只显示 7 个训练 CUDA 进程，模型构建后 8 张卡全部实际参与。
- 8 个 rank 均完成 7,000 个额外检测序列的 lazy 索引初始化；每 rank 的 8 个 DataLoader worker 曾大量等待 `fuse_get_req`/`wait_on_page_bit_common`，确认首批数据延迟来自共享文件系统 I/O，而不是 NCCL 死锁。完整 train/test DataLoader、967.79M 参数模型、optimizer 和 scheduler 均成功构建，五个任务头全部纳入优化器；epoch 0 进入首个 detection batch。
- BF16 首个 detection forward 未完成。最先确认的 rank 6/GPU 6 和 rank 2/GPU 2 在 SigLIP self-attention 中额外申请 120 MiB 时 OOM；随后 rank 0、1、3、7 也报告同类 OOM。无外部服务的 rank 2/6 当时 PyTorch allocated 均为 73.76 GiB、reserved-but-unallocated 1.93 GiB，整卡只余 44.50/48.50 MiB；rank 0 还受外部服务占用影响，整卡只余 28.94 MiB。rank 4、5 被 Elastic 随故障组终止，不能描述为各自独立完成或 OOM。
- 心跳真实观测的最高整卡占用包括 GPU 0 为 81,061 MiB、GPU 1/3 为 80,979 MiB、GPU 2 为 81,045 MiB、GPU 6 为 81,041 MiB、GPU 7 为 80,991 MiB。GPU 0 的数值包含外部约 8.7 GiB 服务；这些是板级采样值，不等同于各 rank 的 PyTorch 峰值。
- 没有任何 rank 完成首个 detection forward；没有 loss、caption 分支、backward、梯度同步、optimizer step、scheduler step、validation、指标行或 checkpoint。`g9_run3_target_reached=0`、训练退出码 1、探针退出码 1。
- 运行日志为 `reports/g9/20260812_run3_bf16_probe/run.log`，大小 80,063 bytes，SHA256 `5f138b0870b6c865a43c55cb6dcc0a20aba9aec9a2cb950136c94fa49bfd70ef`。输出目录 `outputs/g9/20260812_run3_bf16_probe/` 仅含 resolved config、文本日志和 88-byte TensorBoard 事件文件，没有 checkpoint。
- 启动时间为 `2026-08-12T22:16:16-07:00`，启动器结束时间为 `2026-08-12T22:37:52-07:00`；训练子命令墙钟为 21:07.36，外层峰值 CPU RSS 为 7,736,956 KiB、swap 为 0。该 RSS 不是 8 个 rank 与 64 个 DataLoader worker 的聚合峰值。
- 失败后本次训练、Accelerate 和 DataLoader worker 全部退出。GPU 1、2、3、5、6、7 回到 0 MiB；GPU 0、4 只剩原有 PID 3246998、3247000，各约 8.7/8.8 GiB，外部服务保持存活。没有 fallback、没有自动重跑，也没有修改模型代码。

#### G9 activation checkpointing 最小实现与单层等价测试

- 只修改了本地 `models/siglip2_unisoccer_part_temporal.py`：新增统一的 `Timesformer.forward_block()`，让 image/video 的真实 backbone forward 循环都经过同一入口；仅在训练模式且目标层启用时调用 `torch.utils.checkpoint.checkpoint()`，显式使用 `use_reentrant=False` 和 `preserve_rng_state=True`，并正确传入 `x`、`B`、`T`。
- `UniSoccerBackbone` 新增 `gradient_checkpointing_enable()`、`gradient_checkpointing_disable()` 和 `is_gradient_checkpointing`；现有 `train.py` 的配置开关现在能够控制全部实际 vision blocks。该修改没有增加 parameter 或 buffer，也没有改变 checkpoint/state-dict key 命名。
- 修改后 `models/siglip2_unisoccer_part_temporal.py` 通过 `ast.parse`，工作区通过 `git diff --check`。此时 `USE_GRADIENT_CHECKPOINTING` 仍为 `False`，没有自动改变既有运行配置。
- 用户在重新查看 GPU 后明确批准只在物理 GPU 7 上运行一次真实 SigLIP2 单层 checkpoint 开关等价测试。启动前 GPU 7 为空、约 81,090 MiB 可用；GPU 0、4 各保留另一用户约 8.7/8.8 GiB 的现有服务，本次未触碰。
- 测试使用本地 Python、torch 2.4.1/CUDA 12.1、float32、seed 42、SigLIP2 vision layer 16（含 temporal attention）和输入 `[2,64,1024]`。它没有读取 `epoch_19`、没有创建 dataset、任务头或 optimizer，也没有运行 G9 训练。
- checkpoint 关闭路径没有调用 checkpoint；开启路径恰好调用一次，并确认 `use_reentrant=False`、`preserve_rng_state=True`。两次 loss 均为 `1.0542153120040894`；输出、输入梯度和 25 个参数梯度全部位级一致，最大绝对/相对差均为 0。
- 测试脚本内部耗时 5.311 秒，外层墙钟 6.04 秒，峰值 CPU RSS 1,425,272 KiB；峰值 GPU allocated/reserved 分别为 241,337,856/260,046,848 bytes。退出码为 0。
- 日志：`reports/g9/20260812_checkpoint_equivalence_gpu7.log`，SHA256 `3af40945b4a25568d23c1e59c47a3b5976458660891c32a433d3cb5abcf08f15`。运行记录了 CuBLAS 和 memory-efficient attention 的非确定性 warn-only 警告，因此该结果只证明本次单层输入下两条路径位级一致，不构成完整训练或跨运行严格确定性保证。
- 结束后 GPU 7 回到 0 MiB，只有 GPU 0、4 的两个原有外部进程继续存在；本次没有遗留 GPU 进程。尚未测量 30 帧、512×512 全模型启用 checkpoint 后的显存，也没有完成 G9 optimizer step，G9 仍未通过。

#### G9 run4 checkpointing BF16 单步探针静态准备

- 按状态账本的唯一下一步，只在本地完成静态准备；没有导入 Torch 或项目模块，没有读取模型/checkpoint，没有使用 GPU，也没有启动 forward、backward、optimizer 或训练。
- `train.py` 的现有配置开关后新增 fail-fast 机器断言：启用后必须满足 vision block 数量大于 0、`checkpoint_num == len(resblocks)` 且 `is_gradient_checkpointing=True`；每个 rank 输出一条 `[GRADIENT_CHECKPOINTING_ASSERT]`，包含 rank、enabled/total blocks 和通过状态。目标模型预期必须报告 24/24。
- 新配置为 `.runtime/g9_run4_checkpointing_bf16_probe_config.yaml`。与 run3 配置逐键比较，唯一变化是新增 `USE_GRADIENT_CHECKPOINTING=True`，并使用全新的 `OUTPUTS_DIR` 和 `EXP_NAME`；30 帧、512×512、detection batch 1、Caption batch 2、五头、数据范围、学习率、解冻范围和 SigLIP2 路径均未改变。
- 新启动器为 `.runtime/g9_run4_checkpointing_bf16_probe_launcher.sh`。目标仍是物理 GPU 0–7、8 个 rank、BF16，只在首条 epoch 0/iteration 0 `[Metrics]` 写出后受控发送 SIGINT；该指标在代码中位于 detection 与 Caption 两次 backward、`optimizer.step()`、`zero_grad()` 和 global-step 更新之后。
- 启动器要求报告中出现 8 个唯一 rank 的 `enabled_blocks=24 total_blocks=24 active=True passed=1` 标记，同时首个 optimizer-step 指标必须存在，二者同时满足才允许探针退出码为 0。失败不自动修复或重跑。
- timeout 由 run3 的 7,200 秒增加为 14,400 秒，kill-after 为 120 秒；保留 30 秒心跳、每卡显存/利用率、磁盘状态、offline、本地 Python/ops、显式 `PYTHONPATH`/`LD_LIBRARY_PATH` 和明确退出码。
- Python 文件通过 `ast.parse`，启动器通过 `bash -n`，YAML 通过安全解析和三字段差异白名单检查，必需 Python/ops/config/data-view/14,433,760,820-byte extra-data 索引/SigLIP2 路径均存在；新 report log 与 output 目录均未被占用，`git diff --check` 通过。
- `.runtime/` 由 Git 永久忽略；可提交的本地源码变化新增 `train.py`，原始远端仓库和共享环境没有修改。run4 仍未运行，完整 checkpointing 显存收益和一个 BF16 optimizer step 仍未知，G9 状态不变。

#### G9 run4 checkpointing BF16 单步探针（通过）

- 用户在查看实时 GPU 状态和共存风险后明确批准只运行一次 run4：物理 GPU 0–7、8 个 rank、BF16、24-block activation checkpointing，最多 14,400 秒，只完成首个 detection + Caption optimizer step 后受控停止；失败不自动修复或重跑，不进入 G10。
- 启动前 GPU 0、4 分别保留另一用户 PID 3246998、3247000，各占约 8.7/8.8 GiB；其余 GPU 为空。所有外部进程均被保留，本次没有终止或修改它们。
- resolved config 确认 `USE_GRADIENT_CHECKPOINTING=true`、30 帧、512×512、detection batch 1、Caption batch 2、vision 全部解冻、text 全部冻结；总参数 967.79M、可训练参数 402.22M、五个任务头全部可训练，与 run3 的训练语义一致，唯一算法执行适配是 activation checkpointing。
- 8 个 rank 均输出 `enabled_blocks=24 total_blocks=24 active=True passed=1`，机器断言通过。8 个 rank 均完成 7,000 条额外检测序列 lazy 初始化，完整 train/test DataLoader、模型、optimizer 和 scheduler 均成功构建；detection/Caption DataLoader 长度仍为 12,007/11,410。
- epoch 0/iteration 0 完整处理 detection 和 Caption 两个 dataset 分支，覆盖 `SoccerNetGSR_Detection`、`LinesDetection`、`KeypointsDetection`、`VideoCaption`、`CaptionClassification` 五头；两次 backward、梯度裁剪、DDP 同步、`optimizer.step()`、`zero_grad()` 和 global-step 更新均在首条 `[Metrics]` 之前完成。
- 首步 weighted total loss 为 1221.8020。五头关键 weighted loss 均为有限值：Detection 的 ce/bbox/giou 为 2.1724/0.7928/1.7529，Lines 为 0.2636，Keypoints 为 1170.9922，VideoCaption 为 35.4189，CaptionClassification 为 4.7854。
- backbone 和五头梯度 norm 均为有限正数：backbone 163.5036，CaptionClassification 52.0486，KeypointsDetection 193.5343，LinesDetection 0.1680，SoccerNetGSR_Detection 7.0903，VideoCaption 6.0633。
- rank 间 gather 的 PyTorch 峰值 allocated 为 17,800.998 MiB（约 17.38 GiB/rank）。心跳观测的最高整卡 used 为 GPU 0–7：29,721、24,637、22,891、25,571、29,649、21,297、24,359、24,175 MiB；GPU 0、4 的板级数值包含各自约 8.7/8.8 GiB 外部服务。相比 run3 在空闲卡约 73.76 GiB allocated 后 OOM，checkpointing 已确认显著降低 activation 显存。
- 首步指标写出后启动器按协议发送 SIGINT。4 个 rank 以 `KeyboardInterrupt` 退出，另 4 个已经进入下一批数据/Caption forward、未及时响应 SIGINT，由 Elastic 随后用 SIGKILL 清理；训练子命令因此退出 1。这些 traceback 发生在目标达成后的受控停止阶段，日志中没有 OOM、CUDA RuntimeError 或 NCCL 错误。
- 启动器机器结果为 `g9_run4_target_reached=1`、`g9_run4_checkpoint_assert_rank_count=8`、`g9_run4_train_exit_code=1`、`g9_run4_probe_exit_code=0`。开始/结束时间为 `2026-08-12T23:09:56-07:00` 至 `2026-08-12T23:28:02-07:00`；训练子命令墙钟 17:54.54，外层探针约 18:06，峰值外层 CPU RSS 7,712,712 KiB、swap 为 0。
- 主日志：`reports/g9/20260812_run4_checkpointing_bf16_probe/run.log`，70,441 bytes，SHA256 `cb2c9c38644123d1f30365fab44a441a014cc6588050be7579d7fde4db9d18d7`。训练日志 SHA256 为 `910cb472664ea07b8f9f7e43e6a948e4d2e5870a89acc0b09f62549e5bf4184a`，resolved config SHA256 为 `09e25be76c49864675cd3e3a3470e5a7b08cad294f7eac7a5a2027314ccc64b7`。
- 输出目录只包含 resolved config、文本日志和 TensorBoard 事件文件，共约 36 KiB；探针没有保存 checkpoint。结束后 GPU 1、2、3、5、6、7 回到 0 MiB，GPU 0、4 只剩原有两个外部进程，没有 run4 残留进程。
- run4 证明当前 8×H800 能以 BF16 + activation checkpointing 完成真实完整多任务的一次 optimizer step。这是适配后的单步硬件/数值可行性通过，不是原始 `USE_GRADIENT_CHECKPOINTING=False` 执行方式的复现，也不证明多步稳定性、完整 epoch、evaluation、checkpoint/resume 或 20-epoch 完整训练；因此 G9 仍未通过。

#### G9 run5 六卡三步观察运行（通过，非完整Gate）

- 2026-08-19，用户在查看GPU 0–7全部为空、每张空闲81,090 MiB且无compute process后，明确授权物理GPU 0–5运行3个真实optimizer steps，保留GPU 6–7给其他验证；明确禁止checkpoint、evaluation、完整epoch和自动重跑。启动前最终快照条件不变，在独立tmux `g9_run5_six_gpu_observation:train_3steps`启动。
- `train.py`新增默认关闭的`MAX_OPTIMIZER_STEPS`：只在配置显式给出正整数时，于目标optimizer step完成并记录metrics后同步、干净返回；上层随后跳过scheduler、evaluation和checkpoint。固定配置和启动器分别为`.runtime/g9_run5_six_gpu_3step_observation_config.yaml`和`.runtime/g9_run5_six_gpu_3step_observation_launcher.sh`。
- 六个rank均确认`enabled_blocks=24 total_blocks=24 active=True`。真实30帧、512×512、五头训练完成3/3步，两类dataset分支每步均完成forward、backward、梯度裁剪、DDP同步和optimizer更新；weighted total loss依次为1206.7188、1132.1456、1165.8433，backbone及五头梯度范数均为有限正数。
- PyTorch峰值allocated为20,886.2363 MiB/rank，30秒心跳观测的最高整卡used为26,389 MiB；训练子进程墙钟384.38秒，外层约392秒。训练和观察启动器退出码均为0，未发生OOM、fallback、自动重跑、scheduler step、evaluation或checkpoint写入，`epoch_*`目录数为0。
- 日志中保留既有球场几何NumPy invalid-value warning、`torch.meshgrid`兼容性warning和退出时PyTorch 2.4未显式destroy process group warning；它们没有造成非有限loss、worker失败或资源残留。结束后tmux自动关闭，GPU 0–7均回到0 MiB、0%利用率且无compute process。
- 机器结果和主日志位于`reports/g9/20260819_run5_six_gpu_3step_observation/`，TensorBoard事件位于`outputs/g9/20260819_run5_six_gpu_3step_observation/logs/`。本轮只把G9证据从“单步可行”推进为“六卡三步可行”；六卡全局batch与目标八卡基线不同，且仍未验证完整epoch、evaluation、checkpoint/resume、收敛或最终效果，因此G9完整训练仍未通过。

#### G9 run6 六卡单epoch长跑（超时失败）

- 用户在了解六卡一个epoch约16,009 steps、预计10–13小时、约60–80 GPU-hours、跳过evaluation、保存单个约8–12 GiB checkpoint及文件系统剩余约156 GiB后，明确要求执行；在最终fresh快照显示GPU 0–7均空闲81,090 MiB、利用率0%且无compute process后，再次确认物理GPU 0–5启动，GPU 6–7保留。
- 本地`train.py`新增默认关闭的`MAX_EPOCHS_THIS_RUN`和`SKIP_EVALUATION`，并把epoch checkpoint改为同文件系统临时目录、rank0单写者、文件大小manifest、`COMPLETE`标记和`os.replace`原子落盘；写入失败会广播失败并保留临时证据，不自动重跑。CPU假模型smoke已实际验证完整落盘及拒绝覆盖。
- 固定配置和启动器为`.runtime/g9_run6_six_gpu_one_epoch_config.yaml`与`.runtime/g9_run6_six_gpu_one_epoch_launcher.sh`；BF16、24-block checkpointing、30帧、512×512、每卡detection batch 1/Caption batch 2、目标20-epoch scheduler语义保持不变，本次只限制执行epoch 0、每100步记录一次并跳过evaluation。timeout为64,800秒，30秒heartbeat，输出位于`outputs/g9/20260819_run6_six_gpu_one_epoch/`，日志位于`reports/g9/20260819_run6_six_gpu_one_epoch/run.log`。
- 任务于2026-08-18 23:15:59 MST在tmux `g9_run6_six_gpu_one_epoch:train_epoch0`启动。六个rank均报告`enabled_blocks=24 total_blocks=24 active=True passed=1`，并进入`Training epoch 0 with 16009 iterations`。最后一条正式指标为step `8299/16009`（约51.8%）；运行期间未出现OOM或模型异常，日志中的峰值CUDA allocated约20.883 GiB/rank。
- 2026-08-19 17:15:59 MST达到64,800秒外层timeout后收到SIGTERM；训练子命令退出124，外层one-epoch结果退出1。`COMPLETE`、训练状态和checkpoint manifest计数均为0，没有epoch checkpoint或可恢复的中途checkpoint。该运行证明六卡配置可以稳定长时间训练超过8,000步，但按Harness不能判定完整epoch、checkpoint或G9通过。
- 按已观察速度推算，完整epoch约需35–37小时；由于当前保存点只在epoch结束写出，不能从step 8,299继续。后续若重启长跑，应先决定是接受从头重跑，还是实现并验证安全的mid-epoch checkpoint/resume；本轮失败不自动重跑。

### G10 提前启动与 G10-A 数据质量审计

- 2026-08-13，用户考虑到 checkpointing 适配版 G9 完整 20 epochs 预计耗时数周，明确批准暂停扩展 G9，并在独立边界内正式提前进入 G10。
- 该顺序例外已经写入 `AGENTS.md` 和 `docs/HARNESS.md`。它不改变 G9 结论：G9 仍为“单步可行、完整训练未通过”，G10 不得替代 G9 或声称优于尚未完成的完整 G9 基线。
- 2026-08-12 完成的只读 SoccerFactory 伪标签审计现登记为 G10-A 已有证据。人类可读报告为 `reports/audits/soccerfactory_data_quality_20260812.md`，机器可读摘要为 `reports/audits/soccerfactory_data_quality_20260812.json`。
- G10-A 覆盖前 7,000 个实际训练 PKL 的完整小框统计、固定随机 200 段的号码/角色/相机审计，以及用户参与的可视检查；源数据、原始代码和伪标签全程只读，没有使用 GPU、推理或训练。
- 已确认的主要问题包括：足球被框为人物并标成 `other`；高可读性错误号码在相邻帧传播；普通球员/裁判/门将角色错标在相邻帧传播；相机参数存在无效帧、伪有效极值及明显投影跳变。
- 审计同时确认最终抽查的前 10 个主转播样本无镜头污染，多数人物框和多数人工查看的号码/角色样本可用。由于人工样本有限，不能把人工错误比例外推为全数据集错误率。
- G10-A 的历史机器摘要仍准确记录其执行当时为 `read_only_pre_g10_audit_not_a_gate_result`；本次只把该不可变历史证据纳入后来批准的 G10-A，不把它改写成当时已经通过正式 Gate。
- G10 当前整体状态为“进行中”；G10-B 生成链路、G10-C 训练接口、G10-D 小规模受控实验和 G10-E 扩大规模均未执行。没有形成整个 G10 的通过/失败结论。

#### G10-B 只读设计审查与兼容转换静态验证

- 只读设计审查确认，保存下来的生成主干跨越 TrackLab、`sn-gamestate` 和 Refiner：准备好的 `sn500` 图片由 TrackLab Step 1 生成检测/ReID/StrongSORT 状态，Refiner 生成 `refined_sn-gamestate.pklz`，TrackLab Step 3 加入球场注册、相机、号码、角色和轨迹级聚合，最后再转换为逐序列训练 PKL。
- 当时的只读设计审查尚未找到“原始比赛视频/镜头标签到 `sn500` 图片目录”的实现，也没有找到数据加载器提示的 `split_extracted_info.py`。历史映射中 `SNGS-10004` 的区间 `14439–14645` 按首尾包含207个索引，而实际准备图片为255张，因此当时不能反推或宣称原始切片规则已恢复；2026-08-19 已定位到候选脚本，但其编号映射与该旧 manifest 矛盾，见后文来源核对。
- 固定极小样本为 `SNGS-10004`：255帧、准备图片122,481,437 bytes。固定 manifest 为 `reproduction/manifests/g10_soccerfactory_sngs10004.json`；供后续 Refiner 单序列运行使用的 metadata 为 `reproduction/manifests/g10_sngs10004_refiner_metadata.json`。
- 新增本地兼容转换入口 `reproduction/gates/g10_soccerfactory_convert.py`。它只读取历史 Step-3 ZIP 中的 `10004.pkl` 和 `10004_image.pkl`，按已确认契约生成 `people/K/R/P/valid_cam_params`；所有输出限定在工作区，拒绝覆盖既有文件。该入口明确是对缺失历史脚本的兼容恢复，不冒充原作者源码。
- 2026-08-13 在 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1` 和共享参考 Python 下执行一次 CPU 验证，120秒 timeout 内外层退出码0；`/usr/bin/time` 墙钟8.05秒、峰值CPU RSS 189,496 KiB、swap 0。
- 兼容输出包含255帧、4,260个人物实例和255个有效相机帧。与黄金 `extracted_info/SNGS-10004.pkl` 逐帧比较，人物标量/框不一致数为0，相机矩阵最大绝对差为 `2.9103830456733704e-11`，低于预设 `atol=1e-10, rtol=0`；4项机器断言全部通过。
- 本地输出与黄金文件大小均为435,540 bytes，但 SHA256 不同，因此只判定字段和数值语义兼容，不判定位级 pickle 复刻。机器结果为 `reports/g10/20260813_static_conversion/result.json`，日志为同目录 `run.log`，人类报告为同目录 `README.md`。
- 本子阶段没有使用 GPU，没有运行 TrackLab、Refiner、模型推理、评估或训练，也没有修改原始代码、数据、权重或历史状态。它没有验证新的 Step 1/Refiner/Step 3 产物、模型生成结果的独立重复性或 SoccerMaster 训练读取契约；G10-B 和整个 G10 均仍未通过。

#### G10-B TrackLab Step 1 运行前静态预检（通过）

- 按上一条唯一下一步，为固定样本 `SNGS-10004` 新增本地隔离配置 `reproduction/configs/g10/g10_step1_sngs10004.yaml`、固定运行清单 `reproduction/manifests/g10_soccerfactory_step1_sngs10004.json` 和受保护入口 `reproduction/gates/g10_soccerfactory_step1.py`。目标流水线只包含 `bbox_detector -> reid -> track`，固定255帧、YOLO person、PRTReID、HRNet backbone、4个CPU workers、各模块batch 16，关闭评估、可视化和W&B。
- 未来单卡命令、工作目录和输出位置已固定；Hydra输出只能写入全新的 `.runtime/g10/sngs10004_step1/run1`。入口禁止覆盖既有输出，实际运行要求 `G10_STEP1_GPU_APPROVED=YES` 和恰好一个数字形式的 `CUDA_VISIBLE_DEVICES`，使用3,600秒 timeout、30秒心跳、无自动 fallback/重试，并在超时时先终止整个进程组、60秒后仍未退出才强制结束。
- 运行后机器断言已预先固定：状态ZIP成员必须且只能是 `summary.json`、`10004.pkl`、`10004_image.pkl`，无重复成员且CRC通过；图像表必须有255行、frame严格为0至254、文件名严格为 `000001.jpg` 至 `000255.jpg`；检测表非空，必须含检测框、置信度、ReID embedding/visibility和非空 `track_id`，且图像/视频外键一致。失败时保留局部产物供诊断，retry必须换新run ID。
- 2026-08-13 使用固定 Python `/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`，在 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1` 下执行默认静态预检；外层120秒 timeout，退出码0，脚本墙钟5.980秒。预检确认255张图片共122,481,437 bytes且首尾SHA不变、3份模型文件大小不变、10个上游契约文件存在、6个包的定位符合目标环境、命令与本地路径契约一致，Hydra输出和未来run1报告目录均未占用。
- 机器结果为 `reports/g10/20260813_step1_preflight/result.json`，SHA256 `2f93958be7422b3f7c319d562a3d85a121005a6084be87e54e36a38580817558`；静态预检日志为同目录 `run.log`，明确记录退出码0。配置、manifest和入口SHA256分别为 `adef0a8e683cb6b59bd29246b633294a8bf7ede1830362b5d7913caa93bd5a04`、`20cfb0917948041f9f51d267e91f465d9e64be21e886fd9148f88fc41c9d63ee`、`3c21873b935f7025d36842651c3481408335ca159c21d1db8cc5b820895b2749`。
- 本次只使用 YAML/JSON/AST/路径/大小/有限哈希检查和 `importlib.util.find_spec` 定位包，没有导入或启动 TrackLab、Torch或模型代码，没有执行 `nvidia-smi` 或任何GPU操作，也没有推理、评估、训练或创建Hydra运行目录。Hydra实际组合、模块导入/ABI、GPU容量、模型forward、ZIP生成和跟踪质量仍未知；因此这只是运行前静态预检通过，不是 Step 1 或 G10-B 通过。

#### G10-B TrackLab Step 1 run1（失败，共享存储/数据实例化阶段超时）

- 启动前重新执行 `nvidia-smi`：物理GPU 0–7均为NVIDIA H800，每卡总显存81,559 MiB、可用约81,090 MiB、占用0 MiB、利用率0%，无计算进程。用户随后明确批准仅使用物理GPU 0运行一次固定255帧Step 1，3,600秒timeout、30秒心跳，失败不自动重试，不运行评估、可视化、Refiner、Step 3或训练。
- run1 使用 `CUDA_VISIBLE_DEVICES=0`、固定共享Python、固定manifest/config和受保护 `--mode run` 入口。静态预检再次通过后启动 TrackLab 子进程；外层持续输出30秒心跳，没有改参数、fallback或额外run。
- 子进程绝大多数时间处于Linux不可中断 `D` 状态，内核等待点为 `cxiWaitEventWait`，CPU累计时间远低于墙钟，说明主要等待共享文件系统/网络I/O而非计算。人工GPU采样中物理GPU 0始终为0 MiB、利用率0%，模型推理未开始。
- 约启动20分钟后，日志首次写出 `Using device: 'cuda'` 和完整composed config快照（部分OmegaConf插值仍按原表达式保存）；这确认Python/TrackLab导入、Hydra组合和CUDA可用性判断已经完成。composed config快照的本地SHA256为 `4c805cb4ddc24d016cd64a43d179b8c1f731e52cc762f50d0fb35e9a49280eff`。
- 此后没有新的阶段日志。上游 `tracklab.main` 在 `init_environment()` 后依次执行 `instantiate(cfg.dataset)`、evaluator和三个模型模块，直到全部完成才记录“Starting tracking”；GPU始终0 MiB表明未进入GPU驻留检测器或forward，但现有日志无法在dataset、evaluator与早期模块导入之间精确分界。阻塞在 `SoccerNetGameState` 数据集实例化及NAS目录/JSON读取是较强推断，具体系统调用或文件仍未知，不能写成已确认根因。
- 3,600.6秒时Harness按协议触发timeout并向进程组发送SIGTERM；子进程退出码 `-15`，机器结果 `timed_out=true`、`assertions_passed=false`、`outcome=failed`，外层命令退出码241。Harness记录的峰值子进程RSS为638,636 KiB。失败类别机器字段为较宽泛的 `distributed_order_or_timeout`；本次没有分布式运行，实际观测是单进程启动/实例化I/O超时。
- 机器结果为 `reports/g10/20260813_step1_run1/result.json`，4,738 bytes，SHA256 `895e1e4e7cec83a3a6d62b025d2290f3e50cca29b35a228377669ec782bd0b49`；主捕获日志为同目录 `run.log`，6,971 bytes，SHA256 `516fc778fe5c7b1bbaf109679a370f950e4ffcdc248518722b2a8ac3b35214e3`。本地Hydra目录共25,306 bytes，其中 `main.log` SHA256为 `0dbd8ab3e4fb0d1e928703e1cbba8cf3b6f61888315147f4545edc28a8b0931a`。
- 没有生成 `states/sn-gamestate.pklz`，因此检测、ReID、跟踪和ZIP成员/表结构断言均未执行。结束后确认无本次TrackLab/Harness残留进程，8张GPU均为0 MiB、利用率0%；没有修改原始代码、数据或权重，也没有自动重跑。Step 1、G10-B和整个G10均未通过。

#### G10-B Step 1 分阶段启动诊断 run1（失败，诊断Harness配置错误）

- 按唯一下一步新增CPU/只读诊断入口 `reproduction/gates/g10_soccerfactory_step1_diagnose.py` 和固定manifest `reproduction/manifests/g10_soccerfactory_step1_diagnosis_sngs10004.json`。设计把 `import tracklab.main`、Hydra compose、`instantiate(cfg.dataset)` 分成独立事件和timeout，固定 `CUDA_VISIBLE_DEVICES=''`，禁止调用TrackLab main、evaluator和三个模型模块，禁止读取模型权重或运行推理/评估/训练。
- 静态审查同时发现宿主机有224个逻辑CPU，而上游 `load_set()` 使用无参数 `multiprocessing.Pool()`，会默认尝试约224个进程。诊断预先声明只在子进程内临时把该Pool限制为4个worker，与目标config的 `num_cores: 4` 一致；不修改上游文件，并明确记录这不是原生产并发语义。本次失败发生在此前，Pool适配没有实际执行。
- run1使用新的 `reports/g10/20260813_step1_startup_diagnosis_run1/`，外层timeout 3,600秒、30秒心跳。`import tracklab.main` 成功，耗时203.570秒，确认共享环境真实导入延迟可单独观测；没有GPU操作。
- 随后的Hydra compose在2.879秒后抛出 `MissingConfigException`：诊断入口使用 `initialize_config_dir()` 时只有本地配置和 `sn_gamestate` 插件搜索路径，遗漏真实Step 1 CLI具有的 `pkg://tracklab.configs`，因此找不到TrackLab自带的 `modules/track/bpbreid_strong_sort`。真实Step 1 run1已经对同一配置成功组合，所以这是诊断Harness构造错误，不是目标配置错误。
- 子进程退出后，父进程构造机器结果时又把Python布尔值写成JSON风格的 `true`，触发 `NameError` 并使外层退出码为1。原Harness没有生成result；随后根据不可变 `events.jsonl` 和 `run.log` 补写 `result.json`，字段 `result_origin` 明确标为重建失败摘要，不能冒充原Harness成功落盘。
- 事件文件SHA256为 `43dc3595acf863343e16df30b5839a619d2a08a28cff6532e4dee2025603d5e9`，日志SHA256为 `3b3971f0099753c47fcd245a0c6ca1f48d21d421fe28efe7a7d5f4b786139640`，重建结果SHA256为 `725a3207ce0c8129123351225c56c43d06ef839677037024e834675cd33f703f`。数据集实例化没有开始，因此本次没有缩小NAS数据读取边界。
- 诊断脚本已修正为显式在本地配置之后加入 `pkg://tracklab.configs`，并修正 `True` 拼写；修正后脚本AST、manifest/result JSON和 `git diff --check` 均通过。修正脚本SHA256为 `2c18b378449e9c0819329342698ea68c4ed9b98d3aace14e89c73318f88fa116`。本轮没有自动执行run2，retry必须使用全新run ID和report目录。

#### G10-B Step 1 分阶段启动诊断 run2（失败，沙箱OpenMIM缓存写入）

- 使用全新manifest `reproduction/manifests/g10_soccerfactory_step1_diagnosis_run2_sngs10004.json` 和report目录 `reports/g10/20260813_step1_startup_diagnosis_run2/`。启动前显式构造并断言Hydra搜索路径顺序包含本地config、`pkg://tracklab.configs`、`pkg://sn_gamestate.configs`；run1遗漏路径已修复，report目录未占用，AST/JSON和 `git diff --check` 通过。
- run2固定 `CUDA_VISIBLE_DEVICES=''`、30秒心跳、分阶段timeout和4-worker诊断适配。外层墙钟357.410秒、worker退出码1、未超时、峰值子进程RSS 653,692 KiB；机器结果正常落盘，确认run1的父级结果构造错误已修复。
- `import tracklab.main` 用时2.952秒并通过；Hydra compose用时0.623秒并通过。固定pipeline为 `bbox_detector -> reid -> track`，split为 `sn500`、`nframes=255`、sequence为 `SNGS-10004`、dataset target为 `tracklab.wrappers.SoccerNetGameState`，五项契约均符合预期。与run1的203.570秒导入相比，本次共享文件缓存已热；两次耗时差异不能外推为稳定性能。
- `instantiate_dataset` 在351.161秒后失败。完整traceback确认失败尚在导入 `tracklab.wrappers`：其 `__init__` 连带导入tracklet聚合与 `tracklab.utils.openmmlab`，OpenMIM导入时无条件尝试创建 `/home/tianlin/.cache/mim`；受控沙箱把home根目录设为只读，最终抛出 `OSError: [Errno 30] Read-only file system`。
- 因此本次没有进入 `SoccerNetGameState.__init__`，也没有创建上游 `Pool()`；NAS上的固定单序列JSON/图片读取速度和正确性仍未验证。机器结果把失败宽泛分类为 `data_contract`，但直接证据支持的实际类别是诊断环境缓存写入，不是数据契约或数据损坏。
- 全程没有GPU操作、模型权重读取、evaluator/检测器/ReID/tracker实例化、推理、评估或训练。机器结果SHA256为 `4793d5e3c2ffb4ea9d15a87471adcf98e3c9487cdd955b32d9e26bbdb204a478`，事件SHA256为 `3b369f2804b5162cda61d42e1d266269f618e3502e92ca05f3d3394041ee7d96`，日志SHA256为 `d4d42cd1202a0d5922020971a09b9518c6a89fe82da8ee248bb427cf374177e5`，manifest SHA256为 `d208df4fbb347746034d40fb106d6e9a3c85dd4ed0d6b53d63affc5c2044d882`。
- 诊断入口随后新增局部、显式缓存适配：未来manifest必须指定工作区内 `.runtime/` cache供Matplotlib使用；只在导入wrapper的短暂窗口内拦截且仅拦截OpenMIM创建 `/home/tianlin/.cache/mim`，随后恢复原 `os.makedirs`。它不修改或重定义 `HOME`，不修改共享环境/上游文件，不吞掉其他路径的写入。修正后脚本AST与 `git diff --check` 通过，SHA256为 `c1188b090a25764b9a86cc39bc52a4b7c5e1e53ece79bfb34faccfd5c694823f`；本轮没有自动run3。

#### G10-B Step 1 分阶段启动诊断 run3（通过）

- 使用全新manifest `reproduction/manifests/g10_soccerfactory_step1_diagnosis_run3_sngs10004.json`、report目录 `reports/g10/20260813_step1_startup_diagnosis_run3/` 和cache目录 `.runtime/g10/step1_startup_diagnosis_run3_cache/`。启动前静态断言Hydra三层搜索路径正确、OpenMIM拦截只精确匹配 `/home/tianlin/.cache/mim`、`HOME`未被修改、新report/cache路径未占用；AST/JSON与 `git diff --check` 通过。
- run3固定 `CUDA_VISIBLE_DEVICES=''`、30秒心跳、独立阶段timeout、4-worker Pool适配和局部缓存适配。外层墙钟1,519.752秒，worker退出码0、未超时、峰值子进程RSS 664,608 KiB；全部阶段与最终机器断言通过，`outcome=passed`。
- `import tracklab.main` 耗时767.855秒；Hydra compose耗时7.269秒。compose再次确认pipeline为 `bbox_detector -> reid -> track`、split为 `sn500`、`nframes=255`、sequence为 `SNGS-10004`、dataset target为 `tracklab.wrappers.SoccerNetGameState`。
- `instantiate(cfg.dataset)` 耗时729.441秒并通过。最终dataset仅包含 `sn500`，video rows为1、image rows为255、sequence严格为 `SNGS-10004`，frame严格为0至254；该固定split无ground-truth detection表，记录为null而非错误。
- 关键性能边界：dataset阶段日志中真正由4-worker Pool执行的单视频 `video_dir_to_dfs` 进度只用3.67秒。因此729.441秒的大头不是读取该255帧/单序列元数据本身，而是在Pool开始前的共享 `tracklab.wrappers` 依赖导入与Matplotlib字体缓存初始化。Step 1 run1的一小时等待不能再笼统归因为NAS样本数据读取。
- 局部适配按声明生效：上游无参数Pool仅在诊断子进程中限制为4 workers；OpenMIM home cache创建仅在wrapper导入窗口被精确拦截，之后恢复原 `os.makedirs`；Matplotlib只在工作区生成30,531-byte `fontlist-v330.json`，cache目录总计38,723 bytes；`HOME`保持 `/home/tianlin`。没有修改远端代码、共享环境、数据或权重。
- 全程没有GPU操作、模型权重读取、evaluator/检测器/ReID/tracker实例化、推理、评估、可视化或训练。诊断通过只证明CPU启动、Hydra和dataset契约；不证明Step 1模型链路或状态ZIP。
- 机器结果SHA256为 `40bd56772607875a19e937cc5549a515fa5e46fe20e4597b6172bad1db6fc19b`，事件SHA256为 `926203b5f2d49c878b122a59414863d23fa88f18f525ac9aed7c57e7fa74bb38`，日志SHA256为 `412dababb35c571da079849753bb8d983e1975f3ff9ce75284d1808b3dbe9f00`，manifest SHA256为 `6c8a570ecca589b1f61d0a7cf35818b87747e372b7f0d47d150d185be08ccff2`，执行入口SHA256为 `f1cabd6da7504c167ac8aaa574cd9f9162ee8f6ddf9b7180082f29f03602f7e4`。结束后无诊断进程残留。

#### G10-B TrackLab Step 1 run2 分阶段Harness静态预检（通过，尚未运行）

- 新增固定配置 `reproduction/configs/g10/g10_step1_sngs10004_run2.yaml`、manifest `reproduction/manifests/g10_soccerfactory_step1_run2_sngs10004.json`、受保护父入口 `reproduction/gates/g10_soccerfactory_step1_run2.py` 和分阶段worker `reproduction/gates/g10_soccerfactory_step1_run2_worker.py`。YAML逐键归一化比较确认，相比run1只改变 `experiment_subname`、Hydra run目录和sweep目录；模型、权重、255帧固定样本、batch、阈值、pipeline、关闭评估/可视化/W&B等语义未变。
- worker保留上游 `tracklab.main` 主流程，并为13个边界记录严格有序的started/passed事件：TrackLab导入、Hydra、CUDA环境、dataset、evaluator、YOLO、PRTReID、StrongSORT、Pipeline、TrackerState、engine、255帧track和明确跳过evaluation。每阶段有独立timeout，整体上限10,800秒、30秒心跳，超时先终止进程组再等待60秒强制结束；任何失败不重试、不fallback、不覆盖run1。
- 权重读取边界已经固定：YOLO阶段声明读取195,209,883-byte detector；PRTReID阶段声明读取396,287,605-byte checkpoint和165,587,602-byte HRNet backbone；evaluator和StrongSORT没有额外权重。模型实例化后还会断言 `training_enabled=false`，成功退出后必须同时通过完整事件顺序和状态ZIP成员、255行image、检测列、非空track id等契约。
- 复用run3的两项局部适配：dataset上游无参数Pool只在worker内限制为4 workers；OpenMIM仅在wrapper导入窗口精确忽略 `/home/tianlin/.cache/mim` 创建，Matplotlib缓存固定到新的工作区run2 cache，随后恢复原函数。适配不改变HOME、模型、权重、输入或算法参数。
- 最终静态预检位于 `reports/g10/20260813_step1_run2_preflight_final/`，墙钟5.557秒、退出码0。它确认固定255帧共122,481,437 bytes、旧Step 1 manifest哈希、四个目标类、三份权重路径/大小、源码AST、配置等价性、13阶段及timeout、未来Hydra/report/cache路径未占用。只读取YAML/JSON/Python源码和文件元数据；权重内容未打开，TrackLab/Torch/模型未导入，没有执行 `nvidia-smi`、GPU初始化、推理、评估、可视化或训练。
- 第一次预检证据保留在 `reports/g10/20260813_step1_run2_preflight/`；其后静态增加“模型不得启用训练”和“成功必须具有完整事件序列”两项断言，所以没有覆盖旧结果，而是使用全新final目录重跑。最终result SHA256为 `20e077148fbc99578d48b3bb6180fd15ee58e8be0f23fb6f3fc04e8fc3ab92e9`，manifest为 `5d190065eda972c6de42a25b83499f58541cdbbd02d4f0f641e71bd1437fe3ae`，父入口为 `993042334cf68768b243338b73363d8d8dc4d4a8121f33dbdc7c88318763b402`，worker为 `3b7ce8878584c4f88e3b05f366079230317fdcef7330e248f4359de751605a69`。
- 本子阶段只证明run2运行设计与静态输入契约已经准备好。evaluator/模型真实导入和权重反序列化、CUDA/ABI、显存、forward、检测/ReID/跟踪结果、状态ZIP与质量仍未知；Step 1、G10-B和整个G10均未通过。

#### G10-B TrackLab Step 1 run2（失败，执行沙箱未暴露CUDA）

- 用户在查看8张卡的实时状态后明确批准当前主机物理GPU 0。启动前宿主 `nvidia-smi` 显示8张均为NVIDIA H800，每张总显存81,559 MiB、已用0 MiB、可用81,090 MiB、利用率0且无计算进程；这与用户口述的H200不同。
- run2固定物理GPU 0、共享参考Python、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、固定SNGS-10004的255帧、30秒心跳、分阶段timeout和10,800秒总上限；评估、可视化、W&B和训练关闭，声明失败不重试。
- 正式启动前静态预检再次通过。worker事件确认 `import_tracklab_main` 用时309.889秒、Hydra compose用时3.676秒、`init_environment` 用时0.130秒并依次通过；随后进入 `instantiate_dataset`。
- 运行命令误在Codex受限文件系统沙箱中启动。相同沙箱、相同Python和 `CUDA_VISIBLE_DEVICES=0` 的独立探针直接返回 `torch.cuda.is_available()=false`、`device_count=0` 并告警无法初始化NVML；因此继续到模型实例化只会产生由执行隔离导致的伪CUDA失败。
- 在模型实例化前人工中止，外层退出码130。停止时dataset实例化尚未完成；没有实例化evaluator/YOLO/PRTReID/StrongSORT，没有读取权重内容，没有GPU显存分配、推理、评估、训练或状态ZIP。没有fallback或自动重试。
- Harness未到达原子结果写入；根据事件流、日志、退出码和CUDA探针补写 `reports/g10/20260813_step1_run2/result.json`，字段明确标记为重建失败摘要，不能冒充Harness成功结果。事件和run日志SHA256分别为 `dd111795c1fed1dbe0d5f4a2447a2f8040bcbddf7f20cfd0dac4819ab7956a02` 和 `d4ff823393c6e49da011a7c03e66ad555a33afdbf7f7ab0a105af69f2d90e0bf`。
- 停止后无SoccerFactory进程残留；宿主GPU 0仍为已用0 MiB、可用81,090 MiB、利用率0。GPU 7出现一个与本次无关的本地环境Python进程，占用约516 MiB，未处理。run2 report、Hydra和cache现场全部保留，不得复用或覆盖。

#### G10-B TrackLab Step 1 run3 静态准备与预检（通过，尚未运行GPU）

- 按账本唯一下一步创建全新配置 `reproduction/configs/g10/g10_step1_sngs10004_run3.yaml` 和manifest `reproduction/manifests/g10_soccerfactory_step1_run3_sngs10004.json`。run3相对run1只改变不可变运行身份、独立report/cache/Hydra路径和fail-fast CUDA可见性断言；固定SNGS-10004的255帧、pipeline、三份权重、batch size 16、阈值、4-worker适配以及关闭评估/可视化/W&B/训练的语义不变。
- 现有run2 launcher/worker演进为按manifest读取运行身份和授权变量的共用保护入口；旧run2的report、Hydra、cache、日志和记录哈希没有覆盖。run3要求 `G10_STEP1_RUN3_GPU_APPROVED=YES`、恰好一个数字形式的 `CUDA_VISIBLE_DEVICES` 和宿主执行权限。
- worker的 `init_environment` 阶段现会在上游环境初始化后机器断言：`torch.cuda.is_available()`为true、逻辑CUDA设备数恰为1、逻辑索引0的设备名为 `NVIDIA H800`、总显存至少80,000,000,000 bytes且当时空闲显存至少70,000,000,000 bytes。断言值写入phase事件；不满足时在模型或权重实例化前失败。
- 2026-08-13使用共享参考Python `/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`、`CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1` 执行CPU-only静态预检。外层timeout 120秒，墙钟0.60秒，峰值CPU RSS 18,348 KiB、swap 0，退出码0；脚本记录 `outcome=passed`、全部静态断言通过。
- 预检确认新配置与run1逐键归一化等价，255张固定图片共122,481,437 bytes，三份权重路径/大小不变，13阶段和分阶段timeout不变，run3正式report/cache/Hydra路径均未占用。它只读取本地YAML/JSON/Python源码和资产元数据，没有导入TrackLab/Torch、打开权重内容、执行 `nvidia-smi`、初始化GPU、推理、评估或训练。
- launcher、worker、run3 config和manifest SHA256分别为 `f787fc8bd3bac466d546a6bca67537e6002ac3fff4748130449f610a9cc22caf`、`f8f38ef7f81f6ffd550780b04120a151a31af1192840c34ed4b3453a22df75dd`、`e4368de7d21b088887e44aaa28f00dc5074d9b0328724e81ce63551b0fb20b31` 和 `422a59c11bf1bb06e08cb80828dac2929ad8532018d753e7fdf0a81937344670`。预检结果位于 `reports/g10/20260813_step1_run3_preflight/result.json`，SHA256为 `b2ea00cc6a2c469a6ee518cf0dd8f5b688f387bc4d3aa9a977825c6b2baa343f`；日志SHA256为 `5e9e0060a8fb74e3fbbec92f18d20299b272475a7143b37a74190b7fc2188c15`。
- 本阶段只证明run3静态输入、隔离路径、授权守卫和未来CUDA fail-fast契约已准备好。宿主CUDA断言、dataset、evaluator/三个模型、权重反序列化、显存、255帧推理和状态ZIP仍未知；静态预检不构成GPU授权。

#### G10-B TrackLab Step 1 run3（失败，engine空callback契约）

- 用户在查看宿主GPU实时状态后明确批准物理GPU 0单次执行。启动前GPU 0–6均为NVIDIA H800、已用0 MiB、可用81,090 MiB、利用率0且无计算进程；GPU 7当时有一个外部本地Python进程占用约19,978 MiB，未使用或干预。本次固定GPU 0、SNGS-10004的255帧、共享参考Python、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、30秒心跳、分阶段timeout和10,800秒总上限，关闭评估/可视化/W&B/训练，失败不重试。
- run3以宿主权限启动，静态预检再次通过。新增CUDA fail-fast断言确认仅见逻辑设备0、`torch.cuda.is_available()=true`、设备名为NVIDIA H800、总显存85,028,372,480 bytes、空闲84,478,001,152 bytes；run2的执行沙箱问题没有复现。
- TrackLab导入13.646秒、Hydra compose 0.279秒、环境1.396秒、固定dataset 746.831秒、evaluator 0.009秒、YOLO 3.409秒、PRTReID 25.591秒、StrongSORT 0.003秒、pipeline和TrackerState构造均通过。dataset严格为SNGS-10004单视频255帧；evaluator和三个模型对象均成功实例化，但尚未forward。
- 真正失败阶段为 `instantiate_engine`，不是监控心跳一度显示的 `instantiate_reid`。engine在0.003秒后抛出 `AttributeError: 'NoneType' object has no attribute 'after_saved_state'`。resolved config固定 `engine.callbacks.vis: null`，因为本次按协议关闭可视化；Hydra将该项实例化为None，而上游 `tracklab/engine/engine.py` 第87至88行无条件遍历所有callback并读取 `after_saved_state`。这是上游engine/config的空callback契约错误，不是CUDA、OOM、权重缺失、ReID加载或数据损坏。
- worker墙钟794.853秒、外层墙钟13:16.34、外层和worker退出码均为1、未超时；峰值CPU RSS 1,295,892 KiB、swap 0。worker因失败未到最终GPU资源事件，实际峰值GPU allocated/reserved未知。没有fallback或重试，没有启动track_dataset、evaluation或training，也没有创建状态ZIP。
- 机器结果：`reports/g10/20260813_step1_run3/result.json`，SHA256 `cd59ab0709205d2cee672e3daab7d7101d681e63ee3b5c9bf06c950c944b79d8`；事件SHA256 `b6f9fbc8316191990cf6f7f32f524ec98ae56b887ee6d800c2aab3b24ecaa2b0`；日志SHA256 `85e6612051964585489dcf711105765147fea188ecd4f1c7995f8133dae47fbf`。run3 report、Hydra和cache现场全部保留，不得覆盖或复用。
- 结束后宿主8张GPU均已用0 MiB、利用率0且无计算进程，没有本次残留进程。

#### G10-B TrackLab Step 1 run4 静态准备与预检（通过，尚未运行GPU）

- 按账本唯一下一步创建全新配置 `reproduction/configs/g10/g10_step1_sngs10004_run4.yaml` 和manifest `reproduction/manifests/g10_soccerfactory_step1_run4_sngs10004.json`。run4使用独立的 `20260814_step1_run4` report、cache和Hydra命名空间；正式路径全部未占用，run3现场未覆盖或复用。
- 本地共用worker只在 `instantiate_engine` 边界新增显式适配：先深拷贝engine配置，机器断言原callback键严格为 `ignored_regions`、`progress`、`vis`，唯一解析为None的键严格为已关闭可视化对应的 `vis`；只删除该键，再断言剩余键严格为 `ignored_regions`、`progress` 且均非None，随后调用只读上游engine。输入配置对象、上游源码和其他callbacks不修改。
- run4相对run1只改变不可变运行身份和上述显式适配；固定SNGS-10004的255帧、pipeline、三份权重、batch size 16、阈值、4-worker/cache适配、CUDA fail-fast以及关闭评估/可视化/W&B/训练的语义不变。正式运行要求 `G10_STEP1_RUN4_GPU_APPROVED=YES`、恰好一个数字形式的 `CUDA_VISIBLE_DEVICES` 和宿主执行权限。
- 初始只读契约检查用普通PyYAML读取Hydra保存的resolved文件时，`vis`仍表现为 `${visualization}` 插值文本，因此该检查按预期不满足运行时None假设；没有执行Harness或任何入口。随后改用与Hydra运行时一致的OmegaConf解析，确认原键、唯一空 `vis` 和过滤后的剩余键契约全部通过。
- 2026-08-14使用共享参考Python `/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`、`CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1` 执行一次CPU-only正式静态预检。外层timeout 120秒，墙钟0.17秒、脚本墙钟0.103秒，峰值CPU RSS 18,344 KiB、swap 0，退出码0；`outcome=passed`、全部静态断言通过。
- 预检确认新配置与run1逐键归一化等价、255张图片共122,481,437 bytes、三份权重路径/大小、13阶段和timeout、CUDA守卫及null-callback适配均符合manifest。它没有导入TrackLab/Torch、打开权重内容、执行 `nvidia-smi`、初始化GPU、推理、评估或训练。
- launcher、worker、run4 config和manifest SHA256分别为 `0822273fa03814c2ce102fd7bda5df3449579ed00812408c7e16861969f84721`、`360ff51513aa776304611cdae60a9e8689753551aff3d333a1203a7c57b8302f`、`be8bcccb2e730e0e170289a8bfdee4786920821e06909aab4ccccc9323958fc8` 和 `da1e9582dcc59af0b576399fbfd8295d67db5af80e8f199e5d2c370d6b564070`。预检结果位于 `reports/g10/20260814_step1_run4_preflight/result.json`，SHA256为 `fc3c285b931305d58176c92e3a71536402a078b02f6bde07d1786126426bb49f`；日志SHA256为 `5e9e0060a8fb74e3fbbec92f18d20299b272475a7143b37a74190b7fc2188c15`。
- 本阶段只证明run4静态输入、隔离路径、授权/CUDA守卫和null-vis适配已准备好。运行时callback断言、engine构造、模型forward、255帧跟踪、峰值GPU显存和状态ZIP仍未知；静态预检不构成GPU授权。

#### G10-B TrackLab Step 1 run4（失败，OmegaConf struct模式禁止删除）

- 用户在确认宿主8张H800均已用0 MiB、可用81,090 MiB、利用率0且无计算进程后，批准继续物理GPU 0单次run4。本次固定SNGS-10004的255帧、共享参考Python、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、30秒心跳、分阶段timeout和10,800秒总上限；只允许过滤唯一空 `vis` callback，关闭评估/可视化/W&B/训练，失败不重试。
- run4以宿主权限启动，静态预检再次通过。CUDA fail-fast确认仅见逻辑设备0、设备名NVIDIA H800、总显存85,028,372,480 bytes、空闲84,478,001,152 bytes。TrackLab导入14.287秒、Hydra 0.753秒、环境0.788秒、dataset 16.091秒、evaluator 0.009秒、YOLO 0.308秒、PRTReID 1.713秒、StrongSORT 0.003秒、pipeline和TrackerState构造均通过。该次热缓存耗时不能外推为稳定性能。
- `instantiate_engine` 在0.001秒后失败。run4代码只有在原callback键严格为 `ignored_regions`、`progress`、`vis` 且唯一None键严格为 `vis` 时才会执行删除；实际错误发生在 `del callbacks["vis"]`，因此两项运行时键断言已直接通过。
- 失败直接错误为 `omegaconf.errors.ConfigTypeError: DictConfig in struct mode does not support deletion`，精确路径为 `engine.callbacks.vis`。深拷贝保留了OmegaConf struct mode；本地适配没有使用局部 `open_dict` 修改窗口。这是run4 Harness适配错误，上游engine尚未被调用，不是CUDA、OOM、权重、dataset或模型构造错误。
- worker墙钟36.040秒、外层墙钟36.21秒、退出码1、未超时；峰值CPU RSS 1,295,116 KiB、swap 0。失败前没有最终GPU资源事件，峰值GPU allocated/reserved未知。没有fallback或重试，没有forward、track_dataset、evaluation、training或状态ZIP。
- 机器结果SHA256为 `4311696e87c63fa76577ee761af1133d2a05d0ffb9f711d2c7cca9694a094a90`，事件SHA256为 `61599e421517242d22d8f3414f9a70318cb9aed86c62465198d21452fab0a410`，日志SHA256为 `be568626ecec95f2ba86a61274fea08332a059478a542bf54dab8a809589f246`。run4 report、Hydra和cache现场全部保留，不得覆盖或复用。
- 结束后宿主8张GPU均已用0 MiB、利用率0且无计算进程，没有本次残留进程。

#### G10-B TrackLab Step 1 run5 静态准备与预检（通过，尚未运行GPU）

- 按账本唯一下一步创建全新配置 `reproduction/configs/g10/g10_step1_sngs10004_run5.yaml` 和manifest `reproduction/manifests/g10_soccerfactory_step1_run5_sngs10004.json`。run5使用独立的 `20260814_step1_run5` report、cache和Hydra命名空间；正式路径全部未占用，run4现场未覆盖或复用。
- run5保留run4的深拷贝、原callback键集合、唯一空 `vis`、剩余键和非空性断言；只新增OmegaConf局部 `open_dict` 删除窗口。进入窗口前必须断言复制对象为struct mode，退出窗口后必须断言struct mode已恢复为true，再允许调用只读上游engine。事件将记录删除模式、struct前后状态、原键、过滤键和剩余键。
- 使用共享参考Python执行独立OmegaConf最小测试：构造带 `${visualization}` 插值且根配置为struct mode的callbacks，深拷贝后确认struct为true和唯一空键为 `vis`；在 `open_dict` 内删除后确认struct恢复为true，剩余严格为 `ignored_regions`、`progress` 且均非None。AST、JSON/YAML和 `git diff --check` 同时通过。
- run5相对run1只改变不可变运行身份和上述显式局部适配；固定SNGS-10004的255帧、pipeline、三份权重、batch size 16、阈值、4-worker/cache适配、CUDA fail-fast以及关闭评估/可视化/W&B/训练的语义不变。正式运行要求 `G10_STEP1_RUN5_GPU_APPROVED=YES`、恰好一个数字形式的 `CUDA_VISIBLE_DEVICES` 和宿主执行权限。
- 2026-08-14使用共享参考Python、`CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1` 执行一次CPU-only正式静态预检。外层timeout 120秒，墙钟0.17秒、脚本墙钟0.091秒，峰值CPU RSS 18,596 KiB、swap 0，退出码0；`outcome=passed`、全部静态断言通过。
- 预检确认新配置与run1逐键归一化等价、255张图片共122,481,437 bytes、三份权重路径/大小、13阶段和timeout、CUDA守卫以及open_dict/struct恢复契约均符合manifest。它没有导入TrackLab/Torch、打开权重内容、执行 `nvidia-smi`、初始化GPU、推理、评估或训练。
- launcher、worker、run5 config和manifest SHA256分别为 `cca5a7769697c7b304fcab2a7d04e730db81317e6df04af45e563b049c87cd57`、`11285f8cef4e2e35901fab8cce489a9abafd7ae780023daf2402b45dbc1bb603`、`62547fbe73fec2573ee5e45008a12cb15ae8d5b8bceeea655a46ccf754249d91` 和 `6b8d43e024745c36904cef7da540996e777f38415160052b25b600707a9aa476`。预检结果位于 `reports/g10/20260814_step1_run5_preflight/result.json`，SHA256为 `50bc4985e64c95b229b93f64593c8c6c13b7c3b5845714ffb3923fa2f0060d99`；日志SHA256为 `5e9e0060a8fb74e3fbbec92f18d20299b272475a7143b37a74190b7fc2188c15`。
- 本阶段只证明run5静态输入、隔离路径、授权/CUDA守卫和open_dict适配契约已准备好。运行时struct断言、engine构造、模型forward、255帧跟踪、峰值GPU显存和状态ZIP仍未知；静态预检不构成GPU授权。

#### G10-B TrackLab Step 1 run5（通过）

- 2026-08-14 00:20:36 MST，用户在查看宿主GPU实时状态后明确批准物理GPU 0上的本次单次运行：固定 `SNGS-10004` 全部255帧、最长10,800秒、30秒心跳、失败不重跑，关闭评估、可视化、W&B和训练。启动前GPU 0–6均为空；GPU 7有其他用户进程占用约19,910 MiB，本次没有使用或干扰GPU 7。
- 运行使用commit `90f7ce1ef2cfb848110c3b2ff212dc4cdfcf6a4b` 的既有dirty工作区、共享参考Python `/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`（已验证Python 3.10.16、torch 2.4.1、CUDA build 12.1）、`CUDA_VISIBLE_DEVICES=0`、`G10_STEP1_RUN5_GPU_APPROVED=YES`、`PYTHONPATH=''` 和 `LD_LIBRARY_PATH=''`。输入为固定255帧全量顺序样本，YOLO/PRTReID batch size 16、dataset pool 4 workers；没有随机采样或显式记录推理seed，dtype未由本Harness显式固定或记录。
- 宿主外层命令使用11,000秒timeout包围入口声明的10,800秒总timeout，并保留30秒心跳与退出码。worker从00:20:36运行至00:22:08 MST，worker墙钟92.101秒、外层墙钟92.71秒、退出码0、未超时、无fallback；峰值子进程CPU RSS 2,146,720 KiB、swap 0。
- CUDA fail-fast仅见逻辑设备0，型号NVIDIA H800，总显存85,028,372,480 bytes、启动时空闲84,478,001,152 bytes。13个必需阶段均产生started/passed事件；其中TrackLab导入9.240秒、dataset构造14.462秒、YOLO构造1.210秒、PRTReID构造3.020秒、engine构造0.011秒、255帧 `track_dataset` 60.659秒。
- engine运行时断言确认callback配置已深拷贝，原键严格为 `ignored_regions`、`progress`、`vis`，唯一空键严格为 `vis`；只在局部OmegaConf `open_dict` 窗口删除该键，struct mode在窗口前后均为true，剩余键严格为 `ignored_regions`、`progress`。因此run4的Harness错误已在本次固定协议内修正并实测通过。
- YOLO person、PRTReID和StrongSORT forward/跟踪完成，PyTorch峰值GPU allocated/reserved为3,689,509,888 / 4,756,340,736 bytes。状态归档 `/home/tianlin/SoccerMaster/.runtime/g10/sngs10004_step1/run5/states/sn-gamestate.pklz` 为38,796,789 bytes，SHA256为 `42fc47a85232b91e9ece142d025733ed961e612b9a35179b3e1a56755ca5a7fb`；成员严格为 `summary.json`、`10004.pkl`、`10004_image.pkl`，图像行255、检测行3176、唯一非空track ID数49，必需列和非空track ID断言均通过。
- 评估和训练均未启动。结束后宿主GPU 0–7均为已用0 MiB、利用率0、无计算进程，没有残留SoccerFactory进程。运行结果、事件、主日志和Hydra日志SHA256分别为 `c592d0ad9068419d6836e414008a0b281da5faa47582082ca030fddf02f37ed8`、`53e72fe5c2a3140640ed6c3b644a8245b7550fdc8c6b73c1f40d819befd69483`、`91e5363af012c2c83c7d74f544a13228e521244af1374df6c0f5f664147e0b55` 和 `d57611a82c57649a132cfc9fb0efbb3cc88f66a0135c68172777798954a64c32`；证据说明位于 `reports/g10/20260814_step1_run5/README.md`。
- 本次只证明一个固定序列的Step 1引擎、模型forward、跟踪和状态保存链路。检测行数和轨迹数量不是质量指标；独立重跑一致性、其他序列、Refiner、Step 3、逐序列训练PKL转换和整个G10仍未验证。

#### G10-B run5 → Refiner输入契约审计（审计通过，输入不兼容）

- 按账本唯一下一步，在CPU-only、只读边界内新增固定manifest `reproduction/manifests/g10_refiner_input_contract_audit_run5.json` 和审计入口 `reproduction/gates/g10_refiner_input_contract_audit.py`。它只读取保存版Refiner源码/配置/metadata及run5状态ZIP，不导入Torch或Refiner，不调用上游函数，不运行模型、GPU、评估、训练、Step 3或转换。
- 用户授权范围是继续该只读审查；正式命令固定 `CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=''`、`LD_LIBRARY_PATH=''`、`PYTHONDONTWRITEBYTECODE=1`，使用共享参考Python 3.10.16、NumPy 1.26.4、pandas 2.3.0和PyYAML 6.0.2。120秒timeout内脚本墙钟0.241秒、外层墙钟7.70秒、峰值CPU RSS 173,400 KiB，退出码0、未超时、无fallback。
- 运行使用commit `90f7ce1ef2cfb848110c3b2ff212dc4cdfcf6a4b` 的既有dirty工作区；机器结果保存完整dirty列表。固定归档、metadata、Refiner revision `d4a06f77aebcb45eea1e54b47991dc80ee0f239a`、`inference.py`、`dataset_utils.py`和两份配置的路径、大小与SHA256全部通过。
- 全量读取确认检测表3176行、图像表255行；两表索引均唯一。检测索引为int64但非单调递增；图像索引名为 `id`、object dtype且单调递增。frame严格为0–254，文件名严格为 `000001.jpg`–`000255.jpg`，检测外键均落在图像ID中。
- 3176个 `bbox_ltwh` 均为有限 `(4,)` ndarray，3176个 `embeddings` 均为有限 `(1,256)` ndarray；`track_id` 虽存为float64，但全部非空、有限、整数值且位于1–149，共49个唯一ID。有15帧无检测，每帧最大19个检测，低于Refiner默认上限30，不会在该边界被截断。
- 固定源码契约确认，Refiner在模型forward前无条件读取检测列 `embeddings/bbox_ltwh/bbox_pitch/role/team/jersey_number/track_id` 和图像列 `id/parameters`。run5检测表缺 `bbox_pitch`、`role`、`team`、`jersey_number`，图像表缺 `parameters`；因此 `refiner_input_compatible=false`。审计本身的断言和退出码通过不等于输入兼容。
- 保存版推理配置还继承 `max_frames=750`，而固定归档为255帧；`process_pipeline_video()` 有严格相等断言，因此即使只补列，默认配置仍不能直接运行。此前“Step 1之后直接进入Refiner、Step 3再加入全部球场/相机/语义字段”的设计顺序不完整：现有源码直接证明至少部分字段必须在Refiner之前产生，真实中间生产阶段尚未定位。
- 保存版 `inference.py` 若实际执行会创建输出目录、删除同名输出ZIP并在当前工作目录写临时pickle；此次只审查文本，未触发这些副作用。机器结果、日志和人类报告位于 `reports/g10/20260814_refiner_input_contract_audit/`；结果、日志、入口和manifest SHA256分别为 `91f9ef95b9c0937745799d8a1b726308ecf56428b0c837bac5c5f9602e06c485`、`ef821c5996e5bacf70b49713bdef03f7ec7894cf2c5aab91f1196b2d33f2bc03`、`b5a28d3aa3686a75426567c18b9a2260ede8d6f43ab6c98ba94b528e862157b5` 和 `30158ab2d7268db1d643472debbc2325f0436f33b30974b59b80ccbc697e61d9`。
- 本子阶段没有定位缺失字段生成器、构造兼容归档或255帧配置，也没有验证Refiner checkpoint/model forward、输出、重复性、Step 3或训练PKL。G10-B和整个G10仍未通过。

#### G10-B 历史 Refiner 前置生产链静态审计（通过）

- 按账本唯一下一步新增固定manifest `reproduction/manifests/g10_prerefiner_lineage_audit.json` 和CPU-only静态入口 `reproduction/gates/g10_prerefiner_lineage_audit.py`。正式命令固定 `CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；共享参考Python 3.10.16执行退出码0，脚本墙钟0.0569秒、峰值RSS 16,468 KiB，无fallback。
- 审计以SHA256固定28个小型配置/源码文件，当前保存检出版本为sn-gamestate `bda5ef3f37b66d7d16c22dba234745c05c629ac9`、TrackLab `8987e8525709b5d37751ba60f135760fc91792ad`、Refiner `d4a06f77aebcb45eea1e54b47991dc80ee0f239a`。train/valid/test三份组合配置除了split和实验子名外完全一致，pipeline严格为 `bbox_detector -> reid -> track -> pitch -> calibration -> apply_camera_params -> legibility -> jersey_number_detect -> tracklet_agg -> team -> team_side`。
- 缺失字段生产关系已由源码直接确认：`NBJW_Calib_Keypoints`先产生图像 `keypoints/lines`；`NBJW_Calib_Decouped`由 `keypoints` 产生图像 `parameters/h`；`ApplyParameters`由框和相机信息产生检测 `bbox_pitch`。PRTReID产生逐检测 `role_detection/role_confidence`，Qwen OCR在legibility阈值0.5过滤后产生逐检测号码/置信度，`MajorityVoteTrackletFilter2`按 `track_id` 聚合得到最终 `role/jersey_number`。KMeans轨迹聚类结合embedding与role产生 `team_cluster`，`TrackletTeamSideLabeling`再结合 `bbox_pitch` 产生最终 `team`。
- 固定资产元数据确认YOLO、PRTReID/HRNet、两份NBJW和legibility共六个权重文件存在且大小匹配；Qwen配置路径为只读符号链接，五个safetensors分片逻辑总大小16,584,414,560 bytes。本次只检查路径、链接目标和大小，没有打开权重内容或计算大型权重哈希。
- 三份历史状态归档分别为8,879,607,505、8,790,525,632和6,642,151,935 bytes；只读取各自ZIP内的 `summary.json`，三者均直接列出Refiner所需的检测列 `bbox_pitch/role/team/jersey_number` 和图像列 `parameters`，且完整列集合一致。没有反序列化历史pickle，也没有计算这24,312,285,072 bytes归档的完整SHA256。
- 静态副作用审查确认历史入口不能直接运行：Hydra会chdir并写输出，TrackerState以ZIP append模式写状态，`eval_tracking=true`会写预测/指标/曲线；缺失权重时PRTReID、NBJW和YOLO可能下载，legibility的 `pretrained=True` 可能访问torchvision缓存/网络，Qwen的 `device_map="auto"` 可能使用所有可见GPU和模型缓存。
- 本次没有导入Torch或上游模块，没有执行GPU查询、模型构造、权重反序列化、forward、评估、训练、Refiner、Step 3或转换，也没有修改run5归档。机器结果、日志和人类报告位于 `reports/g10/20260814_prerefiner_lineage_audit/`；结果、日志、入口和manifest SHA256分别为 `e32ee57dd1c391277dfc6badb5608cc4568bd60aa89c6fbf7d60158774473d8e`、`63f773543ee5819e2c99ee1b097d58d527bd546b8546e96d32efab1cbc70c6cb`、`78947f79244de382e8f76fef01269e53ae93d96b2886dac7f8e608ef2ab3edef` 和 `269a48d6aa001d14fffc3146c54af25b6f7a0723cd0997f5f9dcd38d16c277dd`。
- 本阶段只证明保存版配置/源码中的前置字段生产链，以及历史归档schema确实包含目标字段。2025年3月历史运行使用的精确Git commit/依赖仍未知；新生产字段的值、质量、重复性、SNGS-10004运行资源、255帧Refiner配置和forward均未验证。G10-B和整个G10仍未通过。

#### G10-B SNGS-10004前置enrichment run1静态准备与预检retry1（通过，尚未运行GPU）

- 按账本唯一下一步新增本地配置 `reproduction/configs/g10/g10_prerefiner_enrichment_sngs10004_run1.yaml`、固定manifest `reproduction/manifests/g10_prerefiner_enrichment_run1_sngs10004.json`、受保护launcher `reproduction/gates/g10_prerefiner_enrichment.py` 和分阶段worker `reproduction/gates/g10_prerefiner_enrichment_worker.py`。输入严格为Step 1 run5的38,796,789-byte归档，SHA256 `42fc47a85232b91e9ece142d025733ed961e612b9a35179b3e1a56755ca5a7fb`；输出严格为全新的 `.runtime/g10/sngs10004_prerefiner_enrichment/run1/states/sn-gamestate.pklz`，两者绝对路径不同。
- 固定pipeline只包含 `pitch -> calibration -> apply_camera_params -> legibility -> jersey_number_detect -> tracklet_agg -> team -> team_side`，不重复YOLO/ReID/StrongSORT。固定SNGS-10004全部255帧、3176个Step-1检测、49个轨迹；NBJW和相机模块batch 1、legibility batch 16、Qwen batch 64及阈值0.5。评估、可视化、W&B和训练关闭。
- 静态列链逐阶段验证通过：run5已有 `bbox_ltwh/embeddings/role_detection/role_confidence/track_id` 等输入；八模块声明的输入均在其执行点可用，最终预计新增检测列 `bbox_pitch/role/team/jersey_number` 和图像列 `parameters`，静态结构满足Refiner必需列。这里没有运行模块，不能把预计列写成已生成。
- 两份NBJW权重、legibility任务权重和Qwen本地只读符号链接/五个分片的路径与大小通过；Qwen分片逻辑总大小16,584,414,560 bytes。本次没有打开权重内容。共享环境当前缺少torchvision ResNet34基础缓存，而legibility源码在严格加载固定任务checkpoint前调用 `pretrained=True`；未来worker仅在legibility构造局部窗口替换为 `pretrained=False`，并要求严格任务checkpoint加载完成。该适配避免联网/用户缓存写入，最终模型权重仍由固定完整任务state覆盖。
- 未来运行的安全边界已固定：四个dataset workers；MIM/Hugging Face/Transformers/XDG/Matplotlib缓存写入本地run1 cache；offline；单张可见H800且启动空闲显存至少70,000,000,000 bytes；Qwen实例化后参数设备严格只能是 `cuda:0`；engine只过滤唯一空 `vis` callback；输入ZIP运行后必须重算原SHA256。18阶段、30秒心跳、Qwen构造7200秒、track_dataset 21,600秒和总28,800秒timeout均写入manifest；失败无fallback、无自动重跑。
- 首轮CPU预检静态资产/列断言退出0，但随后代码复核发现launcher的run模式仍错误要求空 `CUDA_VISIBLE_DEVICES`、已存在预检报告会误阻塞未来run，且结果将CRC成员读取误标为pickle未读取。首轮证据保留在 `reports/g10/20260814_prerefiner_enrichment_run1_preflight/`，不作为最终运行可达性结论。修正后以仅字符串 `CUDA_VISIBLE_DEVICES=0` 走通run-mode静态路径，未导入Torch或执行GPU操作；随后在全新retry1报告目录执行正式CPU-only预检。
- retry1正式命令使用共享参考Python 3.10.16、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；脚本墙钟0.0712秒、峰值RSS 19,816 KiB、退出码0、无fallback。未导入Torch、TrackLab或sn-gamestate，未查询GPU、反序列化pickle、构造模型、forward、评估或训练；未来Hydra、状态、正式report和cache路径仍未占用。
- retry1机器结果、日志和报告位于 `reports/g10/20260814_prerefiner_enrichment_run1_preflight_retry1/`。launcher、worker、config、manifest、结果和日志SHA256分别为 `80965da17440ea132c3ce44484069f0110e461c6b573176bc068a61c5bf7afd5`、`77cd37c4479e2191f9f8b5f7bc713b2bc390bcdcf097f631552fdc30305d9fd0`、`ce0792ac225436433c82c89ae657e54960f5eefe95d384c8d983da188aeed29e`、`7740784a6754f8c573d3c8900b5996afdd97ccf31fc258bd0b439844c9c80d5f`、`6eaadd28027f9648a139d30c40e79198eb5382a897be587be94d475afa8be3e5` 和 `5e9e0060a8fb74e3fbbec92f18d20299b272475a7143b37a74190b7fc2188c15`。
- 本阶段只证明静态输入、配置、模块列契约、资产元数据、隔离路径与未来Harness守卫已经准备。Hydra实际组合、模型/权重加载、GPU显存、Qwen单卡分配、255帧输出、字段非空率/质量、Refiner和Step 3均未知；本次用户继续指令不构成未来GPU运行授权。G10-B和整个G10仍未通过。

#### G10-B 前置enrichment run1 GPU资源预检（运行前快照，已完成）

- 2026-08-14 01:09:26 MST按账本唯一下一步只读执行宿主 `nvidia-smi`。物理GPU 0–7均为NVIDIA H800，每卡总显存81,559 MiB、已用0 MiB、空闲81,090 MiB，GPU/显存利用率均0%；`--query-compute-apps`无输出，快照时没有计算进程。
- 建议物理GPU 0用于单次run1，以延续Step 1 run5的设备选择。当前没有其他用户进程共存；未来launcher仍会在模型构造前要求仅一张逻辑GPU可见、H800、总显存至少80,000,000,000 bytes且空闲至少70,000,000,000 bytes。Qwen分片、两份NBJW、legibility和推理激活的实际峰值未知；通过守卫不保证无OOM。
- 待授权命令固定物理GPU 0、30秒心跳、内部总timeout 28,800秒和外层28,920秒；输入为SNGS-10004的255帧/run5不可变状态，pipeline为八个前置enrichment模块，输出使用全新本地命名空间。失败不fallback、不自动重跑；不运行评估、可视化、W&B、训练、Refiner、Step 3或转换。
- 该预检阶段只运行资源查询并写入 `reports/g10/20260814_prerefiner_enrichment_run1_gpu_precheck/README.md`；在该时点尚未设置批准变量，也未启动Python、模型或GPU计算。用户随后另行明确批准了本页下一节记录的单次run1；此前其他GPU Gate的批准仍不构成本次授权。

#### G10-B 前置enrichment run1（失败，dataset实例化阶段超时）

- 用户在资源预检后明确批准物理GPU 0上的这一次固定运行。启动前再次只读确认物理GPU 0–7均为H800、每卡总显存81,559 MiB、已用0 MiB、空闲81,090 MiB、GPU/显存利用率0且无计算进程；固定范围仍为SNGS-10004的255帧、八模块前置enrichment、30秒心跳、内部28,800秒/外层28,920秒timeout，失败不重试，不运行评估、可视化、W&B、训练、Refiner、Step 3或转换。
- 运行使用共享参考Python 3.10.16、commit `90f7ce1ef2cfb848110c3b2ff212dc4cdfcf6a4b` 的既有dirty工作区、`CUDA_VISIBLE_DEVICES=0`、`G10_PREREFINER_ENRICHMENT_RUN1_GPU_APPROVED=YES`、空 `PYTHONPATH/LD_LIBRARY_PATH` 和 `PYTHONDONTWRITEBYTECODE=1`。静态预检再次通过；TrackLab导入797.449秒、Hydra组合6.520秒、CUDA环境初始化1.903秒并依次通过，仅见逻辑 `cuda:0`、NVIDIA H800和84,478,001,152 bytes启动空闲显存。
- 首个失败点为 `instantiate_dataset`：阶段发出started事件后在固定1,200秒上限内没有返回passed/failed事件，launcher按协议向worker发送SIGTERM。worker退出码 `-15`，外层退出码241，机器字段为 `timed_out=true`、`timeout_phase=instantiate_dataset`、`failure_category=phase_timeout`、`assertions_passed=false`。总墙钟2,007.889秒，峰值子进程RSS 665,692 KiB。
- evaluator和八个enrichment模块均未实例化，未加载NBJW/legibility/Qwen权重，未发生forward、评估或训练；模型dtype和实际连续GPU峰值不适用/未测。启动前、运行中抽查和结束后GPU 0均为0 MiB、0%利用率；结束后8卡均为0 MiB且无计算进程。`evaluation_started=false`、`training_started=false`、`fallbacks_used=[]`，没有自动重跑。
- 目标 `.runtime/g10/sngs10004_prerefiner_enrichment/run1/states/sn-gamestate.pklz` 不存在；只有32 KiB Hydra配置/日志和40 KiB隔离cache，其中生成30,531-byte Matplotlib `fontlist-v330.json`。输入Step 1 run5归档仍为38,796,789 bytes，SHA256保持 `42fc47a85232b91e9ece142d025733ed961e612b9a35179b3e1a56755ca5a7fb`。
- 同一固定dataset和4 workers在Step 1 run5中实例化只需14.462秒，而本次TrackLab导入本身已慢至797.449秒；共享存储/冷启动I/O抖动是较强推断，但现有事件没有细分wrapper导入、Pool启动、单视频读取和TrackingSet构造，具体阻塞点仍未知，不能写成已确认根因或只靠扩大timeout处理。
- 人类报告、机器结果、事件和日志位于 `reports/g10/20260814_prerefiner_enrichment_run1/`。result、events、run.log、Hydra main.log和resolved config SHA256分别为 `67f1b0c402a712c27c02b8d3c0457de46db103581c3b3323b2c236c2704cd54d`、`a734643831a90b97add231acf35ab1374dca8343a5cfab3cf030b9f312e9249e`、`63211ecebc2ad75cf41f6a3c228b7d797008a45f039a69bb65ad1f53c9e462e9`、`47d3158e960ca8335f3c33b26d518f2593948fb7876b6dd598e761810f104c0a` 和 `bed8f1fa7910a1e8d955a5f90f9690698f19562bb4ab1b8390a6d518fa953ff4`。本次未生成兼容Refiner的状态，G10-B和整个G10仍未通过。

#### G10-B 前置enrichment dataset诊断run2静态准备与首轮预检（失败，retry1已准备未执行）

- 按账本唯一下一步新增CPU-only manifest `reproduction/manifests/g10_prerefiner_dataset_diagnosis_run2_sngs10004.json`、静态预检/未来launcher `reproduction/gates/g10_prerefiner_dataset_diagnosis.py` 和未来worker `reproduction/gates/g10_prerefiner_dataset_diagnosis_worker.py`。它只复用run1的dataset配置，不读取Step 1状态或模型权重，不实例化evaluator/八个enrichment模块，不运行GPU、forward、Refiner、Step 3、转换、评估或训练。
- 未来诊断把run1单一 `instantiate_dataset` 拆为10个事件：`import_tracklab_main`、`hydra_cli_compose`、`import_dataset_wrapper`、`dataset_root_scan`、`pool_create`、`single_video_worker`、`pool_close_join`、`tracking_set_construct`、`tracking_dataset_finalize` 和 `contract_assertions`。4-worker Pool只提交一个固定SNGS-10004任务；单视频worker仍保留1,200秒timeout，没有直接放大run1失败阶段的上限。
- 固定样本实际没有 `Labels-GameState.json`，只有255张JPEG共122,481,437 bytes，因此上游 `video_dir_to_dfs` 目标路径主要执行目录枚举和image metadata构造。未来worker在单视频结果返回后仅用两个显式局部回放适配分别计时TrackingSet聚合和TrackingDataset子采样，避免为后两个阶段再次读取NAS。cache限定在全新run2工作区路径，OpenMIM只精确抑制 `/home/tianlin/.cache/mim` 创建，HOME保持不变。
- 首次正式静态预检使用共享参考Python、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；1.812秒后退出码1。首个且唯一错误为 `worker_ast_contract_phase_order`：检查器用 `ast.walk()` 的树遍历顺序直接对比源码执行顺序，使位于 `try` 节点内的 `single_video_worker/pool_close_join` 被列在后续顶层调用之后。这是本地Harness静态排序错误，不是dataset或未来worker运行结果。
- 失败发生在创建原子结果前；终端traceback原样保存于 `reports/g10/20260814_prerefiner_dataset_diagnosis_run2_preflight/run.log`，并依据命令输出、退出码和运行后路径检查重建同目录 `result.json`，其中 `result_origin` 明确说明不可冒充Harness原生结果。失败命令没有启动worker、导入TrackLab/Torch、读取视频/权重内容或执行GPU操作；未来report/cache/Hydra和retry1预检目录均保持不存在。
- 修复只把收集到的 `run_phase` AST调用按节点 `lineno` 排序后再比较，不改变worker、manifest阶段、timeout、Pool或数据语义；manifest的正式预检输出改为全新 `20260814_prerefiner_dataset_diagnosis_run2_preflight_retry1`。不落盘的针对性单元检查已确认修复后阶段顺序与10阶段manifest严格一致；JSON、AST、`git diff --check` 和未来路径未占用检查通过，但本轮没有执行正式retry1。
- 修复后launcher、worker和manifest SHA256分别为 `1e8b95aa77962e9a0b5fd00a761b8adb77a72e9dffb21621f287131306965bad`、`dd24711407f6359cee40b193aeef2c71d785fcb1db52d1fae429a2138c98238c` 和 `1cd87c0fd4284734090c7792729ed385219a5005fe3c56423988e4214d3d7582`。首轮重建result与原始run.log SHA256为 `be080b590ec28dbd6dd6b6445f5291a6c5755defcc1d17c473f09f71757b23ec` 和 `6145c230aeb81bb28ecbcead08efbd12fc68c54056c7f6708caa86987e983208`。
- 当前只确认诊断设计与首轮Harness失败原因；正式静态预检仍未通过，未来CPU诊断尚未获新指示或执行，运行时阻塞子阶段仍未知。G10-B和整个G10状态不变，未通过。

#### G10-B 前置enrichment dataset诊断run2静态预检retry1（通过，未来诊断尚未运行）

- 用户继续范围严格为修正版CPU-only正式静态预检retry1，不包含未来诊断。固定命令使用共享参考Python、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；脚本墙钟0.799秒、峰值RSS17,700 KiB、退出码0、全部静态断言通过且无fallback。
- retry1确认launcher/worker AST和manifest的10阶段顺序严格一致，并记录各阶段源码行号；只允许未来Hydra `instantiate(cfg.dataset)`，静态禁止TrackLab main/evaluate、Pipeline、TrackerState、模型配置、Torch/CUDA和nvidia-smi调用。launcher、worker和manifest SHA256仍分别为 `1e8b95aa77962e9a0b5fd00a761b8adb77a72e9dffb21621f287131306965bad`、`dd24711407f6359cee40b193aeef2c71d785fcb1db52d1fae429a2138c98238c` 和 `1cd87c0fd4284734090c7792729ed385219a5005fe3c56423988e4214d3d7582`。
- 固定dataset清单再次确认SNGS-10004有255张JPEG共122,481,437 bytes且没有 `Labels-GameState.json`；预检只读取文件名/大小和首尾元数据，没有读取图片内容。4-worker单任务、1,200秒单视频worker上限、两个只读结果回放适配、精确OpenMIM抑制、本地缓存、30秒未来心跳和5,100秒内部总timeout均保持不变。
- 预检机器字段确认 `tracklab_or_model_imports=false`、`torch_imported=false`、`video_or_weight_contents_read=false`、`gpu_operations=[]`；未设置未来诊断批准变量，没有启动worker、TrackLab、Torch、Pool、视频读取、GPU、模型、Refiner、评估或训练。未来 `reports/g10/20260814_prerefiner_dataset_diagnosis_run2/`、run2 cache和Hydra目录仍不存在。
- 正式结果、日志和人类报告位于 `reports/g10/20260814_prerefiner_dataset_diagnosis_run2_preflight_retry1/`；result和log SHA256分别为 `fdd3bb064facff4821daa3205a60a05aa7e9d810c3c69ca35166fa5d9c3fbead` 和 `377248b4d3fed0cc4053a8c8a02d2a685b5a0141a0622f0fef327a09c8bea242`。首轮失败证据保持原样，未覆盖。
- 本阶段只证明未来CPU诊断的静态输入、代码边界、事件、timeout、成功条件和副作用隔离已准备；所有运行时子阶段、真实阻塞边界及enrichment重跑可行性仍未知。G10-B和整个G10仍未通过。

#### G10-B 前置enrichment dataset诊断run2（失败，Hydra完整快照解析了sweep-only字段）

- 用户明确继续一次固定CPU-only run2诊断：外层5,220秒、内部5,100秒、30秒心跳、失败不重试；`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和显式CPU诊断批准守卫。允许TrackLab CPU依赖和固定单视频目录读取，禁止GPU、evaluator、enrichment/模型模块、状态/权重、推理、Refiner、Step 3、转换、评估、可视化和训练。
- `import_tracklab_main` 用时256.954秒并通过，来源严格为只读TrackLab `main.py`。随后 `hydra_cli_compose` 在0.489秒后失败；worker退出码1、未超时，总墙钟279.310秒、峰值子进程RSS402,296 KiB，`evaluation_started=false`、`training_started=false`、`gpu_operations=[]`、`fallbacks_used=[]`。
- 固定pipeline/dataset字段已在worker内完成组合和对比；失败发生在随后 `OmegaConf.save(cfg, resolved_path, resolve=True)` 强制解析完整Hydra树。`hydra.sweep.subdir` 依赖仅在multirun job中存在的 `hydra.job.num`，单次RUN配置缺少该值，抛出 `InterpolationToMissingValueError`。因此这是本地Harness配置快照边界错误，不是dataset、NAS、Pool或数据契约失败。
- `import_dataset_wrapper`、`dataset_root_scan`、Pool、单视频worker、TrackingSet/TrackingDataset和最终契约均未开始；本次没有缩小run1的dataset内部阻塞位置。TrackLab传递导入的具体依赖未记录，但CUDA显式不可见且没有任何GPU操作事件。
- run2 report约32 KiB；Hydra目录为空，cache仅有空Matplotlib目录。没有resolved config、Pool、子任务或视频内容读取。运行后无launcher/worker残留；Step1 run5状态外部复核SHA256仍为 `42fc47a85232b91e9ece142d025733ed961e612b9a35179b3e1a56755ca5a7fb`，worker本身未读取该状态。
- 人类报告、机器结果、事件和日志位于 `reports/g10/20260814_prerefiner_dataset_diagnosis_run2/`；result、events和log SHA256分别为 `8fd8187269011a1d5f639ab28c7f401445e692986d927f8373c2dfd303d34d35`、`59e3f61b7a1be1aff5e322f1cf1e77a6519914a3d42459bff39da7c5ac0758d2` 和 `5b799141c5a21c7e354ecad0e69364fac79fbb9b4026a07de3f04e3350f625ea`。run2现场保留，不得复用或覆盖。
- 候选修复是用全新run3分别保存“完整未解析Hydra配置”与“单独已解析dataset task配置”，避免把未解析快照误称resolved，也不静默删除错误字段；该方案尚未准备或验证。本次停止，不自动创建run3或重跑。

#### G10-B 前置enrichment dataset诊断run3静态准备与预检（通过，未来诊断尚未运行）

- 按账本唯一下一步新增专用 `reproduction/gates/g10_prerefiner_dataset_diagnosis_run3.py`、`reproduction/gates/g10_prerefiner_dataset_diagnosis_run3_worker.py` 和 `reproduction/manifests/g10_prerefiner_dataset_diagnosis_run3_sngs10004.json`，不修改run2 launcher/worker/manifest或运行现场。run3 report/cache/Hydra均使用全新命名空间。
- run3只改变配置证据写法：完整Hydra配置保存为明确命名的 `full_config_unresolved.yaml`，固定 `resolve=false`；实际诊断消费的 `cfg.dataset` 另存 `dataset_config_resolved.yaml`，固定 `resolve=true`。AST静态禁止run2的完整 `OmegaConf.save(cfg, ..., resolve=True)`，并要求两个新调用同时存在；不删除或伪造sweep字段，也不把完整未解析快照写成resolved。
- run2 result语义和哈希重新固定：退出码1、未超时、失败在 `hydra_cli_compose`、缺失插值为 `hydra.job.num`、dataset wrapper未开始、无GPU/fallback。run3仍保留相同10阶段、4-worker单任务、1,200秒单视频worker上限、两个结果回放、本地cache/精确OpenMIM抑制、30秒心跳和5,100秒内部总timeout。
- 正式CPU-only预检使用共享参考Python、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；脚本墙钟1.648秒、峰值RSS17,856 KiB、退出码0、全部静态断言通过且无fallback。
- 预检只读取小型源码/配置/JSON、run1/run2结果和图片文件元数据，没有组合Hydra、执行两个OmegaConf保存点、导入TrackLab/Torch、读取图片/视频/状态/权重内容或使用GPU。机器字段为 `tracklab_or_model_imports=false`、`torch_imported=false`、`video_or_weight_contents_read=false`、`gpu_operations=[]`；未来run3路径仍不存在。
- launcher、worker和manifest SHA256分别为 `57f1e2bce5883f0053e6d14a1aea5473be9d58fabdb2408ce45cb3bf7f67fb74`、`3f355a873e5f30122e55982efd07fb2f0d9ca8127e3f70230a51bcf8d25ee9a7` 和 `9390968e26c7d135e6a94bf16908737f8a4b18174e119b4780e7533d1c61f639`。正式result/log/报告位于 `reports/g10/20260814_prerefiner_dataset_diagnosis_run3_preflight/`，result和log SHA256为 `5a39c2a854970cfefbc61e2ce5d8a97af422a76ede3b322d6fc5110fc7e6d996` 和 `377248b4d3fed0cc4053a8c8a02d2a685b5a0141a0622f0fef327a09c8bea242`。
- 本阶段只证明run3双快照设计和未来诊断静态边界已准备。dataset-only运行时解析、TrackLab/wrapper/Pool/NAS/TrackingSet/TrackingDataset和真实阻塞边界仍未知；G10-B和整个G10仍未通过。

#### G10-B 前置enrichment dataset诊断run3（通过）

- 用户明确继续一次固定CPU-only run3 dataset诊断。外层timeout 5,220秒、内部总timeout 5,100秒、30秒心跳、失败不重试；环境固定 `CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和显式run3批准守卫。允许TrackLab CPU依赖、双配置快照和固定单视频目录，禁止GPU、evaluator、enrichment/模型模块、Step-1状态/模型权重、推理、Refiner、Step 3、转换、评估、可视化和训练。
- 2026-08-14 02:31:02–02:50:18 MST实际运行；外层和worker退出码均为0，机器结果 `outcome=passed`、`assertions_passed=true`，未超时，无fallback。总墙钟1,156.366秒，峰值子进程RSS 665,376 KiB；运行时commit为 `90f7ce1ef2cfb848110c3b2ff212dc4cdfcf6a4b`，完整dirty列表保存在result且既有改动未覆盖。
- 10个阶段各通过一次：TrackLab导入15.520秒；Hydra compose和双快照0.178秒；dataset wrapper导入1,136.700秒；目录扫描0.319秒；4-worker/1-job Pool创建0.458秒；单视频worker 0.021秒；Pool close/join 0.012秒；TrackingSet 1.768秒；TrackingDataset 0.006秒；最终契约断言通过。wrapper导入期间宿主只读观测到worker为 `D` 状态、等待点 `cxiWaitEventWait`，直接证明主要时间消耗在共享文件系统/互连等待，但具体依赖文件未知。
- run2的 `hydra.job.num` 错误未复现。完整未解析快照10,576 bytes，保留sweep插值；已解析dataset快照756 bytes且不含 `${...}`。其SHA256分别为 `1c6c3cd1c072b9002e85af5855154d4f863c31d037e19a16c93b0cc835d98de6` 和 `683ad9f92217719db8ce9ed9ba4ab365c40664bb353cfb1ddd848b0f4f95bf1c`。
- 固定SNGS-10004目录仍无 `Labels-GameState.json`，255张JPEG共122,481,437 bytes；只读元数据扫描后，单视频结果、TrackingSet和TrackingDataset均确认1个视频、255张图像、frame严格0..254、`detections_gt=None`。没有读取JPEG内容、Step-1状态或模型权重；`evaluation_started=false`、`training_started=false`、`gpu_operations=[]`、`fallbacks_used=[]`。
- 人类报告、机器结果、事件和日志位于 `reports/g10/20260814_prerefiner_dataset_diagnosis_run3/`；result、events和log SHA256分别为 `c9a68d175b2ad7d81284b0268d7a2d451d990b0ad6362036b5b8025dff336bb5`、`eae60d71998132fd1f49620b19b46330cc0cb2aa532a3131c20215588195a985` 和 `92ed0e32db86de89411babe496a1995585baa650ca1e35544f781f9135b7fa09`。运行后无残留launcher/worker/Pool，旧现场未覆盖。
- 本次证明固定dataset链路通过，并用运行证据解释run1的1,200秒阶段预算余量很小；它不证明enrichment evaluator/八模块、Qwen分片、GPU forward、状态保存或Refiner兼容。G10-B和整个G10仍未通过。

#### G10-B 前置enrichment run2静态准备与预检（通过，GPU运行尚未授权）

- 用户继续范围严格为CPU-only run2静态准备和一次正式预检；没有执行 `nvidia-smi`，没有设置未来GPU批准变量，没有导入TrackLab/Torch/模型，没有打开Step-1状态归档或读取权重内容，没有GPU、推理、enrichment、Refiner、Step 3、转换、评估、可视化或训练。
- 新增专用config、manifest、launcher和worker，使用全新的run2 report/cache/Hydra/state及预检命名空间；run1、run2/run3诊断和旧预检现场均未修改或复用。worker相对run1只改变manifest身份，归一化后逐字相同；config逐叶差异严格只有实验子名、Hydra run目录和sweep目录。
- run2 manifest逐叶差异限制为身份/说明、固定run1/run3证据、新源码/config身份、全部隔离路径、GPU批准变量/范围和dataset阶段timeout。固定SNGS-10004、255帧、3,176 detections、49 tracks、八模块、现有权重、batch/阈值、Qwen bfloat16/greedy、KMeans `random_state=0`、单卡H800 guard、18阶段、其余timeout、关闭评估/可视化/W&B/训练及失败无fallback/重试均保持不变；`algorithm_semantics_changed=false`。
- 唯一运行预算变化是 `instantiate_dataset` 从run1的1,200秒改为1,800秒；run3 wrapper实测1,136.700秒，静态余量663.300秒。内部总上限仍为28,800秒。这是有证据的启动预算适配，不保证冷缓存或共享存储抖动一定低于1,800秒。
- 正式预检命令使用共享参考Python、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；退出码0，脚本墙钟1.963秒、峰值RSS20,600 KiB、全部机器断言通过且无fallback。
- Step-1状态只检查路径和38,796,789-byte大小；固定SHA256/ZIP成员没有重算或打开，pickle未反序列化。255张JPEG只读文件名/大小元数据；权重只核对路径、类型和大小，五个Qwen逻辑分片共16,584,414,560 bytes，内容未读取。机器字段为 `tracklab_or_model_imports=false`、`torch_imported=false`、`weight_contents_read=false`、`gpu_operations=[]`，未来run2 report/cache/Hydra/state仍不存在。
- launcher、worker、config和manifest SHA256分别为 `1108186be79d189d1dd519f6f3fda24ef00c38fcd85b297b7a9adf60e105316d`、`01257e86b6e21c08c87dc93c2580081d3b44c85df332efd90931aa4a7be2df45`、`a690847587f04d5a76d5ad315e4da47c3ea37718bfec6e7551628ec3ce514ebe` 和 `304583f149df166376288297b4ff8bc11a7eb578d4c8f6cda2fa85fe605b3f61`。正式result/log/报告位于 `reports/g10/20260814_prerefiner_enrichment_run2_preflight/`，result和log SHA256为 `3d6d9c637309ed318ad38083593687a4643ddf89618e51196fee7274443e3f34` 和 `5e9e0060a8fb74e3fbbec92f18d20299b272475a7143b37a74190b7fc2188c15`。
- 本阶段只证明未来run2的静态输入、算法等价边界、隔离路径、timeout和GPU授权守卫准备完成；实际CUDA、dataset冷启动、八模块、Qwen、显存、forward、状态归档与Refiner兼容性仍未知。G10-B和整个G10仍未通过。

#### G10-B 前置enrichment run2 GPU资源检查与授权

- 2026-08-14 03:05:36 MST按账本唯一下一步只读执行宿主 `nvidia-smi`。物理GPU 0–7均为NVIDIA H800，每卡总显存81,559 MiB、已用0 MiB、空闲81,090 MiB，GPU/显存利用率均0%；compute-apps查询无输出。
- 用户看到资源报告后明确表示“如果你看到可用的 gpu，我批准你启动 run2”。由于8卡资源相同且run1使用GPU 0，本次选择物理GPU 0；授权只覆盖一次固定SNGS-10004 run2、外层28,920秒/内部28,800秒、30秒心跳及失败无fallback/重试，禁止Refiner、Step 3、转换、评估、可视化和训练。
- 资源检查报告位于 `reports/g10/20260814_prerefiner_enrichment_run2_gpu_check/README.md`。启动前run2 launcher/worker/config/manifest哈希与正式预检固定值一致，report/cache/Hydra/state路径均未占用。

#### G10-B 前置enrichment run2（通过）

- 2026-08-14 03:06:47–03:17:02 MST在物理GPU 0执行；`CUDA_VISIBLE_DEVICES=0`，共享参考Python，空 `PYTHONPATH/LD_LIBRARY_PATH`，`PYTHONDONTWRITEBYTECODE=1`，固定GPU批准守卫。外层和worker退出码均为0，`outcome=passed`、`assertions_passed=true`，未超时、无fallback；总墙钟615.704秒，峰值子进程RSS4,947,152 KiB。
- CUDA guard确认只见逻辑 `cuda:0`、设备NVIDIA H800、总显存85,028,372,480 bytes、启动时空闲84,478,001,152 bytes。Qwen参数全部在 `cuda:0`；PyTorch峰值allocated/reserved为19,076,444,672/19,639,828,480 bytes，心跳板级最高观测约18,615 MiB。
- 固定18阶段均按顺序started/passed：TrackLab导入90.353秒、Hydra 1.208秒、CUDA 0.889秒、dataset 106.377秒、pitch 28.118秒、legibility 4.545秒、Qwen OCR加载72.824秒、team 22.970秒、engine 0.012秒、`track_dataset` 284.752秒；evaluation明确skipped。dataset远低于新的1,800秒上限，run1超时未复现。
- pipeline、SNGS-10004全部255帧、3,176条Step-1检测、49 tracks、4 dataset workers、batch/阈值、Qwen bfloat16/greedy、KMeans `random_state=0`均按manifest执行。运行读取固定calibration/legibility权重和16,584,414,560-byte Qwen逻辑分片；Triton只在 `/tmp` 编译临时CUDA辅助模块，没有写共享环境、上游源码或原始资产目录。
- 新状态归档位于 `.runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz`，39,016,979 bytes，SHA256 `3b1512e2987ecb86874b1e18e7f6451228652b72a57ed28ec867f4d4995ad606`。ZIP成员/CRC、255 image rows、3,176 detection rows、49 unique tracks、frame 0..254、无空track_id和Refiner核心列均通过机器断言。
- 非空统计为 `bbox_pitch=3176`、`role=3176`、`team=3176`、`jersey_number=1801`、`parameters=255`。输入Step-1归档运行后SHA256仍为 `42fc47a85232b91e9ece142d025733ed961e612b9a35179b3e1a56755ca5a7fb`；输入与输出绝对路径不同。
- `evaluation_started=false`、`training_started=false`、`fallbacks_used=[]`，没有Refiner、Step 3或转换。运行完成首个查询显示GPU 0已释放且无计算进程；稍后外部PID 2057630开始在8卡各占约528–546 MiB，不属于run2。没有残留run2 launcher/worker/Pool。
- 人类报告、机器结果、事件和日志位于 `reports/g10/20260814_prerefiner_enrichment_run2/`；result/events/run.log SHA256分别为 `5d96df53e98623b62e18c44d01c84d09fd2ece223e3716b7a27d0c31343fe3d8`、`1f306e1cf9cd1dec4cf2745fa6737ff474d101ee82e6c8ec7cb6c0a6b34932a5` 和 `b8e953388d2613bd745bad14d4ce2d30f721c9d0084724c9e82f30d12b9de9ae`。实际result未重复嵌入git字段，commit和相关dirty由启动前哈希、正式预检result及运行后只读状态补充记录；这是Harness证据字段缺口，不影响现有运行产物字节。
- 本阶段确认固定样本的前置enrichment生产链和结构契约通过，但非空字段不等于语义正确，单样本不能外推总体质量；保存版Refiner默认750帧与本次255帧冲突仍未解决。G10-B和整个G10仍未通过。

#### G10-B enrichment run2 → Refiner静态契约/质量审计（通过）

- 用户明确继续账本中的CPU-only只读审计。正式命令固定共享参考Python、`CUDA_VISIBLE_DEVICES=''`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和120秒timeout；脚本内墙钟0.312秒、外层11.761秒、退出码0、未超时、峰值RSS178,656 KiB，无fallback。
- 审计以路径、大小和SHA256固定run2 manifest/result/events、39,016,979-byte新归档、metadata、Refiner revision `d4a06f77aebcb45eea1e54b47991dc80ee0f239a`、两份源码和两份配置。run2结果血缘确认通过；ZIP三成员/CRC、summary/DataFrame列一致性、全部255 image rows与3,176 detection rows均通过。
- 图像frame严格0–254、文件名严格 `000001.jpg`–`000255.jpg`；检测外键全部解析。15帧无检测，其余帧最多19条，低于固定上限30。`bbox_ltwh`均为有限 `(4,)` 且宽高为正，embedding均为有限 `(1,256)`；49个track ID全部为1–149整数值。
- Refiner无条件读取的检测列和图像列无缺失。3,176个 `bbox_pitch` 字典键/有限值通过，0个会被固定边界clip；1,801个非空号码均为1–99整数。255条相机参数的必需键、2/3/3×3 shape、有限性和正焦距全部通过，旋转行列式约为1。
- 角色分布为player 3,176；球队分布left 3,119/right 57；号码有1,801条非空、1,375条缺失，非空仅11个唯一值；pitch-x全部为负（-50.4615至-16.3323）。这些只是不使用真值时的分布观察，不能判为正确或错误，也不能外推总体质量。
- 机器判定 `refiner_255_frame_static_contract_ready=true`：若显式覆盖 `max_frames=255`，归档满足固定数据预处理源码的静态入口契约。同时 `refiner_input_compatible=false`：保存版默认解析为750帧并有严格相等断言，因此不能原样运行。审计通过不等于默认配置兼容或模型forward正确。
- 未导入Torch/Refiner、未加载模型或checkpoint、未使用GPU，也没有forward、评估、可视化、训练、Step 3、转换或输入写入。人类报告、机器结果和日志位于 `reports/g10/20260814_prerefiner_enriched_contract_audit_run2/`；result/log/入口/manifest SHA256分别为 `fa7c1baf0c4215c3421babe54530d51fb039e417ad2ef55ae6f915250ee3d6a6`、`a6ed629290ebc7d98922beed5d906f351eb5ed87dea2dd559790712b58203e43`、`6d736b113a644c35f5856a00c895e7ff02de61342c1045923fa5a219653ac724` 和 `e28e6ef39ad96dfaa4604e503dc0459df2ec5cbcf5d17ba137940771184259a0`。G10-B和整个G10仍未通过。

#### G10-B 255帧Refiner单样本probe静态准备/预检（通过）

- 用户明确继续但要求暂不测试可能使用GPU的内容。本阶段仅准备和执行CPU-only静态预检；未执行 `nvidia-smi`，也没有获得或声明GPU运行授权。
- 该静态预检执行当时，gpu200尚未挂载 `/remote-home`。按用户说明通过 `GPU202-tianlin` 只读访问同一GPFS，确认文件系统为GPFS、Refiner提交为 `d4a06f77aebcb45eea1e54b47991dc80ee0f239a`；`inference.py`、`dataset_utils.py`、原推理override和`base.yaml`的路径/大小/SHA256均与既有审计固定值一致。2026-08-17 后续只读复核已确认 gpu200 的该 GPFS 挂载恢复；这不改写预检当时的资产核验来源。
- 历史 `inference.sh` 唯一checkpoint固定为 `/remote-home/haolinyang/sports/soccernet/Refiner/outputs/train_timesformer_100clip_coord_only_not_0init_l2_xyflip_seed42_20250328_224427/best_model.pth`，323,985,486 bytes，SHA256 `d8e383e93e209e4ac3aa8fb97d8b209fe10b183f622cbae40d9124160238fe5a`。这里只读计算文件哈希，没有导入Torch/Refiner或反序列化checkpoint。
- 新增基线override、255帧target config、固定manifest和受保护入口：`reproduction/configs/g10/g10_refiner_sngs10004_baseline_override.json`、`reproduction/configs/g10/g10_refiner_sngs10004_255_probe_run1.json`、`reproduction/manifests/g10_refiner_probe_run1_sngs10004.json`、`reproduction/gates/g10_refiner_probe_run1.py`。两份配置逐叶差异严格只有 `data.max_frames`；固定上游base解析值为750，target显式为255。
- 正式静态预检使用本地已验证Gate Python、空 `CUDA_VISIBLE_DEVICES/PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和180秒外层timeout，退出码0。输入归档/metadata/既有契约审计身份、ZIP/CRC、255帧合同、配置唯一差异、未来路径未占用和禁止可视化参数均通过。
- 保存版入口会创建输出目录、删除同名既有输出ZIP、在当前目录写两份临时pickle并自动选择device。未来launcher以全新workspace-only work/output/report目录隔离，所有目标必须不存在，省略全部可视化参数，并固定30秒心跳、14,400秒timeout、120秒终止宽限、无fallback、失败不重跑；运行模式还要求 `G10_REFINER_PROBE_RUN1_GPU_APPROVED=YES`、恰好一张可见H800和至少70,000,000,000 bytes空闲显存。
- 机器结果与日志位于 `reports/g10/20260817_refiner_probe_run1_preflight/`。结果明确记录 `torch_imported=false`、`refiner_imported=false`、`checkpoint_deserialized=false`、`gpu_operations=[]`、`forbidden_actions_executed=[]`；没有model load、forward、Step 3、转换、评估、可视化或训练。实际Refiner probe、checkpoint加载和输出正确性仍未知，G10-B和整个G10仍未通过。

#### G10-B 255帧Refiner单样本probe run1（失败，未进入GPU）

- 2026-08-18 12:07 UTC，用户在看到资源报告后明确批准只在物理GPU 7运行一次固定probe，并要求使用新的tmux窗口。启动瞬间fresh `nvidia-smi`显示GPU 7为NVIDIA H800、总显存81,559 MiB、已用0 MiB、空闲81,090 MiB、利用率0%、无计算进程；GPU 1有既有PID 111449占516 MiB，其余卡无计算进程。
- 新建tmux会话 `g10_refiner_probe_gpu7_run1`、窗口 `refiner_probe`，固定 `CUDA_VISIBLE_DEVICES=7`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和一次性批准守卫，调用本地Gate Python的受保护launcher。静态预检打印started/passed后，launcher在 `validate_runtime_assets()` 的首个Refiner revision检查失败：`git -c safe.directory=* -C /remote-home/haolinyang/sports/soccernet/Refiner rev-parse HEAD`退出码128，Git报告该只读仓库因所有者为`nobody:nogroup`而是dubious ownership。
- 只读诊断确认当前Git为2.34.1；即使把命令行配置改成固定 `safe.directory=/remote-home/haolinyang/sports/soccernet/Refiner`，仍得到同一拒绝。没有修改全局Git配置、远端仓库或共享环境。按预定失败策略，本次没有fallback或自动重跑。
- 失败发生在 `cuda_guard()`、checkpoint哈希/反序列化、模型加载和forward之前；固定work/output/report目标均未创建，因此launcher也未写正式run result。唯一运行证据是 `reports/g10/g10_refiner_probe_gpu7_run1_tmux.log`。异常退出后的只读GPU快照显示GPU 7仍为已用0 MiB、空闲81,090 MiB、利用率0%且无计算进程。
- 主要失败分类为 `environment_or_abi`，并暴露launcher在早期运行资产异常时不能写结构化失败result的Harness缺口。实际Refiner checkpoint/model forward仍未知，G10-B和整个G10仍未通过。

#### G10-B 255帧Refiner单样本probe run2静态准备/预检（通过）

- 按run1失败后的唯一下一步新增全新run2身份：`reproduction/gates/g10_refiner_probe_run2.py` 和 `reproduction/manifests/g10_refiner_probe_run2_sngs10004.json`。run2继承固定run1数据、checkpoint和255帧配置契约，但使用全新的work/output/report/preflight路径和 `G10_REFINER_PROBE_RUN2_GPU_APPROVED=YES` 守卫，不复用run1运行现场。
- run2不修改全局Git配置或只读Refiner仓库，改为直接只读解析 `.git/HEAD` 及其loose/packed ref，并严格比较固定40位commit。它同时捕获revision/资产检查等早期异常并在run report路径写结构化失败result，补齐run1暴露的Harness缺口。
- CPU-only静态预检固定本地Gate Python、空 `CUDA_VISIBLE_DEVICES/PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和180秒timeout，退出码0；机器结果为 `reports/g10/20260818_refiner_probe_run2_preflight/result.json`。结果记录 `stage=refiner_255_probe_run2_preflight`、`assertions_passed=true`、future paths未占用、`gpu_used=false` 且禁止动作为空。没有导入Torch/Refiner、反序列化checkpoint、使用GPU或运行forward。
- 预检后fresh资源查询显示GPU 6/7当时各空闲81,090 MiB、利用率0%、无计算进程，但既有一次性解说autorun已于12:10:54 UTC通过smoke，并于12:11:55 UTC进入 `running_eight_shards`，正按45秒间隔从GPU 0至7启动八个worker；GPU 0–4已有该任务进程，GPU 6/7是其已授权八卡阶段的保留设备。为避免资源冲突，没有启动Refiner run2或48条attribute probe。

#### G10-B 255帧Refiner单样本probe run2（通过）

- 2026-08-19 05:08 UTC，用户明确批准只在物理GPU 7启动Refiner run2。启动前fresh `nvidia-smi`显示GPU 7已用0 MiB、空闲81,090 MiB、利用率0%且无compute process；GPU 1–6同样空闲。GPU 0有其他用户PID 1269353，已用18,479 MiB、空闲62,611 MiB、利用率100%，本任务未使用或改变该进程。
- 新建tmux会话 `g10_refiner_probe_gpu7_run2`、窗口 `refiner_run2`，固定 `CUDA_VISIBLE_DEVICES=7`、空 `PYTHONPATH/LD_LIBRARY_PATH`、`PYTHONDONTWRITEBYTECODE=1` 和一次性批准守卫，调用本地Gate Python的run2受保护入口。只读revision解析确认Refiner提交 `d4a06f77aebcb45eea1e54b47991dc80ee0f239a`；CUDA guard确认单张H800且启动时空闲84,478,001,152 bytes。
- `SoccerTrackerTransformerTimeSformer`的coord-only checkpoint加载和forward完成；SNGS-10004固定255帧按三个100帧clip处理，末段按既定逻辑padding。进程退出码0，总墙钟7.009秒，其中视频处理/推理日志约1.30秒；无timeout、fallback或自动重跑。
- 输出归档为 `.runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz`，38,952,351 bytes，SHA256 `76eec06718b9f5c0e24e4d881a80af8be0b942c20f97f5d9d63b29f5c20a6041`，ZIP成员严格为 `10004.pkl` 和 `10004_image.pkl`。机器结果和主日志位于 `reports/g10/20260818_refiner_probe_run2/result.json` 与 `reports/g10/20260818_refiner_probe_run2/run.log`，tmux管道日志为 `.runtime/g10_refiner_probe_gpu7_run2_tmux.log`。
- 本次只验证单场255帧coord-only Refiner执行及输出结构，没有评估坐标语义质量、可视化、Step 3、转换或训练，也不能外推其他比赛或重复运行稳定性。GPU 7在进程结束后已释放。G10-B和整个G10仍未通过；本分支下一步是CPU-only比较Refiner前后球场坐标和轨迹连续性，再决定是否推进Step 3。

#### G10-B Refiner前后坐标效果审计（完成，出现质量风险）

- 新增CPU-only入口 `experiments/soccerfactory_visualization/audit_refiner_coordinate_effect.py`，只读比较固定enrichment归档和run2 Refiner输出；不使用GPU、模型、训练、Step 3或人工二维坐标真值。输入检测身份和顺序严格保持3,176行、49 tracks；有检测的帧为240/255。
- 首次执行在绘图阶段因当前Matplotlib不支持`tick_labels`参数而退出码1，未生成正式result/figure；失败日志保留为 `reports/g10/20260819_refiner_coordinate_effect/run.log`。只把两处参数改为兼容的`labels`后进行一次针对性重跑，120秒timeout内退出码0、墙钟1.70秒、峰值RSS312,024 KiB，正式日志为同目录 `run_retry1.log`。
- Refiner改动3,176/3,176个bottom-middle坐标；位移中位数0.204米、均值0.234米、95分位0.511米、最大1.091米。前后均为3,176/3,176条在固定球场边界内，没有新增越界。
- 相邻连续帧位移中位数从0.154增至0.253米，比例1.643；二阶差分抖动代理中位数从0.213增至0.401米，比例1.883。49条轨迹中，40条的相邻步长中位数增大、仅3条减小；35条的二阶差分中位数增大、仅3条减小。该变化不是仅由少数极端点造成。
- 已确认本样本上Refiner没有表现出时间平滑效果，构成进入Step 3前的质量风险；但没有二维真值，因此不能由此断言绝对坐标精度下降。机器结果、说明和图位于 `reports/g10/20260819_refiner_coordinate_effect/`。唯一下一步是在CPU上画出高位移轨迹及100/200帧clip边界附近的前后轨迹，判断问题是普遍逐帧噪声还是分clip/末段padding伪影；在此之前不推进SoccerFactory Step 3。

#### G10-B Refiner clip边界归因（完成，边界不是主因）

- CPU-only诊断读取同一固定前后归档，并按保存版`max_clip_frames=100`把连续步和二阶差分分为100/200帧边界邻域与clip内部；入口为`experiments/soccerfactory_visualization/diagnose_refiner_clip_boundaries.py`，没有重新运行模型、GPU或Step 3。
- 保存版推理对0–99、100–199、200–254三个clip独立forward后直接拼接，无重叠或融合；坐标头在归一化空间预测输入坐标的residual。跨边界27个连续步的中位数由0.169米增至0.615米，中位增量为+0.354米，说明边界会放大局部跳变。
- 边界不能解释整体退化：段内2,933个连续步中位数仍由0.154米增至0.251米，中位增量+0.075米；去掉边界邻域后，2,768个二阶差分中位数仍由0.212米增至0.397米。边界邻域只占全部二阶差分样本的1.84%。
- 机器结果、说明和逐帧图位于`reports/g10/20260819_refiner_coordinate_quality/`。当前输出保持结构合规，可用于后续Step 3接口smoke，但不得表述为坐标质量改善；SNGS-10004没有可用二维人物球场坐标真值，绝对精度变化仍未知。

#### SNGS-10004同场Qwen角色替换及多视图后续（完成）

- 固定SNGS-10004人物框、49条轨迹和137个代表视图后，单帧Qwen明确`goalkeeper/referee`命中3/8条人工门将/裁判，外场误拒0/39；相同比赛PRTReID为0/8。把`other`视为非外场虽命中8/8，但误拒25/39，不能采用。
- 后续多视图实验在物理GPU 1完成49/49条轨迹，退出码0、耗时25.7秒、峰值allocated/reserved约15.9/16.7 GiB，无OOM、训练、fallback或重跑。明确角色只命中2/8、外场误拒0/39；新识别门将42，但把单帧已识别的裁判4、17改判为player。
- 正式判定为`track_multiview_does_not_improve_single_view_baseline`，不保留当前多视图提示，也停止在现有两场继续调prompt和`other`阈值。保留结论仅为：单帧Qwen明确角色输出可作高精度补充，但召回不足，不能独立承担角色门控。
- 结果位于 `reports/g10/20260819_qwen_role_swap_sngs10004/track_multiview_evaluation.json`，对比图为同目录 `method_comparison.png`。若继续该分支，优先在一场新的独立比赛验证单帧高精度现象；若不能重复，再转向专门的足球角色分类器。

#### SNGS-10002新比赛单帧Qwen盲推理（完成，等待独立标签）

- 用户在fresh资源报告后确认物理GPU 1与Q-Former阶段3并行运行。新比赛固定为历史Step-3归档中数值最小且未参与SNGS-10001/SNGS-10004角色诊断的SNGS-10002；722帧、33 tracks，按相同三时间段最高框置信度规则得到89个视图。GPU manifest不包含历史`role/role_detection/role_confidence`或人工角色标签。
- 首次CPU准备因把“最多3视图”误写为严格99视图而退出，未写manifest；修正为真实33 tracks/89视图后一次针对性重跑通过。正式推理在新tmux会话 `g10_qwen_newmatch_gpu1`、窗口 `qwen_sngs10002` 执行，物理GPU 1、BF16、batch 4，89/89完成，退出码0、耗时27.95秒，峰值allocated/reserved约17.41/20.59 GB，无OOM、训练、fallback或重跑。
- 盲输出分布为`other=62`、`player=21`、`referee=5`、`goalkeeper=1`；这只是模型输出，不能在独立标签缺失时计算精度、召回或外场误拒。manifest、原始预测和日志位于 `reports/g10/20260819_qwen_role_newmatch_sngs10002/`。下一步是冻结独立人工角色标签后再评价，不能用历史Qwen伪标签给当前Qwen打分。
- 运行后只读自一致性统计显示33条轨迹中16条的不同视图输出不一致，17条一致；按多数输出有27条`other`、4条`player`、2条`referee`。明确非外场输出涉及轨迹10/13/14/22/23的referee视图和轨迹27的goalkeeper视图。目视抽查轨迹27的三帧均更像同一名蓝队外场球员，而Qwen输出为两次player、一次goalkeeper；轨迹22单帧黑衣目标的referee输出在视觉上较合理。该抽查不是正式标签，但与高`other`率共同表明当前输出不稳定，且轨迹ID switch会使简单轨迹多数评价失真。独立页面应先增加身份一致性和逐视图角色字段，再形成正式评价。

#### 解说诊断阶段1 → 阶段2条件式继续授权

- 2026-08-18，用户明确授权：阶段1 autorun 正常完成后，可以不再询问而直接进入阶段2的只读完整性检查、错误统计和CPU-only轻量grouped probe。正常条件至少包括八个worker均退出码0、3,256个固定开发样本无缺失/重复、合并成功、必需中间表示shape/身份契约成立且没有非有限值或未声明fallback；任一条件失败即停止并报告，不得自动修复或重跑。
- 该授权允许在CPU上拟合固定协议中的match-grouped轻量probe，不训练或修改视觉backbone、Q-Former、projector、Llama或其他主模型，不写模型checkpoint，也不进入阶段3 oracle intervention、holdout evaluation或任何主模型训练。
- 该授权不豁免GPU强制规则。阶段2中的GPU 6冻结特征抽取仍需在阶段1释放资源后fresh执行并报告 `nvidia-smi`，然后等待用户看到资源/进程与精确命令范围后的当次确认；Refiner run2同理且属于独立G10任务。

#### 解说诊断阶段1完成与阶段2首轮CPU逐接口probe

- 解说autorun的smoke于2026-08-18 12:10:54 UTC退出码0；八个正式worker随后各完成407/407条并全部 `status=passed`，CPU merge完成后于12:54:57 UTC按配置停在 `review_required`。合并manifest为 `reports/commentary_autorun_3256_20260817_run1/manifest.json`，记录3,256条索引精确覆盖、八份各453,447,584-byte缓存和对应基线预测；没有训练、backward、optimizer、scheduler或checkpoint写入。
- 八个正式worker累计约5.210 H800·小时，单样本smoke约0.159 H800·小时，合计约5.370 H800·小时；正式worker墙钟约36.4–41.3分钟，整段从smoke启动到 `review_required` 约54分钟。每卡峰值allocated约19.51 GB、reserved约20.03 GB。
- 按用户条件式阶段2授权，新增CPU-only入口 `experiments/commentary_generation/run_stage2_layer_probe_cpu.py`，把已锁定的48条Codex视频事实开发标签无歧义映射到完整缓存中的dataset index，覆盖36场比赛。它严格核对3,256条cache/prediction索引、视频/比赛身份、六个表示键和有限性，然后复用4-fold match-grouped、fold-local StandardScaler/PCA与L2 logistic协议；不导入主模型、不使用GPU、不运行模型forward或写checkpoint。
- 正式CPU probe退出码0；17个有足够正负支持的任务在六个接口上均完成。macro ROC-AUC依次为：`visual_frame_global=0.81883`、`layer_normalized=0.81862`、`temporal_output=0.80771`、`qformer_input=0.80771`、`qformer_output=0.77552`、`projector_output=0.77609`。最大相邻macro下降为Q-Former input→output的-0.03219；时间阶段约-0.01090，projector约+0.00057。
- 任务拆分显示下降并非所有属性一致：pass、possession sequence、shot attempt、saved、foul/tackle等在Q-Former后下降，而cross等任务没有同向下降。该结果只基于48条/36场、Codex视频盲标开发标签和轻量probe，说明Q-Former是优先候选但不能构成因果归责；正式因果排名仍需interface-matched intervention，且当前缓存不含decoder logits或Llama内部层probe。结果为 `reports/commentary_stage2_layer_probe_48_20260818/result.json`。
- probe完成后的fresh `nvidia-smi`显示外部PID 254199在GPU 0–7各占约36.16–36.32 GiB且利用率均99%，GPU 6/7分别只空闲44,891/44,859 MiB。没有启动GPU 6 attribute extraction或GPU 7 Refiner run2，也没有与外部任务共存。

#### 48条attribute probe一次性GPU heartbeat autorun（已arm，等待资源）

- 2026-08-18用户明确批准每30分钟查询GPU 0–7，并在任一H800满足空闲显存至少70,000 MiB、利用率不超过5%且无compute process时运行下一项attribute probe；设备不固定为GPU 6。新增固定配置 `experiments/commentary_generation/ATTRIBUTE_GPU_AUTORUN_CONFIG_20260818.json` 和supervisor `experiments/commentary_generation/attribute_probe_gpu_autorun.py`。
- 配置固定连续两次、间隔60秒确认同一候选卡，选择稳定候选中最低物理索引；只允许一次48条冻结backbone inference-only特征抽取，成功后一次CPU match-grouped probe和阶段3准备条件判定。失败、timeout或输出契约不符时停止，不重跑、不在worker启动后换卡、不与任何compute PID共存。
- CPU-only `--check`退出码0，精确配置SHA256为 `15d4b6a463f7368220b996c154528726c81e4907cc978dadf4b2099270617932`；随后按用户授权以该哈希一次性arm，并在新tmux会话 `attribute_probe_gpu_autowatch_20260818`、窗口 `watcher` 启动。运行时状态/日志位于 `.runtime/attribute_probe_gpu_autorun_20260818/`。
- 14:38:27 UTC首个heartbeat确认外部PID 254199仍在8张卡各占约36.16–36.32 GiB、利用率99–100%，无eligible GPU；状态为 `waiting_for_one_idle_h800`，下一次检查间隔1,800秒。尚未导入模型、运行attribute GPU worker或CPU probe。
- 用户同时表达：若阶段2证据确实只剩Q-Former，则直接授权阶段3首轮因果干预。attribute supervisor的已arm配置仍只按固定机器规则停在 `stage3_ready`，禁止运行阶段3 GPU命令，也不会因后续静态文件而改变其已冻结哈希或权限范围。

#### 48条attribute probe一次性GPU heartbeat autorun（完成，停在stage3_ready）

- 2026-08-19 05:08:40 UTC，heartbeat发现物理GPU 1–7均满足固定空闲条件；按最低稳定索引规则选择GPU 1，并在60秒后再次确认其空闲81,090 MiB、利用率0%且无compute process。GPU 0的外部PID 1269353未被使用或改变。
- GPU 1冻结backbone特征抽取于05:09:40 UTC启动，48/48条完成，退出码0、耗时115.81秒，峰值allocated约3.33 GB；输出三个表示为`global_mean [48,1024]`、`global_sequence [48,30,1024]`和`local_late_sequence [48,30,1024]`。没有主模型训练、backward、optimizer、scheduler、任务头推理或checkpoint写入。
- 随后CPU match-grouped attribute probe退出码0，17项任务完成，最佳表示的macro ROC-AUC为0.77758。固定readiness规则全部成立：既有逐接口probe中Q-Former相邻下降-0.03219，约为时间阶段下降-0.01090的2.95倍；projector变化+0.00057，不构成竞争瓶颈。
- supervisor于05:12:10 UTC写入`phase=stage3_ready`并正常停止，一次性授权已消耗；阶段3没有执行。停止原因是尚无可执行的interface-matched Q-Former干预launcher、固定manifest、输出契约和公平性断言。特征和结果位于 `reports/commentary_attribute_probe_pilot_48_20260818/`，运行状态位于 `.runtime/attribute_probe_gpu_autorun_20260818/state.json`。
- 该48条/36场开发pilot支持“Q-Former是首要因果干预候选”，但不是总体benchmark，也尚不能证明Q-Former是唯一瓶颈。下一步是实现并CPU-only预检阶段3首轮Q-Former oracle steering入口；任何后续GPU执行仍须遵守新的资源检查和当次授权范围。

#### 阶段3 Q-Former oracle steering静态设计/预检（通过，未执行）

- 新增机器配置 `experiments/commentary_generation/QFORMER_ORACLE_STEERING_PILOT_CONFIG_20260818.json`、人读协议 `QFORMER_ORACLE_STEERING_PILOT_PROTOCOL_20260818.md` 和CPU-only预检入口 `preflight_qformer_oracle_steering.py`。配置固定在原生Q-Former输出 `[32,768]` 内做fold-local class-centroid residual steering，保持历史projector、Llama、checkpoint、词表限制和确定性五束解码不变；48条/36场使用4-fold match-grouped拆分。
- 首轮只保留每个训练折正负样本均不少于4的12个事件/动作/结果/阶段任务；预注册baseline、norm-matched cyclic-task control和 `alpha=0.5/1.0` 两个oracle剂量，残差总范数受训练折自然尺度约束。primary review只使用锁定视频事实标签，不使用参考解说。该实验即使阳性也只说明Q-Former输出信息损失是因果贡献者，不能证明它是唯一瓶颈；因使用oracle标签和centroid residual，明确不进入最终模块公平排名。
- CPU静态预检退出码0，结果为 `reports/commentary_qformer_oracle_steering_pilot_48_20260818/preflight.json`。它只读核对8个Stage-1分片均为 `[407,32,768]` F32、48条身份全部命中、12个任务逐折支持及安全标志；没有导入Torch、查询/使用GPU、加载模型、准备干预张量或生成文本。机器状态为 `WAITING_FOR_ATTRIBUTE_PROBE_RESULT`，因为attribute结果文件尚不存在；`execution_authorized=false`。
- 静态检查同时确认 `cache_layers_shard_gpu.py` 用 `captures["qformer_input"].clone()` 填充 `temporal_output`。因此Stage-2中两者相同的AUC是按构造得到，不能当作时间模块与Q-Former输入的两份独立证据。
- 随后按用户要求实现但不arm独立continuation：CPU准备器为 `prepare_qformer_oracle_steering_cpu.py`，inference-only GPU worker为 `run_qformer_oracle_steering_gpu.py`，一次性supervisor为 `qformer_stage3_continuation.py`，冻结模板为 `QFORMER_STAGE3_CONTINUATION_CONFIG_20260818.json`。它只接受原attribute watcher精确SHA-256对应的 `stage3_ready` 且全部checks为真；CPU准备成功后等待同一70,000 MiB/5%/零compute PID条件连续两次成立，再在一张H800上运行48条×4条件，成功后停在 `review_required`。失败不重跑、不在worker启动后换卡。
- continuation CPU-only `--check`最终退出码0，静态模板SHA-256为 `ab2f79229b3134b0ddb30c6b3677b8c7bd7d90014e419e96e57059b770c4684e`。真实缓存内存dry-run生成四个 `[48,32,768]` 有限条件、4折和等范数control；45条有至少一个clear稳定任务，`CE200-024`、`CE200-101`、`CE200-002` 无可用任务，按预注册规则四条件保持原样并从干预效应分母排除。check没有写artifact、导入Torch、查询GPU或加载模型。
- supervisor固定校验CPU准备器、GPU worker、自身、pilot配置和共享decoder runtime的SHA-256；GPU worker还要求授权后配置状态、`execution_authorized=true` 和仅由supervisor注入的 `QFORMER_STAGE3_GPU_APPROVED=YES`。截至原静态准备结束时模板为 `static_unarmed`/`execution_authorized=false`，没有arm、tmux进程或GPU授权；2026-08-19的后续授权与启动状态见下节。

#### 阶段3 Q-Former oracle steering正式运行（生成完成，等待盲审）

- 2026-08-19用户明确要求与新比赛Qwen并行推进，并在看到fresh `nvidia-smi`报告后确认GPU0+GPU1范围。启动瞬间GPU0–7均为H800、各空闲81,090 MiB、利用率0%且无compute process。
- 阶段3配置只把状态改为`authorized_once_stage3_continuation`、`execution_authorized=true`，最终静态检查SHA-256为 `3627d58ce1cce5d731f9730ff243a9514991671742a9a53bce031481d691c545`；CPU/GPU worker检查均ready，48条四条件张量dry-run、4折、12个稳定任务和输出未占用契约通过。一次性arm后在新tmux会话 `commentary_qformer_stage3_gpu0`、窗口 `stage3` 启动。
- CPU准备于05:31:58 UTC完成并写出48条干预张量；supervisor连续两次确认GPU0空闲后，于05:32:58 UTC在物理GPU0启动48条×4条件的inference-only生成。48/48条四条件生成全部完成，GPU worker退出码0，生成阶段耗时331.829秒；checkpoint加载一次，峰值allocated/reserved约15.16/15.25 GiB。没有视觉或Q-Former forward、主模型训练、backward、optimizer、scheduler、checkpoint写入、fallback或自动重跑。
- supervisor已停在`review_required`并释放GPU0。机器结果为`reports/commentary_qformer_oracle_steering_pilot_48_20260818/result.json`，盲化配对审阅包为同目录`paired_review_packet.json`。相对baseline，错配方向control、oracle alpha 0.5和alpha 1.0分别改变23、24和34条token序列；这只证明干预影响了解码输出，不等于事实质量改善。
- 当前尚不能给出Q-Former因果效果结论。下一步是CPU/人工按锁定视频事实盲审四条件：只有oracle至少新增5/48条中心事实正确、改善数至少为退化数两倍，且norm-matched control没有获得相同增益，才判定首轮rescue为阳性；盲审前不追加GPU实验。

#### G10-B Step 3静态审查与接口smoke准备（通过，尚未运行接口smoke）

- 按账本唯一下一步完整审查保存版 `gsr_step_3_sn500_1000.yaml`。历史pipeline并非轻量收尾，而是重新执行 `reid -> pitch -> calibration -> apply_camera_params -> legibility -> jersey_number_detect -> role`，随后才进行删除场外轨迹、属性聚合、轨迹拼接和球队分配；原配置还启用tracking evaluation并面向全部sn500，不能直接用于当前固定样本。
- 已确认coord-only Refiner run2修改的唯一业务列为`bbox_pitch`，而历史Step 3中的`apply_camera_params`声明并写入同一列。因此原样运行历史Step 3会重新计算并覆盖Refiner输出，不能用来验证“Refiner之后的结果”。这是一项流程语义冲突，不是单纯路径或显存问题。
- 当前固定Refiner归档仍为3,176 detections、255 images、49 tracks；候选的Refiner保留型后处理链 `remove_outside -> tracklet_agg -> concat_tracklets_by_jn -> concat_tracklets_by_reid -> tracklet_agg -> team -> team_side` 所有声明输入列均已存在。但该链是本地语义变体，不是历史Step 3精确复现，尚未执行。
- 新增空pipeline的隔离接口配置 `reproduction/configs/g10/g10_step3_interface_sngs10004_run1.yaml`、manifest `reproduction/manifests/g10_step3_interface_smoke_run1_sngs10004.json` 和CPU-only预检入口 `reproduction/gates/g10_step3_static_preflight.py`。未来接口smoke只做固定Refiner ZIP的TrackLab load/save，不改写任何业务字段，关闭评估、可视化、W&B和训练，输出使用全新本地命名空间。
- 正式静态预检在 `CUDA_VISIBLE_DEVICES=''`、空`PYTHONPATH/LD_LIBRARY_PATH`下退出码0；机器结果和说明位于 `reports/g10/20260819_step3_static_preflight/`。本轮没有导入或执行Step 3模块，没有GPU、推理、评估或训练，也没有运行已准备的接口smoke。G10-B和整个G10仍未通过。

#### G10-B Step 3空pipeline接口smoke run1（失败，未加载状态）

- 用户接受Refiner保留型方向并明确继续后，启动一次CPU-only空pipeline TrackLab load/save接口smoke。环境固定`CUDA_VISIBLE_DEVICES=''`、空`PYTHONPATH/LD_LIBRARY_PATH`，配置禁止Step-3模块、评估、可视化、W&B和训练；输入与输出使用不同且全新的本地路径。
- worker在Hydra配置组合阶段退出码1：launcher错误使用`--config-path`，使本地目录替换而不是附加TrackLab package config root；最终搜索路径包含本地配置和`sn_gamestate.configs`，但不含`pkg://tracklab.configs`，因此找不到实际存在于TrackLab包中的`state/save`配置组。
- 失败发生在dataset实例化、TrackerState构造和归档加载之前；没有业务输出ZIP、GPU、Step-3模块、评估、训练、fallback或自动重跑，且没有残留worker。失败日志和结构化结果位于`reports/g10/20260819_step3_interface_smoke_run1/`。
- 已只读确认既有成功enrichment launcher使用的是Hydra `--config-dir`，它会在保留TrackLab主配置根的同时附加本地目录。该差异解释了本次失败，但按失败即停规则没有修改run1现场或自动重跑。G10-B和整个G10仍未通过。

#### G10-B Step 3空pipeline接口smoke run2静态准备/预检（通过，尚未运行）

- 新建隔离run2配置、manifest、launcher和worker，全部使用新的`.runtime/g10/sngs10004_step3_interface/run2`、run2 cache和`reports/g10/20260819_step3_interface_smoke_run2`命名空间，不复用或覆盖run1日志与失败现场。
- run2相对run1失败原因的唯一运行语义修正是Hydra调用使用`--config-dir`附加本地配置，而非用`--config-path`替换TrackLab package配置根。pipeline仍严格为空，输入仍是固定Refiner run2归档，固定3,176 detections、255 images、49 tracks；关闭评估、可视化、W&B和训练，并禁止实例化任何Step-3模块。
- CPU-only静态预检固定`CUDA_VISIBLE_DEVICES=''`、空`PYTHONPATH/LD_LIBRARY_PATH`并在120秒timeout内退出码0。结果为`reports/g10/20260819_step3_interface_smoke_run2_preflight/result.json`，确认未来run2路径未占用、`execution_authorized=false`、`run2_executed=false`且无GPU操作。
- 本阶段只证明run2配置与守卫已准备，尚未再次执行Hydra compose、dataset实例化、TrackerState load/save或输入输出逐表一致性比较。G10-B和整个G10仍未通过。

#### G10-B Step 3空pipeline接口smoke run2（通过）

- 用户看到run2预检结论后明确授权执行一次。运行固定`CUDA_VISIBLE_DEVICES=''`、空`PYTHONPATH/LD_LIBRARY_PATH`，使用CPU、空pipeline、5400秒timeout、30秒heartbeat和全新run2 output/cache/report路径；未查询或使用GPU。
- Hydra通过`--config-dir`成功组合本地配置并保留TrackLab package配置根；固定SNGS-10004 dataset实例化、空Pipeline验证、TrackerState加载与保存全部完成，worker和launcher均退出码0，总墙钟30.54秒。
- 输出为`.runtime/g10/sngs10004_step3_interface/run2/states/sn-gamestate.pklz`，40,023,528 bytes，ZIP成员为`summary.json`、`10004.pkl`和`10004_image.pkl`。输出仍为3,176 detections、255 images、49 tracks；输入输出两个DataFrame按全部列、dtype、索引和值精确一致。
- 运行守卫记录只实例化dataset、禁用evaluator和offline engine三个顶层组件；`step3_modules_executed=[]`，没有评估、训练、可视化、fallback或自动重跑。结果和日志位于`reports/g10/20260819_step3_interface_smoke_run2/`。

#### G10-B Step 3球队颜色替换回放（完成，尚未写回生产字段）

- CPU-only脚本`experiments/soccerfactory_step3/compare_color_team_assignment.py`在禁用ReID合并后的同一49条轨迹上回放固定26维跨帧上半身HSV/色度特征，并与当前37/12 tracks的ReID KMeans逐轨比较。原始Refiner轨迹与号码拼接后的新ID严格一一对应；人工标签只在两种无监督预测完成后用于评价。
- 执行编排误判30秒命令返回为进程终止，意外产生两次同配置CPU回放；两次退出码均为0、墙钟363.40/366.53秒、峰值RSS 801,064/800,740 KiB，核心指标完全相同。在39条人工确认的外场轨迹上，ReID KMeans为32/39（82.1%，balanced accuracy 73.1%），颜色法为38/39（97.4%，balanced accuracy 98.1%）；颜色唯一错误为难辨track 20。两种方法在全部49条轨迹中分歧15条，颜色簇为34/15。
- 固定颜色接受规则接受42/49条轨迹，但当前Step 3的49条最终角色全部是player，不能可靠排除门将/裁判。因此只确认“颜色优于ReID KMeans区分本场外场两队”，不确认完整球队字段生产安全；本轮没有写回源归档，也未改变track_id、role或`bbox_pitch`。
- 结果、总对比图和冲突轨迹图位于`reports/g10/20260819_step3_color_team_replay/`。保留颜色法作为球队候选并放弃当前ReID KMeans球队模块；接入隔离Step-3候选前仍需角色门控。

#### G10-B 当前Step 3到训练PKL转换（通过）

- 新增固定manifest`reproduction/manifests/g10_current_step3_conversion_sngs10004.json`和薄入口`reproduction/gates/g10_current_step3_convert.py`，复用已经用历史黄金验证过的`convert_step3()`字段转换；输入严格为当前no-ReID Step 3 ZIP，输出使用全新本地命名空间。
- CPU-only正式运行退出码0、墙钟0.81秒、峰值RSS 187,504 KiB。输出`.runtime/g10/sngs10004_current_pipeline_conversion/run1/SNGS-10004.pkl`为339,907 bytes，包含连续255帧、3,176个人物实例和255帧有效相机参数；帧键、帧字段、人物字段及写后重载断言全部通过。
- 本轮没有GPU、模型推理、评估或训练，也没有与4,260人物的历史黄金PKL强行做标签一致性比较。历史训练PKL契约本身不携带`team`、`track_id`或`bbox_pitch`；因此本结果只证明当前固定样本成功转换到最终PKL结构，不证明这些信息会被下游使用或标签质量正确。
- 结果和日志位于`reports/g10/20260819_current_step3_conversion/`。G10-B固定样本生成链已到最终训练PKL；SoccerMaster消费端属于G10-C，尚未执行。

#### G10-C 当前SoccerFactory产物DataLoader消费smoke（通过）

- 新增固定manifest`reproduction/manifests/g10_current_pipeline_dataloader_smoke_sngs10004.json`和CPU入口`reproduction/gates/g10_current_pipeline_dataloader_smoke.py`。本地数据视图只用软链接暴露当前339,907-byte训练PKL与既有255张sn500图片，不复制或修改源资产。
- 真实`build_gsr_detection_dataloader()`按high-resolution配置的30帧video模式、512×512预处理运行；固定batch size 1、workers 0并关闭随机训练增强。Dataset构造出8个合法片段，起点为0、30、60、90、120、150、180和210。
- 正式CPU运行退出码0、墙钟61.02秒、峰值RSS 1,678,324 KiB。首个batch图像shape为`[1,30,3,512,512]`、float32、范围[-1,1]且有限；30帧annotation均含检测/角色/号码及lines/keypoints目标字段，张量长度对齐，所选0–29帧的194个人物与源PKL精确对应。全部12项机器断言通过。
- 本次没有模型构造、forward、loss、backward、optimizer、scheduler、训练或GPU。Matplotlib因home只读自动使用`/tmp`临时缓存，只产生非致命提示，无fallback或重跑。结果和日志位于`reports/g10/20260819_current_pipeline_dataloader_smoke/`。
- 因此可以确认：固定SNGS-10004从已准备255张图片开始，经当前本地SoccerFactory链到SoccerMaster训练消费端已经跑通。仍不能声称原始比赛视频端到端跑通，因为候选切片脚本尚未在隔离本地目录复现且固定 manifest 的 mapping 编号矛盾；也不证明角色、球队、号码或球场坐标语义质量正确。
- 本次证明当前Refiner归档能够通过TrackLab Step-3状态接口，不证明任何Step-3后处理逻辑或字段质量，也不改变Refiner时间连续性变差的已知风险。G10-B和整个G10仍未通过。

#### G10-C 原视频到准备图片的来源核对（候选已定位，manifest 矛盾）

- 已在本地 `scripts/segment_soccernet.py` 和远端同名脚本中找到历史切片候选：读取 `Labels-cameras.json`，只取 `Main camera center` 的 real-time 区间，边界各去约10帧，按最多750帧分段，将视频帧缩放到1920×1080并从 `000001.jpg` 重新编号。脚本的输出名是 `SNGS-{clip_id+10000:05d}`，并会写远端 `sn500` 和 `scripts/fail.txt`，因此本轮没有直接运行。
- 只读核对发现固定样本的编号语义与旧 manifest 不一致。`sn_2_clip.json` 中 `clip_id=4` 是 Chelsea–Burnley 的 `2278–2532`（255帧），按候选脚本命名正好生成 `SNGS-10004`；而旧 manifest/report 使用 `clip_id=10004` 的 Southampton–Liverpool `14439–14645`（207个包含首尾的索引）。
- 固定准备首帧与前者原视频帧的平均绝对像素差约 `0.853`，与后者约 `37.841`；结果、命令环境和未修改声明见 `reports/g10/20260819_raw_frame_source_check/result.json`。这支持实际图片来自 `clip_id=4`，但不是完整原视频复现。

#### G10-C 固定原视频到两条可视化链路（完成，单片段范围）

- 2026-08-19使用Chelsea–Burnley上半场原视频、`Labels-cameras.json`和修正后的mapping `clip_id=4`，在工作区隔离目录重新生成`SNGS-10004`的255张图片；运行记录确认255帧逐帧像素对应检查通过。所有写入均位于`.runtime/one_match/20260819_sngs10004_end_to_end/`，没有运行会写远端`sn500`或`fail.txt`的历史入口。
- CPU阶段把原视频、Step 1人物框/轨迹、球场线/关键点、相机标定/二维坐标、OCR/角色/球队、Refiner前后以及Step 3/PKL/DataLoader画成8张图。其输入仍是既有各阶段固定归档；这证明同一片段的lineage、接口和可视化闭环，不证明这些字段是人工真值。
- 随后一次物理GPU 0 inference-only worker以真实DataLoader张量`[1,30,3,512,512]`完成SoccerMaster五头前向，保存人物检测、球场线、关键点、CaptionClassification top-5和VideoCaption固定23短语检索top-5共5张图。两阶段退出码均为0、无重跑；VideoCaption结果是检索而非自由生成解说。
- 主报告为`reports/one_match/20260819_sngs10004_end_to_end/summary.md`，机器CPU结果为`.runtime/one_match/20260819_sngs10004_end_to_end/cpu_result.json`。因此现在可以表述为“固定单片段从原视频到SoccerFactory训练消费端，以及从同一视频到SoccerMaster五头可视输出，接口均已跑通”；不能外推其他比赛、完整数据集质量、上游原版Step 3等价性或完整训练完成。
- 已新增不具执行授权的候选清单 `reproduction/manifests/g10_raw_video_lineage_candidate_sngs10004.json`，把序列名、mapping `clip_id`、原视频区间和旧 manifest 冲突并列记录；旧 manifest/report 不被覆盖。
- 因此原视频到 `sn500` 图片的实现候选已恢复，但固定 lineage 仍需在隔离本地目录重现；G10 和“原始视频端到端跑通”仍未通过。

#### G10-B Refiner保留型Step 3 CPU后处理首轮（完成，仅保留为诊断）

- 新增并运行单一研究脚本`experiments/soccerfactory_step3/run_refiner_preserving_step3.py`，从当前Refiner归档依次执行`remove_outside -> tracklet_agg -> concat_by_jersey -> concat_by_reid(threshold=0.1) -> tracklet_agg -> team_cluster -> team_side`。显式排除会重算`bbox_pitch`的标定链和全部模型推理；固定CPU-only，无训练或评估。
- 首次在模块导入阶段被OpenMIM尝试创建只读`/home/tianlin/.cache/mim`阻塞，未创建正式report/output。只精确抑制这一条未使用缓存目录创建后一次重跑退出码0；没有扩大为依赖或Harness审计。
- 3,176条检测全部保留，`bbox_pitch`逐值完全不变且没有合并后同帧冲突。`remove_outside`删除0行，号码模块没有真正合并原始轨迹；`concat_by_reid`是轨迹数从49降至32的唯一来源，形成9个多原始ID组，最大组包含6个原始轨迹。
- 角色结果仍为3,176/3,176个player；号码非空由1,801增至1,948。球队输出从历史left 3,119/right 57改为left 1,707/right 1,469（20/12 tracks），但全部坐标仍只覆盖x=-50.4911至-16.6466米，因此不能把更均衡的KMeans输出解释为球队正确。
- 结果、模块变化图和最终球场图位于`reports/g10/20260819_step3_refiner_preserving_cpu/`；隔离输出ZIP位于`.runtime/g10/sngs10004_step3_refiner_preserving/run1/states/sn-gamestate.pklz`。正式判定为`diagnostic_only_not_production`，G10-B和整个G10仍未通过。

#### G10-B Step 3 ReID合并组复核（完成，否定当前模块）

- CPU-only生成9个ReID合并组总览和逐组人物框接触图，并计算组内所有原始轨迹均值embedding的两两余弦距离；没有重新运行检测、ReID模型、Refiner或其他Step-3模块。
- 9组中4组包含距离超过配置阈值0.1的成员，证明当前实现存在链式合并：每次局部合并可以通过阈值，但最终组内任意两成员不受该阈值约束。最大6轨迹组的最大距离为0.15495，共6对超过0.1。
- 目视证据确认至少一组错误：new ID 16把疑似39号与明确14号球员合并。new ID 3的原始track 3和24自身都出现酒红/蓝色球衣切换，说明输入轨迹已有ID污染，基于整轨均值embedding继续合并会放大错误。
- 结果为`reports/g10/20260819_step3_refiner_preserving_cpu/merge_review_summary.json`，总览为同目录`merge_groups_overview.png`，九张细图位于`reid_merge_groups/`。当前`concat_by_reid(threshold=0.1)`不保留；G10-B和整个G10仍未通过。

#### G10-B Step 3禁用ReID合并消融（完成，保留结构决策）

- CPU-only运行与首轮相同的后处理链，仅移除`concat_by_reid`。首次因清空`PYTHONPATH`后使用仓库包导入失败，未创建输出；改为同目录导入后一次针对性重跑退出码0，无GPU、训练或模型推理。
- 3,176条检测、49条原始轨迹和全部`bbox_pitch`逐值保留；没有删除、跨轨合并或同帧冲突。号码非空保持1,801条，不再因错误ReID合并后的二次投票扩散到1,948条。
- 最终球队为right 2,598/left 578行，对应37/12 tracks；角色仍为49/49 tracks、3,176/3,176行全部player，坐标范围仍为x=-50.4911至-16.6466米。带ReID版本的20/12 tracks表面均衡不能视为改进，因为它建立在已确认错误的轨迹合并上。
- 结果与图位于`reports/g10/20260819_step3_no_reid_ablation/`，隔离ZIP位于`.runtime/g10/sngs10004_step3_no_reid/run1/states/sn-gamestate.pklz`。保留“禁用ReID合并”作为更安全的结构候选，但当前球队/角色/坐标质量仍不合格；G10-B和整个G10仍未通过。

### 已知工作区状态

- 2026-08-20 UTC完成本地代码与checkpoint兼容分层：约25 GiB既有物理资产未移动或复制；新实验统一读取`configs/assets/local_assets.yaml`，checkpoint分类入口通过`.local_assets/checkpoints/{soccermaster,soccerfactory}/official/`软链接解析，跨系统转换代码归入`src/soccerfactory_adapters/`。历史Gate manifest和产物路径保持不变；真实Step-3归档的CPU最小复核得到255帧、3,390人物和255帧有效相机参数，不改变任何Gate结论。
- `data/video_caption.py` 存在复制前已有的一行注释删除，不得回滚或覆盖。
- 2026-08-12 用户批准以“上游代码保持原位置、复现与改进分层”的方式整理本地仓库。上游 `models/`、`data/`、`configs/`、`train.py` 和 `eval.py` 未移动、未修改。
- Gate 入口现位于 `reproduction/gates/`，固定输入现位于 `reproduction/manifests/`；复现导航为 `reproduction/README.md`，改进实验登记为 `experiments/README.md`。
- 运行证据按 `reports/g1/` 至 `reports/g7/` 归档；G5 临时数据视图现位于 `.runtime/data_views/g5/`；G7 retry1/retry2 的 step-3 checkpoint 分别位于 `outputs/g7/20260812_retry1_interrupted/step_000003` 和 `outputs/g7/20260812_retry2_full_2epochs/step_000003`。
- 整理后的纯静态检查通过：6 个 Gate 脚本均通过 `ast.parse`，3 个 manifest 均通过 JSON/schema 检查，23 个 report 文件、3 条只读运行时链接和 7 个 G7 output 文件均存在；G5/G6/G7 的 7 个已知证据哈希保持不变，G7 checkpoint 的 5 个 manifest 文件共 302,695,666 bytes，大小与 SHA256 再次通过。Markdown 相对链接和 `git diff --check` 也通过。
- `AGENTS.md`、`README.md`、`REPRODUCTION_STATUS.md`、`reproduction/`、`experiments/`、`reports/`、`outputs/`、`.runtime/`、`.local_deps/`、`.local_envs/` 和 `.conda_pkgs/` 均是本地资产；其中大型或可重建路径受 `.gitignore` 影响，具体状态以每次运行清单为准。
- `.conda_pkgs:/` 是首次 Conda 缓存参数分隔符错误留下的约 20 KB 异常目录，内含零字节 partial 文件；尚未删除。
- G9 run1 结束后本地文件系统剩余空间约 187 GiB。

## 推断

- 本地环境本次 G2 墙钟 4:07.07，历史共享环境 G2 为 15:32.90；本次观测明显更短，但受文件缓存、GPFS 当时负载等因素影响，不能仅凭两次运行把全部差异归因于环境迁移。
- 由于 SigLIP2 和 `epoch_19` 仍在 GPFS，后续首次或冷缓存 checkpoint 读取仍可能受共享存储速度影响。
- 在冻结 backbone/text、只训练 25,220,119 参数 CaptionClassification 头的当前协议中，retry2 的实测峰值 GPU reserved 为 9,481,224,192 bytes；这不能直接推断解冻 backbone 或多任务训练的资源需求。
- G8 run5 冻结 backbone/text、训练五头的实测每-rank 峰值 GPU allocated/reserved 约为12.31/15.59 GiB；该结果不能外推到解冻 backbone 的原始完整多任务训练。
- G9 run1 和 run3 已分别证明原样 FP32 与仅切换 BF16 的 30 帧、512×512、解冻 vision 协议都无法在当前单卡约 80 GiB 的 H800 上完成首个 detection forward。run4 已确认 BF16 + 24-block activation checkpointing 能把首个完整 optimizer step 的 PyTorch 峰值 allocated 降至约 17.38 GiB/rank；后续多步吞吐、显存稳定性和完整训练时长仍不能由单步结果外推。

## 未知

- 本地环境 G2 等价性已经确认；不同缓存与 GPFS 负载条件下的稳定加速幅度仍未知。
- G5 的目标固定小规模指标和同一进程内两遍重复性已确认；完整测试集指标和独立进程/主机间重复性仍未知。
- G7 的固定小规模单任务训练协议、恢复和评估链路已确认。G8 run5 已确认冻结 backbone/text 时五头多任务的两卡、两 epoch、8步 optimizer、validation、事务式 checkpoint 和功能性恢复链路；位级 exact resume 和严格 CUDA 确定性仍未满足。G9 run6已确认解冻vision、五头多任务、BF16 + checkpointing可稳定运行超过8,000步，但18小时timeout时只完成约51.8%，没有完整epoch、evaluation或checkpoint/resume，因此G9仍未通过。
- G5 已验证 `SNGS-116` 两个真实 clip 上的检测、球场线和关键点评估；其他序列和更大数据范围的表现仍未知。
- G7 retry2 已确认完整 2 epochs、最终指标、全部最终断言、step-3 exact-resume 探针和整段峰值资源；更大数据规模、不同 seed、独立进程重复性和最终模型 checkpoint 保存仍未知。
- SoccerReplay-1988 的完整 train/valid 视频可用率尚未盘点；目前只有标注数量和三个 train 视频抽查结果得到确认。
- G10-A已发现SoccerFactory最终伪标签中的多类噪声；G10-B已静态确认历史Step-3到逐序列PKL兼容契约。Step 1 run5、前置enrichment run2及255帧coord-only Refiner forward均已在固定SNGS-10004上完成；Refiner输出结构合规但时间平滑性变差。新字段和坐标的真值质量、独立重跑一致性、其他序列及新产物的Step 3仍未知。

## 风险

- `pip check` 报告 `decord 0.6.0 is not supported on this platform`，但实际 `import decord` 已成功。
- 本地 NumPy 1.26.4 目前由 Python 元数据管理，不再保留错误的 Conda 2.0.1 记录；以后不得使用 Conda 操作 NumPy，除非先制定并批准一致性迁移方案。
- 本地环境中 `tracklab.pth` 和 `sn_gamestate.pth` 仍指向只读 GPFS 路径；相关功能仍可能受 GPFS 速度影响。
- `.local_envs/`、环境 tar 归档和 Conda 缓存不得提交到 Git；清理前需要用户批准。
- 多个上游遗留脚本仍含绝对 `/remote-home/...` 路径，个别脚本包含远端输出目标；未经逐个静态审查，不得把它们当作安全入口直接运行。
- G5 指标来自为链路验证设计的固定小规模清单；每个 Caption 类别只有一个样本，不应用于宣称数据集总体性能。
- G6 只在固定的缓存 backbone 特征上更新最后的线性分类器；它证明标签/loss/梯度/optimizer 的最小链路，不能替代 G7 的完整单任务训练。
- 原训练入口会把 VideoCaption `test` split 用作 epoch 评估，并且当前 checkpoint/RNG 状态不足以证明 exact resume；未经本地受限入口隔离和机器断言，不得直接用于 G7。
- `MatchTime` 与 `SoccerNet-v2` 的 caption JSON 仍在，但视频软链接目标当前缺失；把它们保留在 G7 数据集列表会在解码阶段失败。
- 原始 full-backbone high-resolution 多任务日志的每-rank 峰值约 123,115 MiB，超过 80 GB H800；G9 run1 FP32 与 run3 BF16 均在首个 forward 接近耗尽 80 GiB 后 OOM。run4 已确认 checkpointing 能完成一个 optimizer step，但它增加重计算开销，且一次受控探针不能证明长时间显存无增长、数据无坏样本或 checkpoint/resume 可靠；未经新的完整训练方案审查和 GPU 授权，不得直接扩展为 20 epochs。
- G8 run5 依赖 `warn_only=True` 兼容不具确定性实现的 CUDA 操作；虽然数值差异远低于预设功能容差，但它不能保证不同主机、驱动、GPU 或更长训练中的漂移上界。
- 后续长时间命令仍必须使用 timeout、心跳和明确退出码。
- G10 在 G9 完整训练前提前执行；因此任何 G10 结果都必须保持为隔离的流水线或小规模实验结论，不能表述为相对完整 G9 的总体提升。
- G10-B Step 1仍依赖只读GPFS上的TrackLab/sn-gamestate源码和约757 MB的三份模型资产。run5已在固定样本上完成实际Hydra组合、CUDA初始化和模型forward，但不能排除冷缓存、其他主机、其他序列或重复运行中的依赖、资源与数值问题；后续实际运行仍不得绕过单卡授权守卫、全新输出目录、timeout和失败即停止策略。
- G10-B Step 1 run1曾在共享存储/NAS等待中耗尽一小时timeout且GPU全程空闲；run5热缓存下总墙钟仅约93秒。两者差异很大，不能把run5耗时外推到冷缓存或其他主机，也不能用单纯扩大timeout代替分阶段观测。
- 启动诊断曾暴露Hydra搜索路径、结果落盘以及OpenMIM/Matplotlib缓存副作用。本地4-worker、缓存重定向和精确抑制适配已随run5实测通过，但它们仍是本地Harness边界，不能表述为上游生产入口已修复或无需审计。
- 分阶段Harness已覆盖本次固定样本的权重加载、CUDA kernel、engine和255帧推理，但不能证明所有上游副作用、其他序列或后续Refiner/Step 3都被覆盖。未来运行仍必须严格单卡、保留完整日志与失败现场，并在首次失败后停止分析，不能自动重试。
- 保存版Refiner `inference.py` 会删除同名既有输出ZIP并在当前工作目录写临时pickle，且默认750帧配置与固定255帧样本不匹配。本地255帧隔离配置、全量副作用审查、新输出命名空间和授权守卫已用于run2并完成一次固定运行；任何未来Refiner重跑仍必须fresh `nvidia-smi`、资源/进程报告和新的单卡授权，且不得绕过launcher直接运行上游入口。
- 历史`combination`配置自身启用评估和ZIP append，并包含缺权重自动下载、torchvision预训练缓存访问以及Qwen `device_map="auto"` 等副作用；即使生产链已定位，也不得直接运行原配置。未来前置enrichment必须关闭评估/可视化/W&B、限制单卡可见性、固定现有权重并使用全新本地状态路径。
- 前置enrichment的Qwen资产为五个逻辑总计约16.58 GB的分片，并可能对最多3176个检测候选执行OCR；即使legibility阈值会过滤一部分，实际通过数量、冷缓存启动时间、单卡峰值显存和总耗时仍未知。28,800秒只是未来run1安全上限，不是性能预测。
- 前置enrichment run1已在模型加载前的dataset阶段耗尽1,200秒子阶段timeout；其长达797秒的TrackLab导入和历史冷热运行差异支持共享存储/冷启动抖动这一推断，但没有子阶段或系统调用证据。不得直接放大timeout、复用run1目录或把同命令重跑当作诊断。
- run2诊断首轮预检的AST排序错误已由retry1静态修复，但实际run2又在完整Hydra快照强制解析sweep-only字段时失败。不得把task字段已比较写成整个compose阶段通过，也不得通过静默删除Hydra字段或复用run2目录掩盖失败。
- run2在本次缓存/负载条件下dataset只用106.377秒并完成全链路，但run3曾观测wrapper导入1,136.700秒和 `cxiWaitEventWait`；单次快速结果不能保证冷缓存、其他主机或其他序列耗时。不得把单样本结构通过写成伪标签质量、Refiner或整个G10已通过。
- run2静态审计发现role全为player、team为left 3,119/right 57、号码仅11个唯一非空值且pitch-x全负。没有真值时这些分布既不能判错，也不能被静默当作高质量伪标签；进入后续训练或总体质量结论前必须保留这一风险。保存版Refiner默认750帧仍与本样本255帧严格冲突。

## 下一步

- G0 至 G7 均已达到当前 Harness 定义的最小证明范围；G7 retry2 完整 2 epochs 并以退出码 0、8/8 机器断言通过。
- G8 run5 已达到当前 Harness 定义的功能性恢复证明范围并以退出码0通过；位级 exact resume 仍明确为 false。
- G9 run1 已验证原样 FP32 在首个 detection forward OOM；run2 BF16 按用户要求中止；run3 已确认仅切换 BF16 仍 OOM；run4 已确认 BF16 + 24-block checkpointing 完成一个真实五头 optimizer step并以探针退出码 0 通过。G9 完整训练仍未通过。
- checkpointing 适配版 G9 的只读预算与恢复方案已于 2026-08-13 形成；用户随后明确批准暂停扩展 G9并隔离提前进入 G10。G9 状态不变，完整训练未通过。
- G10-A的既有只读数据质量审计已经正式登记，范围受限；G10-B Step 1 run5和前置enrichment run2均已完成固定样本运行。run2新归档的ZIP/DataFrame一致性、全部字段、dtype/shape/值域、逐帧/逐track覆盖和相机参数结构已通过CPU静态审计；全新隔离的255帧Refiner probe也已完成配置/资产/副作用/运行守卫静态预检。字段语义质量、实际Refiner checkpoint/model forward、Step 3和整个G10仍未通过。
- 2026-08-18新增CPU-only ID Switch候选回放：在固定SNGS-10004归档中，连续帧异常分数整体较低，未形成已确认切换；49条轨迹中41条出现过多帧间断，筛出20个间断后重现候选。结果与图位于 `reports/g10/20260818_track_switch_candidates_v2/`，原始 `track_id` 未改写。下一步仅人工复核前5个候选，不能把启发式分数当作ID准确率。
- 2026-08-19接管批次1经用户明确批准后完成：13项主干/SoccerFactory权重与固定SNGS-10004输入复制到`.local_assets/`，清单apparent size为26,492,433,131 bytes；脚本退出码0，13项目标均存在，完成后可用空间136,747,585,536 bytes。4份YAML和1份JSON本地配置解析通过且无`/remote-home`引用。该结果证明资产迁移完成，不等于本地权重forward、完整训练或字段语义质量通过。
- 2026-08-19用户批准GPU7本地推理后，SoccerMaster固定30帧五头inference-only动态复核退出码0：日志明确从`.local_assets/checkpoints/soccermaster_epoch19`加载backbone、text encoder及五个任务头，输入图片通过本地`.local_assets`数据视图读取。五头和5张图齐全，墙钟99.573秒，峰值allocated/reserved为6,422,995,456/8,529,117,184 bytes，`training=false`。这证明固定推理路径本地化，不代表完整训练或总体语义质量。
- 2026-08-19用户批准GPU7后，SoccerFactory本地Step 1固定255帧inference-only退出码0：TrackLab从`vendor/soccerfactory/tracklab`导入，YOLO/PRTReID/HRNet从`.local_assets`读取；墙钟34.037秒，峰值allocated/reserved为3,689,509,888/4,756,340,736 bytes，得到3,390条检测和48条轨迹，`training=false`、`evaluation=false`。历史共享环境run5为3,176条/49条；环境和运行时不同，当前只证明本地接口运行，不能声明数值等价、质量提升或退化。
- 2026-08-19用户批准GPU7后，本地enrichment对新Step 1的3,390条检测/48轨迹运行退出码0：TrackLab、标定、legibility和Qwen均从vendor/`.local_assets`读取；墙钟95.097秒，峰值allocated/reserved为19,076,444,672/19,639,828,480 bytes。3,390条均有`bbox_pitch`、role、team，1,999条号码非空，255帧均有相机参数；这只证明字段生成，不证明字段准确。
- 2026-08-19用户批准GPU7后，本地Refiner源码/checkpoint完成255帧coord-only inference，GPU进程退出码0、墙钟5.006秒，3,390行和48轨迹保留且全部坐标被更新。首次产物检查把六字段`bbox_pitch`字典误当四值数组而误报失败；一次CPU检查器修复后复用既有输出通过，未重跑GPU，误报记录保留在结果JSON中。
- 随后CPU-only保留版Step 3、训练PKL转换和真实SoccerMaster DataLoader消费均退出码0：Step 3保留3,390行/48轨迹且无同帧冲突，PKL含255帧/3,390人物/255帧有效相机参数，DataLoader形成8个合法片段并输出`[1,30,3,512,512]`首批次。固定样本从本地资产到最终消费端的接管闭环完成。
- **唯一下一步**：不再为资产接管自动运行GPU。后续若继续研究，先从角色、球队、ID Switch或坐标真值中选择一个独立质量问题并制定最小实验；G9完整训练仍保持未通过。
