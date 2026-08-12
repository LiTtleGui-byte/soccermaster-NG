# SoccerMaster 复现进度

更新日期：2026-08-12

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
| G7 单任务训练 | 未通过（retry1 用户中止） | retry1 已进入真实训练，完成 epoch 0 的训练与 train/valid 评估并继续到 global step 111；在 epoch 1 未完成时按用户要求停止，退出码 141，无最终结果 JSON/完整断言 |
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

### 已知工作区状态

- `data/video_caption.py` 存在复制前已有的一行注释删除，不得回滚或覆盖。
- 2026-08-12 用户批准以“上游代码保持原位置、复现与改进分层”的方式整理本地仓库。上游 `models/`、`data/`、`configs/`、`train.py` 和 `eval.py` 未移动、未修改。
- Gate 入口现位于 `reproduction/gates/`，固定输入现位于 `reproduction/manifests/`；复现导航为 `reproduction/README.md`，改进实验登记为 `experiments/README.md`。
- 运行证据按 `reports/g1/` 至 `reports/g7/` 归档；G5 临时数据视图现位于 `.runtime/data_views/g5/`；G7 step-3 checkpoint 现位于 `outputs/g7/20260812_retry1_interrupted/step_000003`。这些是同一本地文件系统中的路径整理，没有重跑 Gate、没有读取模型、没有使用 GPU。
- 整理后的纯静态检查通过：6 个 Gate 脚本均通过 `ast.parse`，3 个 manifest 均通过 JSON/schema 检查，23 个 report 文件、3 条只读运行时链接和 7 个 G7 output 文件均存在；G5/G6/G7 的 7 个已知证据哈希保持不变，G7 checkpoint 的 5 个 manifest 文件共 302,695,666 bytes，大小与 SHA256 再次通过。Markdown 相对链接和 `git diff --check` 也通过。
- `AGENTS.md`、`README.md`、`REPRODUCTION_STATUS.md`、`reproduction/`、`experiments/`、`reports/`、`outputs/`、`.runtime/`、`.local_deps/`、`.local_envs/` 和 `.conda_pkgs/` 均是本地资产；其中大型或可重建路径受 `.gitignore` 影响，具体状态以每次运行清单为准。
- `.conda_pkgs:/` 是首次 Conda 缓存参数分隔符错误留下的约 20 KB 异常目录，内含零字节 partial 文件；尚未删除。
- 当前本地文件系统剩余空间约 189 GiB。

## 推断

- 本地环境本次 G2 墙钟 4:07.07，历史共享环境 G2 为 15:32.90；本次观测明显更短，但受文件缓存、GPFS 当时负载等因素影响，不能仅凭两次运行把全部差异归因于环境迁移。
- 由于 SigLIP2 和 `epoch_19` 仍在 GPFS，后续首次或冷缓存 checkpoint 读取仍可能受共享存储速度影响。
- 在冻结 backbone/text、只训练 25,220,119 参数 CaptionClassification 头的前提下，G7 的额外 optimizer/gradient 显存应明显小于全量 backbone 反向；retry1 因中止未完整采集峰值，仍需在后续完整受控运行中测量。

## 未知

- 本地环境 G2 等价性已经确认；不同缓存与 GPFS 负载条件下的稳定加速幅度仍未知。
- G5 的目标固定小规模指标和同一进程内两遍重复性已确认；完整测试集指标和独立进程/主机间重复性仍未知。
- G6 的限定分类器 tiny-overfit 链路已确认；backbone 解冻后的反向传播，以及 G8 以后的训练行为仍未知。
- G5 已验证 `SNGS-116` 两个真实 clip 上的检测、球场线和关键点评估；其他序列和更大数据范围的表现仍未知。
- G7 retry1 已执行完整 CaptionClassification 头的 backward、optimizer、scheduler、一个完整 epoch 的 train/valid、事务式 step-3 checkpoint 和恢复后继续训练；但运行在第二个 epoch 中途停止，没有最终结果 JSON，因此完整 2 epochs、最终指标、全部最终断言、严格 exact resume 和整段峰值资源仍未确认。
- SoccerReplay-1988 的完整 train/valid 视频可用率尚未盘点；目前只有标注数量和三个 train 视频抽查结果得到确认。

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
- 后续长时间命令仍必须使用 timeout、心跳和明确退出码。

## 下一步

- G0 至 G6 均已达到当前 Harness 定义的最小证明范围；G7 retry1 已进入真实单任务训练，但在第二个 epoch 中途由用户停止，因此 G7 未通过。
- G7 的只读设计、固定 manifest、受限验证脚本和静态审查已经完成；首次运行尝试因受限沙箱看不到宿主机 CUDA 而失败，retry1 修正前置检查后运行到 global step 111。
- 唯一建议的最小下一步：在用户新的明确授权下，对保持完整 2 epochs 的 G7 retry2 做一次只读运行前审查，使用新的报告和输出路径以保留 retry1 证据；审查完成后停止，不自动启动训练。
- 未经用户新的明确批准，不再运行 G7，不进入 G8。
- 任何 GPU 操作前必须重新执行 `nvidia-smi`、报告显存占用并等待用户明确批准。
