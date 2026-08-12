# 可复现环境、资产与 checkpoint 管理

## 状态

- 当前状态：`待研究`
- 优先级：`P0`
- 目标：让代码可分享、环境可重建、资产可定位，同时避免复制不必要的大文件

## 当前已知情况

- 当前仓库已经是 Git 仓库，并连接上游 Soccer-Backbone。
- 真正的下游解说代码仍位于只读 UniSoccer 目录，尚未整理到当前仓库。
- 代码中存在 `/mnt/...`、`/remote-home/...` 等硬编码路径。
- 固定 `tracklab2` 环境可运行 SoccerMaster backbone，但缺少 `peft`。
- 仓库内 `.local_envs/SoccerMaster-repro` 是一次失败留下的不完整环境，不能执行 Python。
- MatchTime 视频软链接目标缺失，但标注、权重和部分 UEFA 视频仍可读。
- 完整解说 checkpoint 约 17GB，逐 epoch 保存会快速耗尽个人磁盘。

## 建议的资产边界

```text
Git仓库：代码、配置、文档、测试和小型清单
个人环境目录：Conda/venv，不进入Git
只读共享存储：原始数据和基础权重
实验输出目录：日志、预测和checkpoint，不进入Git
```

## 环境复现

最终应保存：

- 明确的 Python 版本。
- 直接依赖及固定版本。
- CUDA/PyTorch 兼容信息。
- 可重建的 environment 文件或 lock 文件。
- CPU smoke test 和 GPU smoke test 命令。

不直接移动已有 Conda 环境；Conda 环境内部可能含绝对前缀，应在目标位置重新创建。

## 路径管理

所有数据、权重和输出路径通过配置或命令行传入。代码中不保留个人用户名和服务器专用绝对路径。

建议使用逻辑名称：

```text
data_root
pretrained_model_root
checkpoint_root
output_root
```

运行开始时打印解析后的绝对路径，并在资产缺失时立即失败。

## 资产清单

为大型资产记录：

```text
逻辑名称
绝对路径或获取方式
文件大小
哈希（可行时）
只读/可写属性
许可证或来源
对应实验
```

清单不包含密码、token 或其他秘密。

## Checkpoint 策略

解说任务优先保存：

- SoccerMaster 视觉编码器状态。
- Q-Former、query token 和投影层。
- Llama LoRA adapter。
- tokenizer 新增 token 信息。
- 完整配置、epoch、优化器和随机状态（需要恢复训练时）。

基础 Llama 权重通过名称和哈希引用，不在每个 checkpoint 中重复保存。

默认保留：

```text
best
last
少量周期checkpoint
```

在实现轻量保存前，需要先验证它能够严格恢复推理结果和继续训练状态。

## Git与来源

- 不在当前仓库中嵌套另一个 Git 仓库。
- 从 UniSoccer 提取代码时只复制任务所需的最小部分。
- 保留来源、原始提交、许可证和本地修改说明。
- 在公开未发布代码或权重前先确认授权。
- 环境目录、数据、权重、输出和日志应加入 `.gitignore`。

## 完成标准

- [ ] 新机器可以按照文档重建环境。
- [ ] CPU smoke test 不依赖个人 shell 设置。
- [ ] 数据和权重路径不再硬编码在模型源码中。
- [ ] Git 状态不会出现本地环境和大文件。
- [ ] checkpoint 可以恢复推理，并能说明依赖的基础权重。
- [ ] 分享代码时不会包含未经授权的数据、权重或第三方源码。
