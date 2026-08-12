# SoccerMaster 长期工作规则

本文件适用于 `/home/tianlin/SoccerMaster` 及其所有子目录中的后续复现工作。

## 文档职责

- 本文件只规定长期有效的强制规则和授权边界。
- `docs/HARNESS.md` 规定 Gate 的执行协议、证据标准、数据与 checkpoint 契约。
- `REPRODUCTION_STATUS.md` 只记录当前进度、已经获得的证据、未知项和下一步，不把临时状态提升为长期规则。
- `docs/future_improvements/` 保存研究假设和候选改进。未经验证的假设不得写成已确认缺陷，也不得直接变成基线行为。
- `reports/` 保存支撑结论的日志和机器可读结果；文档中的结论应能定位到相应证据。

## 路径与环境

- 工作目录固定为：`/home/tianlin/SoccerMaster`
- 原始代码、数据、权重目录固定为：`/remote-home/haolinyang/sports/Soccer-Backbone`
- 原始目录永久只读，绝不能修改、删除、覆盖或写入其中的任何内容。
- 所有代码修改只能发生在 `/home/tianlin/SoccerMaster`。
- 共享参考 Python 固定为：`/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`。该共享环境只读，不得安装、升级、删除或修改其中的任何内容。
- 实际 Gate 使用的 Python 和环境必须以 `REPRODUCTION_STATUS.md` 中当前已确认状态及用户授权范围为准；两者不一致时停止执行并报告。
- 不依赖 `conda activate`。
- 使用本地环境时，必须明确记录 `PYTHONPATH` 和 `LD_LIBRARY_PATH`；不得混用其他 Conda 环境的 `site-packages`。
- 优先使用绝对路径。
- 不把 Markdown 链接格式写入 shell 命令或文件名。

## 安全规则

- 不使用 `sudo`。
- 不输出或读取密码、SSH 私钥、Codex `auth.json`、访问令牌等秘密。
- 禁止使用 `git reset --hard`、`git checkout --`、`rm -rf` 等破坏性命令。
- 不使用 `danger-full-access` 或 `--yolo`。
- 大文件复制前，必须先报告来源、目标、大小和目标文件系统剩余空间，并等待用户明确批准。
- 本地环境目录 `.local_envs/`、本地 Conda 缓存和环境归档都是大型未跟踪资产；不得提交到 Git，删除或清理前必须先报告精确目标和可恢复性，并等待用户批准。
- 任何 GPU 操作前，必须重新执行 `nvidia-smi`，报告显存占用，并等待用户明确批准。
- 未经用户明确批准，不启动训练。

## 执行与报告规则

- 执行、修改或判定任何 Gate 前，必须完整阅读 `docs/HARNESS.md`。
- 每次只推进一个 Gate；完成后报告并停止，不自动进入下一 Gate。
- 只读分析和文档维护不算推进 Gate；运行该 Gate 的验证命令、改变其状态或形成通过/失败结论，均算推进 Gate。
- 报告必须明确区分：已确认、推断、未知、风险、下一步。
- 长时间命令必须设置 `timeout`、提供心跳输出，并报告明确退出码。
- Gate 的“通过”必须同时具备可重复命令、明确退出码、机器可判定的断言、可定位日志和输入资产记录；仅仅“没有抛出异常”不足以判定通过。
- 每次 Gate 执行必须记录实际使用的 Python、解析后的配置、代码版本与 dirty 状态、输入资产、命令、时间、资源峰值和输出路径。记录中不得包含秘密。
- 不得为了让命令成功而静默跳过缺失资产、吞掉异常、切换模型或设备、关闭任务头，或者修改 batch size、精度、数据范围和其他实验语义。
- 如需 fallback，必须在执行前定义触发条件，并在结果中记录触发原因、实际行为及其对结论范围的影响。
- 数据检查、smoke test 和审计应使用明确断言和非零失败退出码；只打印 warning 不算验证失败条件已经被覆盖。

## 证据与可复现性规则

- `已确认` 只用于有代码、日志、测试或资产清单直接支持的事实；`推断` 必须说明推理依据；证据不足的内容保留为 `未知`。
- `目标不变量` 表示系统应满足但尚未被验证的性质，不得与已确认事实混写。
- 声称“可恢复训练”时，必须覆盖恢复模型、优化器、scheduler、scaler（如有）、Python、NumPy、PyTorch CPU、所有 CUDA device、DataLoader worker、distributed sampler 和每个 rank 所需的状态。
- 只恢复模型参数或模型与优化器时，必须准确描述为对应范围的恢复，不得称为 exact resume。
- 声称“可复现推理”时，至少需要证明同一代码、配置、checkpoint、输入和确定性解码设置产生满足既定容差的相同输出。
- checkpoint 目录存在不代表 checkpoint 完整；是否允许自动恢复必须依据 `docs/HARNESS.md` 中的完整性契约判断。

## Gate 顺序

1. G0 资产定位
2. G1 Python/依赖/CUDA 环境
3. G2 完整 checkpoint 加载
4. G3 随机张量 forward
5. G4 单个真实视频
6. G5 固定小规模评估
7. G6 tiny overfit
8. G7 单任务训练
9. G8 小规模多任务训练
10. G9 完整训练
11. G10 SoccerFactory 分支
