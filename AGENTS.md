# SoccerMaster 长期工作规则

本文件适用于 `/home/tianlin/SoccerMaster` 及其所有子目录。它规定长期安全边界和 Gate 执行纪律；当前事实与进度以 `REPRODUCTION_STATUS.md` 为准，具体判定协议以 `docs/HARNESS.md` 为准。

## 1. 固定边界

- 工作目录：`/home/tianlin/SoccerMaster`
- 原始代码、数据和权重：`/remote-home/haolinyang/sports/Soccer-Backbone`
- 原始目录永久只读：绝不能修改、删除、覆盖、编译或写入其中的任何内容。
- 所有源码、脚本、配置和文档修改只能发生在本地工作目录。
- 优先使用绝对路径；shell 命令和文件名中不得出现 Markdown 链接语法。

## 2. Python 与依赖

- 共享参考 Python：`/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`。
- 共享环境永久只读，不得安装、升级、删除或修改包。
- 当前已验证的本地 Gate Python：`/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`。
- 不依赖 `conda activate`；命令中直接使用 Python 绝对路径。
- 使用本地环境时，必须显式记录 `PYTHONPATH` 和 `LD_LIBRARY_PATH`。
- 不得通过 `PYTHONPATH` 混入其他 Conda 环境的 `site-packages`。
- 实际 Gate 环境必须与 `REPRODUCTION_STATUS.md` 的当前状态一致；不一致时停止并报告。
- `.local_envs/`、`.conda_pkgs/`、`.local_deps/` 和环境归档属于大型本地资产，不得提交。删除或清理前必须报告精确目标、大小和可恢复性并等待批准。

## 3. 安全规则

- 不使用 `sudo`。
- 不读取或输出密码、SSH 私钥、Codex `auth.json`、访问令牌等秘密。
- 禁止 `git reset --hard`、`git checkout --`、`rm -rf` 及等价破坏性操作。
- 不使用 `danger-full-access`、`--yolo` 或其他绕过审批的模式。
- 保留用户已有 Git 改动；不得回滚或覆盖无关 dirty 文件。
- 大文件复制前，先报告来源、目标、大小和目标文件系统剩余空间，并等待明确批准。
- 不向原始目录、共享环境或其他用户目录安装、编译、复制或写入文件。
- 上游遗留脚本可能包含硬编码的远端读写路径；任何尚未由当前 Gate 验证过的脚本，运行前必须静态审查全部输入、输出和副作用，不能仅凭文件名判断安全。

## 4. GPU、评估与训练授权

- 任何 GPU 操作前都必须重新执行 `nvidia-smi`。
- 必须报告每张相关 GPU 的显存、利用率和现有进程，然后等待用户明确批准。
- 一次批准只覆盖当次说明的设备、Gate、命令范围和重跑策略；不自动授权后续 Gate。
- 与其他用户 GPU 进程共存时，必须说明额外显存预算和 OOM/性能风险。
- 未经明确批准，不运行 GPU 推理、评估或训练。
- 未经明确批准，不启动任何训练、backward、optimizer 或 scheduler。
- Gate 失败后不得盲目自动重跑；只有事先约定的安全 fallback 才能使用。

## 5. Gate 执行纪律

- 执行、修改或判定 Gate 前，必须完整阅读 `docs/HARNESS.md`。
- 每次只推进一个 Gate；完成后报告并停止，不自动进入下一 Gate。
- 只读分析和文档维护不算推进 Gate；运行验证、改变状态或形成通过/失败结论才算推进。
- 不得为了成功而静默跳过资产、样本、任务头或断言，也不得擅自改变设备、精度、batch size、数据范围或 checkpoint。
- 长时间命令必须包含 `timeout`、30 秒级心跳和明确退出码。
- Gate 通过必须同时具备：可重复命令、退出码 0、机器断言、日志、输入资产记录和明确的未验证范围。
- 所有本地输出只能写入工作区内事先说明的路径；运行证据优先放在 `reports/`。

## 6. 每次运行必须记录

- 用户授权范围和明确非目标；
- Git commit 与相关 dirty 文件；
- Python 绝对路径以及关键依赖版本；
- `PYTHONPATH`、`LD_LIBRARY_PATH`、CUDA 可见设备和 dtype；
- resolved config、checkpoint 类型和绝对资产路径；
- 固定 seed、样本 manifest、batch size 和采样策略；
- timeout、心跳、开始/结束时间和退出码；
- 峰值 CPU 内存、GPU 显存及适用的分阶段耗时；
- 断言结果、日志和机器可读产物路径；
- 是否发生 fallback，以及它对结论范围的影响。

## 7. 报告用语

每次报告必须区分：

- **已确认**：有命令、代码、日志或资产证据直接支持；
- **推断**：依据现有证据作出的判断，但尚未按目标协议验证；
- **未知**：证据不足；
- **风险**：可能影响正确性、资源安全或结论范围；
- **下一步**：只给出一个最小、可验证、不会影响其他用户的动作。

不得把单样本结果写成总体指标，不得把模型权重加载写成 forward 正确，也不得把部分状态恢复描述为 exact resume。

## 8. 文档与证据职责

- `README.md`：项目入口、当前总体状态和导航。
- `AGENTS.md`：长期规则和授权边界。
- `docs/HARNESS.md`：Gate 执行模板、数据/checkpoint 契约和通过标准。
- `REPRODUCTION_STATUS.md`：唯一当前状态账本。
- `reproduction/README.md`：复现入口、Gate 脚本、manifest 和证据导航。
- `reproduction/gates/`：本地 Gate 审计入口。
- `reproduction/manifests/`：固定的小规模输入清单。
- `experiments/`：后续改进与消融实验的登记和配置；不得与已验证复现基线混用。
- `reports/`：原始日志、JSON 和可视证据。
- `.runtime/`：可重建的临时数据视图和运行时链接，Git 永久忽略。
- `docs/future_improvements/`：尚未验证的研究假设；不得作为当前基线事实。

## 9. Gate 顺序

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

前一 Gate 通过不代表后一 Gate 已授权或已通过。
