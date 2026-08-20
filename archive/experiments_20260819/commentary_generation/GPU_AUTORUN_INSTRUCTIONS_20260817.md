# 解说诊断 GPU 自动排队说明

状态：已经实现并通过纯 CPU 静态检查；用户已批准一次性窄范围例外及 smoke
共存修订。修订配置哈希
`0439fe41b81cb5b91f95740cfcaf19641ae48a104772490466aaf693e7c75199`
已经 arm，原 tmux 名称 `commentary-gpu-autowatch` 已重新启动。当前运行状态以
`.runtime/` 中的 `state.json` 为准；不得修改已 arm 的配置文件。

## 给值守 Agent 的 instructions

你在 `/home/tianlin/SoccerMaster` 的 GPU200 上值守一个有边界的研究任务。
完整读取 `AGENTS.md`、本文件和 `GPU_AUTORUN_CONFIG_20260817.json`。不得扩大
自动范围，不得修改阈值，不得把失败任务自动重跑。

这个任务只允许完成三个阶段：

1. 每 30 分钟运行只读 `nvidia-smi`，记录全部 8 张卡和计算进程；
2. 当至少一张卡连续两次满足空闲条件时，在其中一张卡上运行一条视频的
   inference-only smoke test；
3. smoke test 通过后继续等待。当 8 张卡连续两次全部满足空闲条件时，启动
   8 个互相独立的分片，完成 3,256 条开发数据的逐层缓存和基线生成；CPU
   合并完成后写出 `REVIEW_REQUIRED.md` 并退出。

到达人工审核闸门后，不得继续运行模型评审、probe、oracle、训练或 holdout。

任一阶段退出码非 0、超时、GPU 在 worker 启动前不再空闲、输出路径已经存在、
资产身份不符或任一断言失败时，立即停止全部属于本次任务的 worker，保留日志，
不得自动修复或重跑。不得终止、暂停或改变其他用户的进程。

## “空闲 GPU”的定义

八卡正式阶段中的一张卡必须同时满足：

- 空闲显存至少 60,000 MiB；
- GPU utilization 不高于 5%；
- 没有任何 compute process；
- 相隔 60 秒的两次 `nvidia-smi` 都满足以上条件。

即使显存足够，只要利用率高或存在其他进程，就不使用。worker 真正启动前还会
再次检查自己对应的物理 GPU，避免确认和启动之间被别人占用。

一次性 smoke 修订只允许物理 GPU1 在空闲显存至少 30,000 MiB、利用率不超过
5% 时与 PID 3375814 共存。其他 PID 仍会阻止运行。该例外不传递到八卡阶段。

## 自动运行的内容

Smoke test 固定为 test dataset index 0、30 帧 middle sampling、batch size 1。
它执行真实视觉 forward、Q-Former、projector 和一次解说生成，检查所有捕获张量
的 shape 和有限值。它不创建 optimizer/scheduler，不执行 backward 或 training。

八卡阶段按 `dataset_index % 8` 分片，每张卡恰好处理 407 条。每条样本保存：

- 视频路径、比赛 ID、实际帧索引和预处理身份；
- `visual_frame_global [30,1024]`；
- `layer_normalized [30,1024]`；
- `temporal_output/qformer_input [30,1024]`；
- `qformer_output [32,768]`；
- `projector_output [32,4096]`；
- reference、生成文本和生成 token IDs。

每条视频使用 `20260817 + dataset_index` 作为生成 seed，因此结果不依赖它被分给
哪张卡。八个分片合并时必须证明 0–3,255 每个 index 恰好出现一次。

首 token logits 和 fact-token NLL 暂不跑；它们要根据这批结果选择诊断子集后再
申请授权。局部 patch 特征也不在本轮保存，避免无必要地产生数百 GB 数据。

## 为什么现在不能直接启动

现行 `AGENTS.md` 要求每次最新 GPU 报告之后再由人批准命令。无人值守的条件式
启动需要用户对本配置给出一次范围非常窄的例外授权。授权只覆盖当前配置哈希、
GPU200、run1 输出目录和上述三个阶段。

授权前配置中的 `execution_authorized` 是 `false`，因此程序即使被误启动也会
拒绝运行 GPU。用户确认后，主 Agent 已把该字段改为 `true`，并在 `AGENTS.md`
记录以下窄范围例外：

> 用户一次性批准 `GPU_AUTORUN_CONFIG_20260817.json` 对应哈希的条件式运行：
> 每 30 分钟查询 GPU；满足配置中的双重空闲检查后，允许一次单样本 smoke；
> smoke 通过后，满足全部八卡空闲检查时允许一次八分片 inference/cache；CPU
> 合并后必须停在人工审核闸门。失败不重跑，禁止训练、backward、optimizer、
> oracle 和 holdout。该例外在完成、失败、配置改变或人工中止时立即失效。

## 授权前检查

下面的命令只检查配置和输出路径，不查询或使用 GPU：

```bash
cd /home/tianlin/SoccerMaster
PYTHONDONTWRITEBYTECODE=1 /home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python experiments/commentary_generation/commentary_gpu_autorun.py --config experiments/commentary_generation/GPU_AUTORUN_CONFIG_20260817.json --check
```

配置有任何变化都必须重新检查并重新授权，因为配置 SHA-256 会改变。

## 授权后只执行一次的启动步骤

先创建与当前配置哈希绑定的 arm 文件：

```bash
cd /home/tianlin/SoccerMaster
PYTHONDONTWRITEBYTECODE=1 /home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python experiments/commentary_generation/commentary_gpu_autorun.py --config experiments/commentary_generation/GPU_AUTORUN_CONFIG_20260817.json --arm --authorization-note "user approved bounded commentary autorun in Codex"
```

然后创建 tmux session：

```bash
tmux new-session -d -s commentary-gpu-autowatch "cd /home/tianlin/SoccerMaster && exec /home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python experiments/commentary_generation/commentary_gpu_autorun.py --config experiments/commentary_generation/GPU_AUTORUN_CONFIG_20260817.json --run"
```

查看状态：

```bash
tmux attach -t commentary-gpu-autowatch
```

或者不进入 tmux，直接查看精简事件日志：

```bash
tail -f /home/tianlin/SoccerMaster/.runtime/commentary_gpu_autorun_20260817/supervisor.log
```

手动中止时，在 tmux 中按 `Ctrl-C`。中止后不要直接重新启动；先检查 state、日志、
输出目录和 GPU 残留，再决定是否为新的 run 编号重新授权。

## 输出位置

- 状态与监控日志：`.runtime/commentary_gpu_autorun_20260817/`；
- smoke 和八个分片：`reports/commentary_autorun_3256_20260817_run1/`；
- 最终合并清单：同目录的 `manifest.json`；
- 必停文件：同目录的 `REVIEW_REQUIRED.md`。

到达 `review_required`、任何 `failed` 状态或 tmux 退出后，值守 Agent 应向用户报告
最后一次 GPU 快照、实际执行范围、退出码和输出路径，然后停止。
