# SoccerMaster 解说生成实验

状态：CPU-only checkpoint 完整加载、MatchTime 数据可用性、固定单样本生成、
固定 200 样本三策略消融，以及基于公共前缀缓存的 E1/E2/E3 已通过；尚未执行完整
测试集评估或训练。

本目录用于复现 SoccerMaster 论文中的下游解说生成分支。它与主线的
`VideoCaption` 对比学习头不同：目标链路是 SoccerMaster 视觉编码器、
Q-Former、线性投影和 Llama-3-8B 的自回归 next-token prediction。

## 当前边界

- 只读使用服务器上的历史代码、基础模型、checkpoint 和 MatchTime 视频。
- 不修改 `/remote-home/haolinyang/sports/Soccer-Backbone`、共享环境、NAS 数据或
  其他用户目录。
- 不把两个历史 UniSoccer 代码树加入 `PYTHONPATH`。
- 已执行一次 CPU-only、offline、load-only 验证。
- 已在物理 GPU 7 上执行固定单样本和固定 200 样本解码对比；两次退出码均为 0。
- 200 样本结果包含 BLEU-1 至 BLEU-4、ROUGE-L、CIDEr 和多样性指标；它不是完整测试集评估。
- 尚未执行完整 3,256 条测试集 inference、backward 或 train。
- 后续 GPU 推理仍需重新检查 `nvidia-smi` 并获得单独授权。

## 文件

- `assets.json`：固定资产、已验证数据根、验证结果和当前阻塞。
- `sources.json`：历史生成代码的来源、SHA256 和本地模块名。
- `audit_entry.py`：纯标准库资产审计；不会运行模型。
- `load_checkpoint_cpu.py`：已经验证的 CPU-only load-only 入口。
- `infer_one.py`：已经通过静态副作用审查并成功执行一次的固定单样本 GPU 入口。
- `decode_ablation_200.py`：固定 200 样本、一次视觉 forward 配三种解码策略的 GPU 入口。
- `build_prefix_cache_200.py`：固定 200 样本的 cache-only GPU 入口；不生成文本。
- `cached_prefix_experiments.py`：E1/E2/E3 共用的缓存校验、decoder-only 加载和指标工具。
- `run_e1_decoder_sweep.py`：已成功运行一次的固定缓存解码参数扫描入口。
- `run_e2_visual_sensitivity.py`：已成功运行一次的正确/错配/全零视觉条件入口。
- `run_e3_mask_pad.py`：已成功运行一次的 mask 与 PAD/EOS 三臂入口。
- `event_separability_200.py`：固定200个缓存前缀的CPU-only reference事件可分性v1入口。
- `EVENT_DICTIONARY_V2_PREREGISTRATION.md`：基于v1歧义文本审阅、在结果前冻结的v2词典与评估协议。
- `event_separability_200_v2.py`：已按冻结协议正式运行一次的CPU-only v2入口；结果位于`reports/commentary_event_separability_200_20260816_v2/`。
- `EVENT_DICTIONARY_V3_PREREGISTRATION.md`：基于v2 `other`/`other_rare`文本审阅、在结果前冻结的最小漏检修正规则。
- `event_separability_200_v3.py`：已按冻结协议正式运行一次的CPU-only v3入口；结果位于`reports/commentary_event_separability_200_20260816_v3/`。
- `EVENT_SEPARABILITY_SYNTHESIS_20260816.md`：v1/v2/v3标签敏感性、逐类混淆和pooled/query-slot配对错误的最终综合；词典迭代在v3停止。
- `VIDEO_EVENT_ANNOTATION_PROTOCOL_200.md`：独立双人视频事件标注本体、盲标、锁定与仲裁协议；尚未访问视频或执行人工标注。
- `build_video_event_annotation_packet_200.py`：只从既有本地JSON复制固定200条视频路径字符串，生成两份不同顺序的盲标空表，不打开视频。
- `PARALLEL_EXPERIMENTS.md`：三组并行实验的固定协议和安全边界。
- `render_semantic_review.py`：标准库人工审查页面生成器；只读取既有 E1/E2/E3 JSON。
- `render_decode_ablation_report.py`：标准库 HTML 生成器；只读取已有 JSON，不运行模型。
- `runtime/`：经过隔离的最小本地运行模块，不把远端源码树加入导入路径。

历史实现来自以下两棵永久只读代码树：

- 生成模型和训练/推理入口：
  `/remote-home/haolinyang/sports/dirty_code/UniSoccer`
- dataset、optimizer、评估工具和限制词表：
  `/remote-home/haolinyang/sports/UniSoccer`

`sources.json` 固定了已审查的精确来源。训练 optimizer 和完整评估工具仍只登记
来源、没有复制；不得在运行时混用两个远端源码根。

## 固定模型资产

- Llama：`/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct`
- Q-Former BERT 配置：`/remote-home/share/huggingface/bert-base-uncased`
- SigLIP2：`/remote-home/haolinyang/sports/Soccer-Backbone/pretrained_models/google/siglip2-large-patch16-512`
- 视觉 backbone：历史生成实验实际使用的不带 `_high_resolution` 的
  `extra_7000/epoch_19/backbone.pt`
- 生成 checkpoint：历史 MatchTime 实验的 `model_save_11.pth`
- 生成限制词表：`match_time.pkl`

不能把视觉 backbone 静默替换成主线 Gate 使用的 `_high_resolution` 版本；两个
文件大小相同，但已确认归档字节不同。

## Tokenizer 契约

CPU-only 离线审计已确认：

- 初始 `len(tokenizer) == 128256`；
- 新增 `[PLAYER]`、`[TEAM]`、`[COACH]`、`[REFEREE]`、`([TEAM])` 后为 128261；
- 新 token ID 依次为 128256–128260；
- BOS 为 128000，运行时 EOS 为 128009；
- tokenizer 默认没有 PAD；历史 dataset 显式使用 128001
  (`<|end_of_text|>`) 作为 padding。

任何本地实现都必须在加载生成 checkpoint 前断言上述契约，并在添加五个 token
后把 Llama embedding resize 到 128261。不得隐式选择新的 PAD token。

## CPU-only checkpoint 加载结果

2026-08-13 的单次 load-only 验证通过：

- checkpoint epoch 为 11；checkpoint 与模型均有 953 个状态键；
- 模型参数量为 8,418,890,760；
- `llama_model`、`visual_encoder`、`video_Qformer`、`video_query_tokens`、
  `ln_vision`、`llama_proj` 和 `video_frame_position_embedding` 全部加载；
- 全局和七个组件的 missing/unexpected keys 均为空；
- 总耗时 534.38 秒，峰值 RSS 37,208,656 KiB；
- `CUDA_VISIBLE_DEVICES=""`，Torch 可见 GPU 数为 0；
- 没有创建 dataset、DataLoader、optimizer 或 scheduler，也没有执行 forward、
  generate、eval、inference、backward 或 train。

原始日志：`reports/audits/commentary_checkpoint_load.log`；SHA256 为
`85dc00c0a9b2bde2fd6b43d29487df8d380e43a47e103b9a029e3ee7d7fe6525`。

## MatchTime 数据结果

旧软链接目标 `/remote-home/jiayuanrao/dataset/matchtime/video_clips` 已失效，但数据
已经迁移，并非缺失。2026-08-13 的只读逐文件元数据检查确认：

- 训练根：
  `/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/SoccerNetv2/MatchTime/train`
  - 24,027 条标注、23,425 个唯一视频；23,425/23,425 存在；
  - 总大小 144,499,033,648 bytes；缺失、无权限和非普通文件均为 0。
- 测试根：
  `/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/SoccerNetv2/MatchTime/SN-Caption-test-align`
  - 3,256 条标注、3,251 个唯一视频；3,251/3,251 存在；
  - 总大小 19,389,988,464 bytes；缺失、无权限和非普通文件均为 0。

一条测试视频已由 OpenCV 成功打开并解码首帧。一条训练视频已按历史 Decord
逻辑以 `middle` 模式采样 30 帧，输出 `(30, 3, 224, 398)`、`torch.uint8`、CPU，
退出码为 0。上述结果只证明固定资产覆盖和代表性解码，不等于所有视频都已逐个
解码，也不证明模型生成质量。

## 固定单样本解说生成结果

2026-08-13 在物理 GPU 7 上执行了唯一一次固定 MatchTime 测试样本推理：

- 测试集 index 0，batch size 1，30 帧 `middle` 采样，seed 42；
- checkpoint epoch 11，953 个状态键，missing/unexpected keys 均为空；
- forward 和 generate 均完成，退出码 0，未执行训练；
- 参考解说：
  `[PLAYER] ([TEAM]) gets on the end of a pass on the edge of the box but his shot is blocked.`
- 生成解说：
  `[PLAYER] ([TEAM]) sends a pass into the box, but the opposition's defence is alert to the danger and intercepts the ball.`
- 总耗时 160.700 秒；forward + generate 为 2.088 秒；
- CPU 峰值 RSS 37,644,280 KiB；GPU 峰值 allocated 19,506,608,128 bytes，
  reserved 20,004,732,928 bytes；
- 运行结束后 GPU 7 为 0 MiB、0% 利用率，无计算或残留推理进程。

原始日志：`reports/audits/commentary_infer_one.log`，SHA256 为
`b1021589f0d5806159c56ab2860a2de02c53402142beca45b3851186d29fbe0e`。
机器结果：`reports/audits/commentary_infer_one_result.json`，SHA256 为
`ee115aa29e27aca2b2bd4599ad2a3fafda8cbdda85569a65724f634813125b99`。

该结果只证明一个固定真实样本的端到端生成链路能够运行。它不是总体指标，也不
证明生成文本与参考答案等价或论文指标已经复现。

## 固定 200 样本解码消融

2026-08-14 在物理 GPU 7 上完成一次固定 200 样本对比。模型和 checkpoint 只加载
一次；每个样本只执行一次视觉编码和 Q-Former forward，再复用同一个
`[1,32,4096]` 表示执行三种解码：

| 策略 | 唯一输出率 | Top-10 模板占比 | BLEU-4 | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: |
| 历史 Beam + Sampling | 41.0% | 50.0% | 0.08190 | 0.26193 | 0.33514 |
| 确定性 Beam | 39.5% | 50.5% | 0.08079 | 0.26446 | 0.33354 |
| Nucleus Sampling | 92.5% | 12.0% | 0.05973 | 0.23999 | 0.32214 |

运行完成 200 条、600 个非空输出，pipeline 退出码为 0；没有训练、backward、
optimizer、scheduler、DataLoader、fallback 或自动重跑。Nucleus 显著减少完整句子
重复，但当前词面指标下降，因此只能确认“多样性改善”，不能直接确认“语义质量
改善”。

原始证据：

- `reports/audits/commentary_decode_ablation_200_run1_20260814.log`
- `reports/commentary_decode_ablation_200_20260814/result.json`
- `reports/commentary_decode_ablation_200_20260814/predictions.jsonl`
- [可排序逐样本 HTML](../../reports/commentary_decode_ablation_200_20260814/index.html)
- [HTML 生成清单](../../reports/commentary_decode_ablation_200_20260814/report_manifest.json)

HTML 是标准库对既有结果的离线整理，没有重新运行模型。它可以按策略分歧、模板
重复和 Nucleus 词面变化排序，并把参考答案与三种输出并排显示。页面中的 Token F1
与事件词线索只用于人工检查，不是正式指标。

## 固定 200 样本公共视觉前缀缓存

2026-08-14 经当次授权在物理 GPU 7 执行一次 cache-only 运行，为后续并行解码和
视觉敏感性实验保存公共视觉条件：

```text
30帧视频
→ Visual Encoder
→ 时间融合
→ Q-Former
→ llama_projection
→ [200,32,4096] float32
```

- 固定 indices 与通过的 200 样本消融完全一致，范围为 dataset index 58 至 3251；
- 200/200 视频完成解码和视觉/Q-Former forward，checkpoint missing/unexpected keys 为空；
- 缓存原始 tensor 为 104,857,600 bytes，safetensors 文件为 104,859,528 bytes；
- 缓存 SHA256 为 `8b1723926eacfe381ceae2ec5433767574f56028d894a1b28d7c7222c69b6c97`；
- 写入后重新加载逐元素完全相同；独立 CPU 复核的 200 个逐样本 SHA256 全部匹配；
- 运行耗时 485.432 秒，峰值 RSS 37,449,740 KiB；GPU 峰值 allocated/reserved 为
  19,506,609,152 / 19,958,595,584 bytes；
- 没有文本生成、DataLoader、optimizer、scheduler、backward、训练或 fallback；
- 结束后 GPU 7 恢复 0 MiB、0% 利用率，无残留计算进程。

缓存属于大型派生资产，由 `*.safetensors` Git 规则忽略，不应提交。证据为：

- `reports/commentary_prefix_cache_200_20260814_run1/visual_prefixes.safetensors`
- [manifest](../../reports/commentary_prefix_cache_200_20260814_run1/manifest.json)
- [机器结果](../../reports/commentary_prefix_cache_200_20260814_run1/result.json)
- `reports/audits/commentary_prefix_cache_200_gpu7_20260814.log`

该缓存只证明固定 200 条的投影视觉前缀可重复读取；它不证明解码质量，也不改变
此前三策略结果。后续实验必须核对 cache/manifest SHA256，不得按位置猜测样本身份。

## 静态审计

```bash
cd /home/tianlin/SoccerMaster
/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python \
  experiments/commentary_generation/audit_entry.py
```

默认模式检查资产、依赖、来源和本地证据文件。`--strict` 当前也应返回 0，并报告
`ready_for_end_to_end_inference=True`。审计输出中的 `model_imported=False`、
`checkpoint_loaded=False` 和 `gpu_used=False` 描述的是静态审计进程自身；已经执行
的单样本结果以 `assets.json` 的 `verification` 和原始日志为准。审计不会导入机器
学习库或读取模型权重内容。

## 固定缓存 E1/E2/E3 结果

2026-08-14 在 GPU1/2/3 并行完成三组各一次固定 200 样本实验，退出码均为 0：

- E1 找到当前最有希望的解码折中：Nucleus `temperature=0.70, top_p=0.90` 的唯一率
  为 77.0%、Top-10 模板占比 21.5%、CIDEr 0.48184；历史法分别为 41.0%、50.0%、
  0.33514。其 ROUGE-L 也由 0.26193 升到 0.27227，但 BLEU-4 从 0.08190 略降到
  0.07730。
- E2 中，正确前缀与循环错配前缀导致 187/200 条文本变化，CIDEr 从 0.33354 降到
  0.08771；全零前缀导致 200/200 条变化并坍缩为一个不产生 EOT 的 128-token 输出。
  这确认 decoder 确实使用视觉条件。
- E3 的历史隐式行为、显式全 1 mask、显式 mask/PAD/EOS 三臂在 200/200 条上 token
  完全一致，说明这些警告不是当前 batch-size-1 模板重复的直接原因。
- 三组同进程基线均与此前消融对应输出 200/200 完全一致。

完整协议、限制和证据见 `PARALLEL_EXPERIMENTS.md`。这些仍是固定 200 条子集结果，不是
完整 3,256 条测试集或人工语义质量结论。

人工审查入口：[semantic_review.html](../../reports/commentary_parallel_20260814/semantic_review.html)。
页面把参考答案、历史输出和 `T=0.70/P=0.90` 并排展示，支持搜索、排序、候选优劣、
事件/动作/结果/角色关系勾选、浏览器本地保存和 JSON 导出。它没有重新运行模型，也
不会把人工标注写回服务器；需要主动点击“导出标注 JSON”保存可迁移结果。

## 当前状态与下一步

模型、checkpoint、tokenizer、训练视频、测试视频、固定单样本端到端链路和固定
200 样本三策略消融和公共视觉前缀缓存已经在上述有限范围内验证。当前关键风险仍是历史
`inputs_embeds` 生成路径没有显式 `attention_mask`。实际 generation config 的 EOS
为 `[128001,128009]`，PAD 未设置时会临时采用第一个 EOS 128001；不能静默修改后
直接与历史结果比较。

E1 解码扫描、E2 视觉敏感性和 E3 mask/PAD 已按固定协议各运行一次并通过。后续最有
价值的候选工作是：使用 `semantic_review.html` 人工审查 `T=0.70, P=0.90` 是否真的改善事件、
动作、结果和角色关系，而不只是提高词面指标；随后再决定是否扩展到完整测试集。
这些候选项不构成新的 GPU 推理、完整评估或训练授权。

## 因果瓶颈实验（2026-08-17）

解说研究的下一阶段不再把单参考文本相似度或单层 probe 直接当作事实正确性。完整协议见：

- [因果瓶颈实验计划](CAUSAL_BOTTLENECK_EXPERIMENT_PLAN.md)
- [比赛分组 v3 修正协议](MATCH_GROUPED_V3_DIAGNOSTIC_PROTOCOL_20260817.md)
- [存储恢复检查表](STORAGE_RECOVERY_CHECKLIST_20260817.md)
- `LAYER_CACHE_CONTRACT_20260817.json`
- `ORACLE_INTERVENTION_CONTRACT_20260817.json`
- `schemas/`

现有 fixed-200 覆盖完整测试集 49 场比赛中的 48 场，因此 3,256 条全部降为开发数据；最终结论需要存储恢复后从新比赛构建且排除训练重叠的 Locked Match Holdout。

按比赛分组修正后的 reference-relative v3 probe 已完成一次 CPU-only 运行：mean-pooled macro-F1 为 0.6057，query-slot-preserving 为 0.5575，对应打乱标签基线为 0.1257/0.0919。该结果继续支持 projected prefix 与 reference 事件有关联，但不证明视频事实正确或任何模块具有因果责任。
