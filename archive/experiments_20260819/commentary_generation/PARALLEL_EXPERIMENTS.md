# 固定 200 样本并行实验协议

状态：2026-08-14 已按本协议各运行一次，E1/E2/E3 均通过。本文档不授权重跑。

三组实验共同读取已经验证的公共缓存：

`reports/commentary_prefix_cache_200_20260814_run1/visual_prefixes.safetensors`

缓存固定了相同的 200 条 MatchTime 样本和每条 `[32,4096]` 的
`llama_projection` 输出。三组运行都必须重新核对缓存、manifest 和逐样本 SHA256，
在自己的进程中加载同一个 epoch-11 生成 checkpoint，并写入互不覆盖的目录。

## E1：解码参数折中

入口：`run_e1_decoder_sweep.py`

- 同进程历史基线：Beam 5 + Sampling，`top_p=0.9`、temperature 1.0；
- 四个 Nucleus 组合：temperature 0.70/0.85、`top_p=0.90/0.95`，以及一次
  repetition penalty 1.05；
- 每个条件对同一样本重置为同一 seed；
- 主要观察唯一输出率、Top-10 模板占比、BLEU-4、ROUGE-L 和 CIDEr。

输出目标：`reports/commentary_parallel_20260814/e1_decoder_sweep_run1/`

## E2：视觉条件敏感性

入口：`run_e2_visual_sensitivity.py`

- 正确视觉前缀作为同进程基线；
- 固定循环错配一个样本，保证 200/200 都不会拿到自己的前缀；
- 全零前缀作为第二个负对照；
- 三组均使用确定性 Beam，避免采样噪声；
- 比较完整文本变化率，以及限制词表上的首 token Jensen-Shannon divergence。

输出目标：`reports/commentary_parallel_20260814/e2_visual_sensitivity_run1/`

## E3：Attention Mask 与 PAD/EOS

入口：`run_e3_mask_pad.py`

- 历史行为：不显式传 mask 或 PAD/EOS；
- 只传长度 33 的全 1 attention mask；
- 显式传全 1 mask、PAD 128001、EOS `[128001,128009]`。

基础 `config.json` 的 EOS 是 128009，但实际 `generation_config.json` 的 EOS 为
`[128001,128009]`。历史 PAD 为空时，Transformers 4.51.3 会临时使用第一个 EOS
128001 作为 PAD。因此第三臂保留两个 EOS，只把历史隐式行为改成显式行为，不改变
停止规则。三臂使用确定性 Beam；预期结果是否相同仍须实测，不能预先写成结论。

输出目标：`reports/commentary_parallel_20260814/e3_mask_pad_run1/`

## 共同安全约束

- 每个入口只允许看到一张经当次批准的物理 GPU；
- 启动前必须重新执行 `nvidia-smi` 并取得当次授权；
- 固定本地 Python、offline、30 秒心跳、timeout 和明确退出码；
- `PYTHONPATH` 只含本地扩展目录和本仓库，不混入其他环境的 site-packages；
- 不创建 dataset、DataLoader、optimizer 或 scheduler；
- 不读取视频、不执行 Visual Encoder/Q-Former forward、不 backward、不训练；
- 每个进程只把 Llama 解码器移到 GPU，视觉模块留在 CPU；
- 任一组失败都不自动重跑，也不影响另外两组的结果判定。

## 首次运行结果

三组在 GPU200 并行运行，分配为 E1→GPU1、E2→GPU2、E3→GPU3。启动前各卡均为
0 MiB、0% 利用率；每组只加载一次模型，checkpoint missing/unexpected keys 均为空，
退出码均为 0。结束后 GPU1/2/3 均恢复 0 MiB、0%，没有残留计算进程。

### E1

| 条件 | 唯一率 | Top-10 占比 | BLEU-4 | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: | ---: |
| 历史 Beam + Sampling | 41.0% | 50.0% | 0.08190 | 0.26193 | 0.33514 |
| Nucleus T=0.70, P=0.90 | 77.0% | 21.5% | 0.07730 | 0.27227 | 0.48184 |
| Nucleus T=0.85, P=0.90 | 87.5% | 12.5% | 0.07138 | 0.25389 | 0.37677 |
| Nucleus T=0.85, P=0.95 | 90.5% | 12.0% | 0.06491 | 0.25174 | 0.39666 |
| Nucleus T=0.85, P=0.90, Rep=1.05 | 86.5% | 12.5% | 0.07035 | 0.25016 | 0.36519 |

`T=0.70, P=0.90` 是当前固定子集上最有希望的折中：多样性、ROUGE-L 和 CIDEr
同时高于历史法，但 BLEU-4 略低。它还不是完整测试集或人工语义质量结论。

### E2

- 正确前缀对循环错配前缀：187/200 条完整文本变化；首 token top-1 变化 44/200，
  平均 JS divergence 为 0.10260。
- 正确前缀对全零前缀：200/200 条完整文本变化，首 token top-1 变化 200/200，
  平均 JS divergence 为 0.63925。
- 正确前缀 CIDEr 为 0.33354；循环错配降至 0.08771；全零降至 0。
- 全零前缀 200 条全部产生同一个输出，均达到 128-token 上限且没有产生 EOT。

这些结果确认 decoder 会使用视觉前缀，而不是完全忽略视频条件。循环错配后的输出集合
分布仍与正确前缀相同，是因为同一组 200 个前缀被循环置换；但它们与各自参考答案的
对应关系被破坏，因此质量指标明显下降。

### E3

三臂 200/200 的完整 token 序列完全一致，文本、长度、多样性和质量指标也完全一致。
这确认在当前 batch size 1、全有效 33-token 输入的协议下，把隐式全 1 mask 和自动
PAD 128001 改为显式传入只消除了不确定性/警告，没有改变生成结果。因此历史警告不是
本次模板重复的直接原因；该结论不能外推到含 padding 的多样本 batch。

### 证据

- `reports/commentary_parallel_20260814/e1_decoder_sweep_run1/`
- `reports/commentary_parallel_20260814/e2_visual_sensitivity_run1/`
- `reports/commentary_parallel_20260814/e3_mask_pad_run1/`
- `reports/audits/commentary_parallel_e1_gpu1_run1_20260814.log`
- `reports/audits/commentary_parallel_e2_gpu2_run1_20260814.log`
- `reports/audits/commentary_parallel_e3_gpu3_run1_20260814.log`
- `reports/commentary_parallel_20260814/semantic_review.html`：历史法与 E1 最佳候选的人工审查页面。
- `reports/commentary_parallel_20260814/semantic_review_manifest.json`：页面来源和输出哈希。

E1/E2/E3 耗时分别为 1056.139、1546.401、843.168 秒；GPU 峰值 allocated 均约
16.2 GB，CPU 峰值 RSS 均约 37.4 GB。上述完成状态不构成任何后续 GPU 推理或重跑授权。

人工审查页面是标准库对已有 JSON 的 CPU-only 整理。页面中的 Token F1、事件词线索
和视觉 JS 只用于筛选，不是正式新指标；人工判断保存在浏览器 localStorage，只有主动
导出 JSON 后才能迁移或汇总。
