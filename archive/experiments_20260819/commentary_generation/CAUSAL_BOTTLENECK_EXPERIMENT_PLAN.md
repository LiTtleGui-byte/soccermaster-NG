# 解说链路因果瓶颈实验计划

状态：协议已确认；NAS/202 资产桥接、纯 CPU 预检与 `nvidia-smi` 资源报告
完成。8 张 H800 均被同一任务以 98–99% 利用率占用，因此未启动模型；后续
仍需空闲资源和单样本 GPU 命令的另行授权。本计划不授权 GPU、推理或训练。

## 1. 目标

在固定输入视频片段上，定位以下链路中对视频事实错误贡献最大的模块，并只改进排名第一的模块：

```text
表征对齐预训练
→ 视觉编码与时间建模
→ Q-Former
→ projector
→ Llama
→ 解码
→ 视频事实评价
```

本轮暂不研究原始比赛如何切成短视频。Temporal Pairing 保留在系统边界中，但视为固定输入，不参加瓶颈排名，也不被宣称为已经正确。

最终输出是带 95% 置信区间的 Causal Bottleneck Ranking。只有某个模块经过公平的 Interface-Matched Intervention 后，端到端事实指标出现最大且可重复的改善，才称它为主要瓶颈；否则报告并列瓶颈或模块交互。

## 2. 已知事实

- 历史 fixed-200 生成的 BLEU-4/ROUGE-L/CIDEr 为 0.08190/0.26193/0.33514，唯一输出率 41%，Top-10 模板占 50%。
- E1 `temperature=0.70, top_p=0.90` 把唯一率提高到 77%、CIDEr 提高到 0.48184，但相对 reference 的人工判断从 54/200 `different` 变为 69/200；调解码没有改善事件一致性。
- E2 中正确 prefix 改为循环错配后 187/200 文本变化、CIDEr 从 0.33354 降到 0.08771；decoder 使用视觉条件，但这不证明条件包含正确事实。
- E3 三种 mask/PAD/EOS 写法在 200/200 条上 token 完全相同；它不是当前模板重复的直接原因。
- 当前唯一的 200 条公共缓存位于 Q-Former 和 projector 之后，shape 为 `[200,32,4096]`，不能拆分视觉编码、时间层、Q-Former 和 projector。
- v3 reference-relative probe 的普通 clip-level 五折 macro-F1 为 0.6169，但同场比赛大量跨 fold；这不是 match-isolated 泛化结果。
- 3,256 条 reference-support 审查为 41.52% accurate、57.40% partial、1.01% wrong、0.06% uncertain。它评价 reference 是否被视频支持，不是独立视频事实标签。
- 3,256 条只覆盖 49 场比赛；现有 fixed-200 已覆盖其中 48 场。剩余片段不能组成真正按比赛隔离的最终测试集。

## 3. 数据角色

### 3.1 开发数据

- fixed-200：用于协议开发、离线诊断、快速筛查和回归测试。
- 完整 3,256：存储恢复后用于开发期 baseline、模型评审和模块定位。
- 所有开发期交叉验证必须按比赛分组；同一场比赛不得跨 train/test fold。
- 这两批数据都已经被观察或用于设计，不能承担最终确认。

### 3.2 最终确认数据

Locked Match Holdout 必须同时满足：

1. 比赛不属于当前 49 场；
2. 比赛和 clip 未用于任何方案选择、词典修改、阈值选择或 prompt 调整；
3. 恢复 NAS/GPFS 后，核对其未进入 generation checkpoint 与相关视觉 backbone 的训练数据；
4. 按约 8–12 个大事件族覆盖，每个主要事件族目标至少 20–30 条可观察样本；
5. 只有最终胜者可以查看一次 holdout 结果。

若找不到满足训练隔离的新比赛，则最终结果只能表述为开发诊断，不得宣称独立泛化提升。

## 4. 视频事实与模型评审

评审模型分三种互不混用的任务：

1. **Video Fact Record**：只看视频，输出主要事件、动作、结果、人物/角色关系和各槽位是否可观察。
2. **Candidate-Grounding Review**：看视频与一条生成解说，不看 reference、条件名或模型身份，判断每个 claim 是否有视频支持。
3. **Reference-Support Review**：看视频与 reference，只审查数据配对和 reference 细节。

事件类别使用约 8–12 个稳定大类；动作、结果和角色保留为独立细粒度槽位。不可观察槽位不进入正确率分母；省略不可观察细节不扣分，主动声称视频不支持的内容计为 Unsupported Claim。

随机 10% 样本更换展示顺序重复评审，并混入明显正确/错误控制项；重复不一致或控制项失败交给更强评审模型复核。评审稳定性单独报告，不用 reference 替代视频事实。

## 5. 主要指标

按以下优先级判定：

1. Core Event Error rate，越低越好；
2. 可观察 action/result/role 槽位的 macro-F1；
3. Unsupported Claim rate；
4. 流畅性非退化；
5. BLEU、ROUGE-L、CIDEr、多样性只作解释性指标。

所有主要比较是同一 clip 上的配对比较。置信区间采用以比赛为聚类单位的 paired bootstrap；离散错误同时报告配对精确检验，多个模块/指标用 Holm 校正。

成功至少要求：

- Core Event Error 绝对下降至少 5 个百分点，或相对下降至少 20%；
- 改善的 95% CI 不跨 0；
- action/result/role 总体事实分数不下降；
- 流畅性不出现明显退化；
- 排名前两位与最终方案各完成 3 个训练 seed；随机解码另分离报告 decode seed。

## 6. 分层观测点

同一 clip、相同帧索引和预处理下记录：

| 层 | 最小输出 | 回答的问题 |
| --- | --- | --- |
| sampled input | 帧索引、预处理身份 | 输入是否一致 |
| visual spatial/frame | 每帧局部或全局视觉表示 | 空间压缩前是否有事实信息 |
| temporal output | 时间建模后的逐帧表示 | 时间层是否丢失事件/结果 |
| Q-Former input | LN、位置融合后的 30-token 表示 | connector 的实际输入是否可分 |
| Q-Former output | `[32,768]` | query 压缩是否丢信息 |
| projector output | `[32,4096]` | 线性映射是否丢信息 |
| decoder logits | 首 token 与事实相关 token 的 logits/NLL | decoder 是否能读出已有事实 |
| generated text | token 与文本 | 解码是否引入事实错误 |

局部 patch 特征可能很大，只在按比赛分组、事件平衡的诊断子集保存；较小的全局/Q-Former/projector 输出可扩展到完整开发集。

## 7. 证据分两步

### 7.1 信息筛查

每层使用相同 Video Fact Record、相同 match-grouped folds 和 fold-local 预处理训练轻量 probe。Probe 只回答“该层信息能否被读取”，不进入正式因果排名。

解释规则：

- 早期局部特征强、逐帧全局特征弱：空间压缩候选；
- 时间层前强、后弱：时间建模候选；
- Q-Former 输入强、输出弱：Q-Former 候选；
- Q-Former 输出强、projector 输出弱：projector 候选；
- projector 输出强、生成事实差：decoder、训练目标或解码候选。

### 7.2 因果救援

只对筛出的模块进行 Interface-Matched Intervention：保持边界 shape、下游 checkpoint、数据、监督量、训练预算、seed、词表和解码不变，只替换一个模块。

结构化事实文字直接提示 Llama 只作为 decoder 理论上限，不能与接口匹配的模块救援幅度直接排名。正式排名依据接口匹配干预的配对事实改善及 95% CI。

若两个模块无法区分，先做二者组合干预；组合额外改善则报告交互，否则优先选择成本更低、修改更小的模块进行改进。

## 8. 胜者对应的改进候选

只实现排名第一的分支：

- 视觉空间压缩：保留受控数量的局部 token，与全局 token 并存；
- 时间建模：调整时间聚合或帧级证据选择，但不改变原 clip 边界；
- Q-Former：调整 query/层数或加入事件事实辅助目标；
- projector：比较等预算 linear 与小 MLP，或加入事实保持目标；
- Llama/训练目标：事实规划后生成、事件/结果辅助 loss，必要时受控 LoRA；
- 解码：只在 prefix 已被证明正确时比较词表限制和解码参数。

候选先单 seed 快速筛查；前两名各 3 seeds；只将最终胜者送入 Locked Match Holdout。

## 9. 存储恢复前准备：已完成

1. 记录术语、ADR 和本计划。
2. 用现有 101 MB prefix cache 重跑 match-grouped v3 probe，量化原结论受比赛泄漏影响的程度。
3. 建立 3,256 条比赛分组清单，明确 49 场及 fixed-200 覆盖 48 场的事实；不伪造内部 holdout。
4. 建立评审输入/输出 schema、主要指标和 match-clustered CI 工具，并用已有 JSON 做结构测试。
5. 建立逐层缓存 manifest 与 oracle 运行配置的静态契约；不得导入模型或访问远端资产。
6. 准备存储恢复清单、输出路径、估算空间和第一条最小运行协议。

已完成 match-grouped probe、数据角色清单、评审/计分 schema、缓存/oracle
静态契约及存储恢复协议。资源包现位于 NAS，身份见
`NAS_COMMENTARY_BUNDLE_MANIFEST_20260817.json`。

## 10. 存储恢复后：顺序执行

每个阶段失败后停止，不自动扩大范围：

1. **资产恢复预检（主体完成）**：视频、generation checkpoint、visual backbone、tokenizer 和训练清单可读且 CPU 预检通过；新比赛训练重叠检查仍须在构建 Locked Match Holdout 时完成。
2. **最小真实样本**：单 clip 逐层输出，确认 shape、身份和副作用。
3. **开发 baseline**：完整 3,256 基础生成与三类模型评审；固定解码与 seed。
4. **分层缓存**：先事件平衡子集，后扩展小体积层；不默认保存全量局部 patch。
5. **match-grouped probes**：筛出信息下降位置。
6. **接口匹配救援**：单模块干预并生成 Causal Bottleneck Ranking。
7. **交互检查**：仅在前两名 CI 重叠时运行。
8. **改进胜者**：单 seed 筛查，前两候选 3 seeds。
9. **最终确认**：锁定协议后只运行一次 Locked Match Holdout。

任何 GPU 阶段仍须按 `AGENTS.md` 重新执行 `nvidia-smi`、报告资源并取得当次授权。本计划不提供 GPU、推理或训练授权。

## 11. 停止条件

- 评审控制项或重复一致性不足：先修评审，不评模型。
- 资产或训练重叠身份不清楚：不建立 Locked Match Holdout。
- probe 只在 clip-level、不能按比赛分组：不得报告泛化分数。
- 只有 probe 下降、接口匹配干预不能改善端到端事实：不得称为因果瓶颈。
- 所有模块 CI 重叠：报告无法唯一排序，执行预先限定的前二交互检查。
- 最终方案未达到实际提升门槛：保留负结果，不扩大训练寻找显著性。

## 12. 产物

- 比赛分组与数据角色 manifest；
- 模型评审 schema、盲评包和稳定性报告；
- 各层 cache manifest 与张量；
- match-grouped probe 结果；
- 每个接口匹配干预的配对预测和指标；
- Causal Bottleneck Ranking；
- 胜者三 seed 结果；
- 一次 Locked Match Holdout 最终报告。
