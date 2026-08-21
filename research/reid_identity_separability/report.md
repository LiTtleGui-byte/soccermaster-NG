# PRTReID 人物身份可分性评估

## 结论

同队不同球员与 Same-ID 大量重叠；当前 PRTReID feature 很可能是 ID switch 的重要来源之一。

这里的判断只针对固定 checkpoint 的 embedding 可分性，不代表 StrongSORT tracking 的最终性能。

## 数据与模型

- 数据：`/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/valid` 的 SoccerNetGS valid，共 59 个视频。
- 真值：`Labels-GameState.json` 的人工 `track_id`、`bbox_image`、`attributes.team`。
- 过滤：仅保留有 team 的 `player` 和 `goalkeeper`；不把球、裁判或模型 track ID 当作 identity GT。
- 有效 crop：645,670；视频内身份轨迹：1,237。
- checkpoint：`/home/tianlin/SoccerMaster/assets/checkpoints/official/soccerfactory/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar`。
- embedding：官方 `test_embeddings=["globl"]`，每个 crop 为 256 维。
- GPU：物理 GPU 0, 1, 2, 3, 4, 5, 6, 7；所有 forward 均为 `model.eval()` 和 `torch.no_grad()`。

## Pair 构造

Pair 只在同一视频内构造：

- A：同一个人工 track ID、不同帧；
- B：不同人工 track ID、同一人工 team；
- C：不同人工 track ID、不同人工 team。

每个视频、每类最多采样 10,000 个不重复 pair，seed=20260820。以下“最难案例”是**已采样 pair 中**的极值，不声称是全部二次方 pair 空间的精确极值。

## StrongSORT 实际距离

当前正式配置使用 `part_based` matching。因为正式 PRTReID 只输出一个 global part，代码会先对 256 维 embedding 做 L2 normalize，计算 normalized Euclidean distance，再除以 2。匹配门槛为 `0.5`；大于该值的 appearance match 会被 gate。报告同时给出普通 cosine distance 和未归一化 embedding 的 Euclidean distance。

## StrongSORT 距离分布

| Pair | 数量 | mean | median | p10 | p90 | 距离 ≤ 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| `same_id` | 580,675 | 0.2345 | 0.2162 | 0.1412 | 0.3609 | 98.62% |
| `different_id_same_team` | 584,900 | 0.3318 | 0.3084 | 0.2295 | 0.4880 | 91.77% |
| `different_id_different_team` | 585,600 | 0.3921 | 0.3633 | 0.2812 | 0.5298 | 81.21% |

在当前 `max_dist=0.5` 下，Same-ID 的 false negative rate 为 1.38%；同队不同人的 appearance false positive rate 为 91.77%，所有不同人的 appearance false positive rate 为 86.48%。这里的 “false positive” 只表示通过 ReID appearance gate，StrongSORT 后面仍有运动和时空门控，不能把它直接等同于最终 ID switch 率。

## 区分指标

| 比较 | ROC-AUC | EER | EER 距离阈值 | 最佳 Youden FPR | 最佳 Youden FNR |
|---|---:|---:|---:|---:|---:|
| Same-ID vs all Different-ID | 0.8453 | 0.2266 | 0.2826 | 0.1428 | 0.2879 |
| Same-ID vs Same-Team Different-ID | 0.7976 | 0.2682 | 0.2685 | 0.1593 | 0.3537 |

ROC-AUC 使用 `-distance` 作为相似度分数，正类为 Same-ID。FPR 表示不同人物被误判为同一人物，FNR 表示同一人物被误判为不同人物。

## 可视化与失败案例

- [Same-ID vs all Different-ID](figures/same_vs_all_different.png)
- [Same-ID vs Same-Team Different-ID](figures/same_vs_same_team_different.png)
- [最容易混淆的 20 个 Different-ID pair](bad_cases/most_confusing_different_id.csv)
- [距离最大的 20 个 Same-ID pair](bad_cases/largest_same_id_distance.csv)
- 完整采样 pair：本地生成产物 `pairs.csv`（约 651 MB，未提交 GitHub；可使用本目录脚本和固定 seed 重建）
- 机器可读指标：[metrics.json](metrics.json)

### 最容易混淆的 Different-ID 代表案例

| 排名 | video | frames | GT IDs | teams | StrongSORT距离 | 图片 |
|---:|---|---|---|---|---:|---|
| 1 | SNGS-046 | 316 / 363 | 9 / 6 | left / left | 0.0846 | [预览](bad_cases/confusing_different_01_pair.jpg) |
| 2 | SNGS-051 | 133 / 384 | 3 / 21 | right / right | 0.0956 | [预览](bad_cases/confusing_different_02_pair.jpg) |
| 3 | SNGS-045 | 322 / 410 | 19 / 15 | left / left | 0.0984 | [预览](bad_cases/confusing_different_03_pair.jpg) |
| 4 | SNGS-048 | 147 / 543 | 18 / 22 | left / left | 0.1015 | [预览](bad_cases/confusing_different_04_pair.jpg) |
| 5 | SNGS-048 | 234 / 245 | 4 / 18 | left / left | 0.1030 | [预览](bad_cases/confusing_different_05_pair.jpg) |

这些案例多为同队、相似球衣且人物 crop 较模糊，说明“同队外观相似”确实会形成低距离负样本。

### 距离最大的 Same-ID 代表案例

| 排名 | video | frames | GT IDs | teams | StrongSORT距离 | 图片 |
|---:|---|---|---|---|---:|---|
| 1 | SNGS-026 | 267 / 633 | 6 / 6 | left / left | 0.6337 | [预览](bad_cases/largest_same_01_pair.jpg) |
| 2 | SNGS-038 | 315 / 652 | 3 / 3 | left / left | 0.6321 | [预览](bad_cases/largest_same_02_pair.jpg) |
| 3 | SNGS-038 | 652 / 670 | 3 / 3 | left / left | 0.6270 | [预览](bad_cases/largest_same_03_pair.jpg) |
| 4 | SNGS-083 | 302 / 670 | 9 / 9 | left / left | 0.6265 | [预览](bad_cases/largest_same_04_pair.jpg) |
| 5 | SNGS-026 | 266 / 633 | 6 / 6 | left / left | 0.6222 | [预览](bad_cases/largest_same_05_pair.jpg) |

这些案例中可见极小 bbox、严重模糊或 crop 几乎只包含草地，因此高 Same-ID 距离不能全部归因于 embedding；GT bbox 可见性与遮挡是明确混杂因素。

## 解释边界

- 这个实验隔离了 GT bbox 上的 ReID embedding；它没有运行 detector 或 StrongSORT。
- team 只用于定义负样本难度，不作为人物 identity 真值。
- Pair 采样避免单个长轨迹产生不可控的二次方文件，但结果仍可能受视频构图、bbox 尺寸和遮挡分布影响。
- 下一步若要定位 tracker，应固定这些 GT detection/embedding，单独评估 StrongSORT 的关联结果。
