# SoccerMaster / SoccerFactory 接管指南（历史归档）

> 本文记录2026-08-19接管过程中的设计和当时路径，已被当前目录结构取代。当前入口见`../../docs/README.md`和`../../docs/LOCAL_PROJECT_LAYOUT.md`。

更新：2026-08-19。

这份文档回答三个问题：现在到底跑通了什么、每个模块的代码在哪里、要怎样逐步摆脱远端路径依赖。具体资产和大小见`ASSET_INVENTORY.md`，逐模块代码见两份`PIPELINE_MAP`。

## 1. 跑通状态

| 系统 | 已跑通 | 没有证明 |
|---|---|---|
| SoccerFactory | 固定SNGS-10004从原始半场视频重建255帧，串到人物检测/跟踪、球场线/关键点、相机与二维坐标、角色/球队/号码、Refiner、适配Step 3、训练PKL和真实DataLoader | 其他比赛、字段语义正确、原版Step 3直接接Refiner、Refiner提高绝对坐标准确率 |
| SoccerMaster主干 | 同一固定片段经真实DataLoader完成检测、球场线、关键点、23类事件分类和23条事件短语检索五头forward，并保存可视图 | 全数据指标、自由生成解说、完整G9训练 |
| SoccerMaster训练 | G1–G8通过；G9六卡跑到8,299/16,009步且无OOM | 单epoch结束、epoch checkpoint、20 epochs及最终模型效果 |
| MatchTime生成解说 | 独立生成模型可以推理；逐层缓存、Q-Former probe和首轮干预已运行 | 尚未完成干预输出的事实盲审，不能确认Q-Former是唯一瓶颈 |

这里的“跑通”只表示输入、代码和输出接口形成了真实闭环，不等于输出标签准确。

## 2. 一站式可视化

固定单场总报告：`runs/reports_legacy_20260819/one_match/20260819_sngs10004_end_to_end/summary.md`。

SoccerFactory八张图依次展示：

1. 原始视频到255帧；
2. 人物框和track ID；
3. ReID轨迹；
4. 球场线和关键点；
5. 相机标定和二维球场坐标；
6. 号码、角色和球队；
7. Refiner前后坐标；
8. Step 3、PKL和DataLoader。

SoccerMaster五张图展示：人物检测、球场线热图、关键点热图、事件分类top-5，以及视频—固定短语检索top-5。

## 3. 代码导航

### SoccerMaster主干

```text
data/*.py
  → models/multi_task.py
      → models/siglip2_unisoccer_part_temporal.py
      → models/modeling_timesformer_siglip.py
      → models/deformable_detr/deformable_detr.py
      → models/lines_detection.py
      → models/keypoints_detection.py
      → models/video_caption.py
      → models/caption_classification.py
  → train.py / eval.py
```

这些代码已经在本地，可以直接修改。SigLIP2预训练目录和完整`epoch_19`权重仍在远端。

### MatchTime生成解说

```text
experiments/commentary_generation/runtime/dataset/commentary.py
  → runtime/model/MatchVision_part_temporal.py
  → runtime/model/matchvoice_Qformer.py
  → runtime/model/matchvoice_model_all_blocks.py
      ├── video_Qformer
      ├── llama_proj
      └── generate_text / Llama
```

运行代码闭包已经vendored到本地，但Llama、BERT配置、生成checkpoint和视觉backbone仍在远端。

### SoccerFactory

```text
本地安全入口：reproduction/gates/g10_*.py
本地固定配置：reproduction/configs/g10/

本地 vendored 实现：
vendor/soccerfactory/tracklab/tracklab + vendor/soccerfactory/tracklab/plugins
  → sn-gamestate/sn_gamestate + sn-gamestate/plugins
  → vendor/soccerfactory/refiner
```

固定样本的 Step 1、enrichment 和 Refiner 已经从本地 vendored 源码动态运行。历史 Gate 仍保留运行当时的远端路径，不能据此误判当前入口仍依赖远端。项目自有的跨系统格式转换统一放在 `research/src/soccermaster/integrations/soccerfactory/`。

更具体的类、函数和修改入口见：

- `docs/PIPELINE_MAP_SOCCERMASTER.md`
- `docs/PIPELINE_MAP_SOCCERFACTORY.md`
- `docs/LOCAL_PROJECT_LAYOUT.md`

## 4. 目标目录

后续不移动当前已经工作的`research/src/soccermaster/models/`、`research/src/soccermaster/data/`和`archive/reproduction_20260819/`。新增本地资产使用下面的隔离布局：

```text
/home/tianlin/SoccerMaster/
├── vendor/
│   └── soccerfactory/
│       ├── tracklab/
│       ├── sn_gamestate/
│       └── refiner/
├── .local_assets/                 # Git忽略
│   ├── models/
│   │   └── soccerfactory/
│   │       ├── pretrained_models/
│   │       └── refiner/
│   ├── checkpoints/
│   │   └── soccermaster_epoch19/
│   └── data/
│       ├── SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1/
│       └── soccernet/
│           ├── raw/sngs10004/
│           └── cameras/sngs10004/
├── reproduction/                 # 固定复现入口
├── experiments/                  # 新实验
└── reports/                      # 证据与可视化
```

`vendor/`只放实际import到的第三方源码和来源说明，不复制TB级历史outputs、训练数据或无关示例。`assets/`只放不可提交的大型权重和固定数据。

## 5. 本地化顺序

1. 静态解析固定G10配置和Python import，列出TrackLab/sn-gamestate/Refiner最小源码闭包。
2. 报告源码闭包的精确文件、大小和许可/来源；复制到`baseline/code/soccerfactory/`。
3. 修改本地配置和`PYTHONPATH`，不再从远端import SoccerFactory源码；用固定SNGS-10004做一次最小结构复核。
4. 已完成批次1迁移：主干`epoch_19`、SoccerFactory权重和固定单场数据位于`assets/`；SigLIP2直接复用用户受控NAS bundle。
5. MatchTime生成解说所需Llama/Q-Former/生成checkpoint已经位于用户NAS bundle，通过环境变量使用，不再复制一份。
6. 最后查询代码与配置中的`/remote-home`依赖；保留来源文档中的历史路径，但运行配置不得再依赖远端。

## 6. 完成标准

只有同时满足以下条件，才称为“代码和固定复现资产已经变成自己的”：

- SoccerMaster和SoccerFactory实际运行源码都位于本地；
- 固定复现所需权重和SNGS-10004输入都位于`assets/`；
- 正式本地配置不再读取远端源码、权重或固定输入；
- 同一固定片段重新产生两条链路的关键结构输出和可视化；
- 资产清单记录来源、大小、用途和验证状态；
- 仍清楚区分接口跑通、完整训练和语义准确率。

固定样本接管已完成。源码已经vendored且关键import通过，批次1权重和输入已迁移，无`/remote-home`运行配置位于`archive/reproduction_20260819/configs/local_takeover/`。SoccerMaster已完成本地五头GPU forward；SoccerFactory已用本地TrackLab/sn-gamestate/Refiner及全部本地权重完成Step 1、enrichment、Refiner、保留版Step 3、训练PKL转换和真实DataLoader消费。统一证据入口为`runs/reports_legacy_20260819/local_takeover/README.md`。

这里的“完成”只指固定样本代码接口和资产接管。G9完整训练仍未通过，SoccerFactory角色、球队、号码和坐标也仍没有独立真值支持，不能写成总体质量已经复现。
