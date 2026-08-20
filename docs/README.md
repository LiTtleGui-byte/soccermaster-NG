# 文档索引

这份索引回答“当前事实在哪里、代码在哪里、资产在哪里、历史材料在哪里”。新工作优先从这里进入，不需要遍历整个仓库。

## 当前权威文档

| 问题 | 文档 |
|---|---|
| 项目现在做到哪一步 | [`../REPRODUCTION_STATUS.md`](../REPRODUCTION_STATUS.md) |
| 整个目录怎样组织 | [`LOCAL_PROJECT_LAYOUT.md`](LOCAL_PROJECT_LAYOUT.md) |
| 数据和checkpoint在哪里 | [`ASSET_INVENTORY.md`](ASSET_INVENTORY.md) |
| SoccerMaster代码如何流动 | [`PIPELINE_MAP_SOCCERMASTER.md`](PIPELINE_MAP_SOCCERMASTER.md) |
| SoccerFactory代码如何流动 | [`PIPELINE_MAP_SOCCERFACTORY.md`](PIPELINE_MAP_SOCCERFACTORY.md) |
| 正式Gate怎样判定 | [`HARNESS.md`](HARNESS.md) |
| 长期安全和授权规则 | [`../AGENTS.md`](../AGENTS.md) |

## 按用途查找

- `guides/`：仍然有效的操作指南，例如训练恢复。
- `research/`：研究术语、每日实验记录和汇报材料。
- `adr/`：已经作出的架构决策及理由。
- `future_improvements/`：尚未验证的想法，不代表当前能力。
- `operations/`：GPU服务器、存储和集群运维知识。
- `paper/`：论文相关材料。

## 历史材料

重构前的脚本、配置和实验位于`../archive/`；原先已经进入Git的小型报告证据归档在`../archive/reports_legacy_text_20260819/`，其余本地运行产物位于`../runs/*_legacy_20260819/`。旧路径到新路径的对应关系见[`../archive/PATH_MAP.md`](../archive/PATH_MAP.md)。已经完成的迁移申请和接管方案位于`../archive/docs_migration_20260819/`，只作历史参考。

判断当前状态时不要从历史报告反推，应始终以根目录`REPRODUCTION_STATUS.md`为准。
