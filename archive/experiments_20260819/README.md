# SoccerMaster 改进实验登记

`experiments/` 只用于基线复现之后的方法改进、消融和错误分析。已验证的 Gate 入口位于 `reproduction/`，未验证的研究假设位于 `docs/future_improvements/`。

## 建议结构

```text
experiments/<experiment_name>/
├── README.md
└── config.yaml
```

运行日志、权重和预测不放在此目录；它们应进入 `reports/` 或 `outputs/`。

## 每个实验必须记录

```text
实验名称：
状态：设计中 / 已批准 / 运行中 / 通过 / 失败 / 中止
研究假设：
对应的 future_improvements 文档：
基线 Git commit：
基线 config 和 checkpoint：
修改文件：
配置差异：
数据和 manifest：
运行命令与授权：
结果路径：
与基线的对比：
结论和未验证范围：
```

## 当前登记

尚无已批准的改进实验。G8 固定小规模多任务基线已完成功能性恢复验证，但位级 `exact_resume` 未通过；GradNorm、PCGrad、动态 loss weighting、adapter 或特征分配改动均尚未执行，也不得回写或混入基线 Gate。
