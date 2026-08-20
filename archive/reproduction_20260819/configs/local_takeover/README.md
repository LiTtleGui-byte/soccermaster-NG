# Local takeover configs

这些配置把运行路径固定到本地`vendor/`、`.local_assets/`和用户受控NAS bundle。它们目前只完成静态准备，不构成GPU、推理、评估或训练授权。

- `assets.yaml`：统一路径注册表。
- `soccermaster_model.yaml`：SoccerMaster主干模型与checkpoint位置。
- `g10_step1_sngs10004.yaml`：本地数据/权重版Step 1。
- `g10_enrichment_sngs10004.yaml`：本地数据/权重版enrichment。
- `g10_refiner_sngs10004.json`：使用vendored Refiner配置的255帧override。

解说生成分支无需新复制，通过以下环境变量使用已经验证的用户NAS bundle：

```bash
SOCCERMASTER_COMMENTARY_ASSET_ROOT=/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817
```

批次1大资产已于2026-08-19迁入`.local_assets/`。固定SNGS-10004已经用这些配置/资产完成SoccerMaster五头、SoccerFactory Step 1、enrichment、Refiner、Step 3、训练PKL转换和DataLoader消费。任何未来GPU重跑仍需新鲜资源报告和当次授权。

批次1复制入口为`scripts/migrate_local_assets_batch1.py`；正式运行结果在`.runtime/local_takeover/batch1/result.json`。该入口拒绝覆盖，因此不是日常运行命令。
