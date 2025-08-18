# VideoCaption失败例子分析功能

这个功能允许你在评估VideoCaption任务时，自动识别和保存失败的检索例子，帮助分析模型性能和改进方向。

## 功能特性

- **自动失败检测**: 识别当vision feature对应的text不是retrieval结果中的第一个时的情况
- **详细失败信息**: 记录失败例子的video路径、原始text、检索到的top-1 text、原始text的排名等
- **分布式支持**: 在多GPU训练/评估环境下正确工作
- **灵活配置**: 可以通过命令行参数启用/禁用此功能

## 使用方法

### 1. 基本用法

```bash
python eval.py \
    --config path/to/your/config.yaml \
    --checkpoint path/to/your/checkpoint \
    --save_video_caption_failures \
    --failure_save_path video_caption_failures.txt
```

### 2. 使用示例脚本

```bash
python example_eval_with_failures.py \
    --config path/to/your/config.yaml \
    --checkpoint path/to/your/checkpoint \
    --output_dir analysis_results
```

### 3. 参数说明

- `--save_video_caption_failures`: 启用失败例子保存功能
- `--failure_save_path`: 指定保存失败例子的文件路径（可选，默认自动生成）

## 输出格式

失败例子文件将包含以下信息：

```
=== Failure Case ===
Video Path: /path/to/video.mp4
Original Text: goal
Retrieved Top1 Text: penalty
Original Text Rank: 3
Similarity to Top1: 0.8234
Similarity to Original: 0.7891

=== Failure Case ===
Video Path: /path/to/another_video.mp4
Original Text: corner
Retrieved Top1 Text: free kick
Original Text Rank: 2
Similarity to Top1: 0.7654
Similarity to Original: 0.7321

...
```

### 字段说明

- **Video Path**: 失败例子对应的视频文件路径
- **Original Text**: 原始的ground truth文本标签
- **Retrieved Top1 Text**: 模型检索到的排名第一的文本
- **Original Text Rank**: 原始文本在检索结果中的排名位置
- **Similarity to Top1**: 视觉特征与top-1文本的相似度分数
- **Similarity to Original**: 视觉特征与原始文本的相似度分数

## 技术实现细节

### 1. 失败检测逻辑

对于每个vision feature：
1. 计算与所有text的相似度
2. 按相似度排序获得检索结果
3. 检查top-1结果是否为正样本（根据标签矩阵判断）
4. 如果不是，则记录为失败例子

### 2. 分布式处理

- 在分布式环境中，先收集所有GPU的vision features、text features和video paths
- 只在主进程（rank 0）进行失败分析和文件保存
- 确保避免多进程同时写入同一文件

### 3. 代码修改位置

#### VideoCaptionLoss类 (`models/video_caption.py`)
- 添加了`metas`参数和失败保存相关参数
- 实现了`_gather_video_paths_distributed`方法
- 实现了`_save_failure_cases`方法

#### 评估函数 (`train.py`)
- 修改`evaluate_one_epoch`函数支持失败保存参数
- 更新VideoCaption损失函数调用方式

#### 评估脚本 (`eval.py`)
- 添加命令行参数支持
- 修改`evaluation_engine`函数传递失败保存参数

## 示例分析工作流

1. **运行评估并收集失败例子**:
   ```bash
   python eval.py --config config.yaml --checkpoint model.pth --save_video_caption_failures --failure_save_path failures.txt
   ```

2. **分析失败例子**:
   - 查看失败例子的视频内容
   - 比较原始text和检索到的text的语义差异
   - 分析相似度分数的分布

3. **可能的改进方向**:
   - 如果相似度分数很接近，可能需要更细粒度的特征学习
   - 如果某些类别经常失败，可能需要增加该类别的训练数据
   - 如果检索到的text语义相近但标签不同，可能需要重新考虑标签策略

## 注意事项

1. **存储空间**: 失败例子文件会随着评估数据的增大而增大，注意磁盘空间
2. **性能影响**: 启用此功能会稍微增加评估时间和内存使用
3. **文件追加**: 失败例子会以追加模式写入文件，多次运行会累积结果
4. **分布式一致性**: 在分布式环境中，确保所有进程使用相同的随机种子以保证结果一致性

## 故障排除

### 常见问题

1. **文件权限错误**: 确保对输出目录有写权限
2. **内存不足**: 在大数据集上可能需要增加内存限制
3. **分布式同步问题**: 确保所有GPU正常通信

### 调试技巧

1. 先在小数据集上测试功能
2. 检查日志中的失败例子数量统计
3. 验证video path和text的对应关系是否正确

这个功能为VideoCaption任务的性能分析提供了强大的工具，帮助你深入理解模型的优缺点并指导后续的改进工作。
