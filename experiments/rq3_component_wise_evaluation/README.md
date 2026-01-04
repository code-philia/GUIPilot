# RQ3：组件级评估实验

本实验用于复现论文中的组件级消融评估。脚本会对 GUIPilot 管线中的关键组件（OCR、匹配器、后处理等）进行消融研究，在统一数据集和指标上比较不同组合的表现。

## 数据准备

### 数据集结构

与 RQ1 相同，数据集根目录需包含若干子目录，每个子目录下包含 `*.jpg` 截图与同名 `*.json` 标注文件。

**示例结构**（参考仓库内的 `datasets/new/`）：

```
datasets/new/
  Adobe/
    1.jpg
    1.json
    6.jpg
    6.json
  Bitget1/
    1.jpg
    1.json
    3.jpg
    3.json
```

### 文件格式要求

与 RQ1 完全相同：

1. **图片文件**（`*.jpg`）：文件名必须为纯数字（如 `1.jpg`、`6.jpg`）
2. **标注文件**（`*.json`）：与图片同名，包含组件列表（四点坐标和类型）

详细格式说明请参考 [RQ1 README](./rq1_screen_inconsistency/README.md#文件格式要求)。

### 数据路径配置

可通过两种方式指定数据集路径：

1. **环境变量**：设置 `DATASET_PATH` 环境变量
2. **命令行参数**：使用 `--dataset` 参数

## 运行方式

### 完整评估（推荐用于论文复现）

```bash
conda activate guipilot  # 或 guipilot-gpu
python experiments/rq3_component_wise_evaluation/main.py \
  --dataset datasets/new \
  --output-dir runs/rq3/full
```

脚本会：
- 使用默认的管线组合（`guipilot_full`、`guipilot_no_postprocess`、`gvt_matcher`）
- 对每个截图应用 5 种扰动（与 RQ1 相同）
- 评估所有管线组合的表现
- 生成汇总指标和详细记录

### 指定管线组合

```bash
python experiments/rq3_component_wise_evaluation/main.py \
  --dataset datasets/new \
  --pipelines guipilot_full,guipilot_no_ocr,gvt_matcher \
  --output-dir runs/rq3/custom
```

### 快速验证（推荐用于 CI 烟测）

```bash
python experiments/rq3_component_wise_evaluation/main.py \
  --dataset datasets/new \
  --limit 1 \
  --pipelines guipilot_full \
  --skip-visualize \
  --output-dir ci-artifacts/rq3
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `DATASET_PATH` 环境变量 | 数据集根目录路径，包含 `*.jpg/*.json` 文件对 |
| `--limit` | int | `None`（处理全部） | 限制评估的屏幕数量（按数字排序后取前 N 个） |
| `--output-dir` | str | `.`（当前目录） | 输出目录，用于存储 CSV 结果和可视化文件 |
| `--pipelines` | str | `guipilot_full,guipilot_no_postprocess,gvt_matcher` | 逗号分隔的管线预设名称列表 |
| `--skip-visualize` | flag | `False` | 跳过生成可视化图片，加快执行速度（推荐用于 CI） |
| `--seed` | int | `42` | 随机种子，用于确保变异器的可复现性 |

### 管线预设说明

| 管线名称 | OCR | 匹配器 | 后处理 | 说明 |
|---------|-----|--------|--------|------|
| `guipilot_full` | ✅ | GUIPilotV2 | ✅ | 完整流程（OCR + GUIPilot 匹配器 + 后处理过滤） |
| `guipilot_no_postprocess` | ✅ | GUIPilotV2 | ❌ | 关闭后处理，观察过滤策略的收益 |
| `guipilot_no_ocr` | ❌ | GUIPilotV2 | ✅ | 跳过 OCR，评估文本识别组件的重要性 |
| `gvt_matcher` | ✅ | GVT | ✅ | 改用 GVT 匹配器，对比匹配器差异 |

### 参数使用建议

**完整评估**：
```bash
--dataset datasets/new --output-dir runs/rq3/full
# 使用默认管线组合，生成完整可视化
```

**消融研究**：
```bash
--pipelines guipilot_full,guipilot_no_postprocess,guipilot_no_ocr
# 对比完整流程、无后处理、无 OCR 三种组合
```

**CI 烟测**：
```bash
--limit 1 --pipelines guipilot_full --skip-visualize --output-dir ci-artifacts/rq3
# 只评估一个屏幕，使用单个管线，跳过可视化
```

## 输出内容

### 1. 汇总结果 CSV（`summary.csv`）

位于输出目录根目录，包含各管线组合的聚合指标：

| 列名 | 说明 |
|------|------|
| `pipeline` | 管线名称 |
| `cls_tp` | 分类真阳性总数 |
| `tp` | 真阳性总数 |
| `fp` | 假阳性总数 |
| `fn` | 假阴性总数 |
| `precision` | 精确率（tp / (tp + fp)） |
| `recall` | 召回率（tp / (tp + fn)） |
| `cls_precision` | 分类精确率（cls_tp / tp） |
| `avg_match_time_ms` | 平均匹配耗时（毫秒） |
| `avg_check_time_ms` | 平均检测耗时（毫秒） |
| `evaluations` | 评估任务总数 |

**示例输出**：

```csv
pipeline,cls_tp,tp,fp,fn,precision,recall,cls_precision,avg_match_time_ms,avg_check_time_ms,evaluations
guipilot_full,45,50,5,10,0.9091,0.8333,0.9,150.23,65.45,100
guipilot_no_postprocess,40,55,15,5,0.7857,0.9167,0.7273,148.12,64.89,100
gvt_matcher,35,45,8,15,0.8491,0.75,0.7778,120.45,58.23,100
```

### 2. 详细记录 CSV（`{pipeline}/evaluation.csv`）

每个管线组合都会在输出目录下创建子目录，包含详细的评估记录：

| 列名 | 说明 |
|------|------|
| `pipeline` | 管线名称 |
| `image` | 图片文件路径 |
| `mutation` | 扰动类型 |
| `matcher` | 匹配器名称 |
| `checker` | 检测器名称 |
| `cls_tp` | 分类真阳性 |
| `tp` | 真阳性 |
| `fp` | 假阳性 |
| `fn` | 假阴性 |
| `match_time` | 匹配耗时（秒） |
| `check_time` | 检测耗时（秒） |

**目录结构**：

```
output-dir/
  summary.csv                    # 汇总指标
  guipilot_full/
    evaluation.csv               # 详细记录
    visualize/                   # 可视化文件（如未跳过）
      {mutation}/
        {subdir}/
          {filename}.jpg
  guipilot_no_postprocess/
    evaluation.csv
    visualize/
  gvt_matcher/
    evaluation.csv
    visualize/
```

### 3. 可视化目录（`{pipeline}/visualize/`）

仅在未使用 `--skip-visualize` 时生成，结构与 RQ1 相同：

```
{pipeline}/visualize/
  {mutation}/
    {subdir}/
      {filename}.jpg    # 配对可视化图片
```

### 4. 控制台输出

脚本运行时会实时输出每个管线组合的评估进度和最终指标：

```
[RQ3] Evaluating 10 screens with pipelines: guipilot_full, guipilot_no_postprocess, gvt_matcher
[guipilot_full] cls_precision=0.90 precision=0.91 recall=0.83 evaluations=100
[guipilot_no_postprocess] cls_precision=0.73 precision=0.79 recall=0.92 evaluations=100
[gvt_matcher] cls_precision=0.78 precision=0.85 recall=0.75 evaluations=100
[RQ3] Summaries written to runs/rq3/full/summary.csv
```

## 环境变量配置

与 RQ1 相同，支持以下环境变量：

- `ENABLE_LOCAL_DETECTOR=1`：启用本地 YOLOv8 检测器
- `DETECTOR_WEIGHT_PATH`：自定义检测器权重路径
- `ENABLE_PADDLEOCR=1`：启用本地 PaddleOCR
- `PADDLEOCR_USE_GPU=1`：尝试使用 GPU 加速 OCR
- `DETECTOR_SERVICE_URL`：使用远程检测服务
- `OCR_SERVICE_URL`：使用远程 OCR 服务

详细说明请参考 `docs/environment-setup.md`。

## 注意事项

1. **随机种子**：默认使用 `42`，确保结果可复现
2. **扰动类型**：与 RQ1 相同，包含 5 种扰动（插入、删除、交换、文本修改、颜色修改）
3. **后处理**：`guipilot_full` 和 `guipilot_no_postprocess` 的差异在于后处理过滤策略
4. **OCR 依赖**：`guipilot_no_ocr` 会跳过 OCR 步骤，评估文本识别的重要性
5. **匹配器差异**：`gvt_matcher` 使用 GVT 匹配器，需要传入屏幕高度参数
6. **错误处理**：如果某个评估任务失败，脚本会跳过并继续处理下一个

## 与 RQ1 的关系

RQ3 复用了 RQ1 的以下组件：

- **数据集格式**：完全相同的数据集结构
- **变异器**：相同的 5 种扰动操作
- **后处理**：相同的过滤策略
- **指标计算**：相同的评估指标

主要差异：

- **评估目标**：RQ1 评估匹配器和检测器的组合，RQ3 评估不同管线配置
- **输出格式**：RQ3 提供汇总指标和分管线详细记录
- **可视化**：RQ3 的可视化按管线分组，便于对比

## CI 烟测示例

CI 工作流使用的命令：

```bash
python experiments/rq3_component_wise_evaluation/main.py \
  --dataset datasets/new \
  --limit 1 \
  --pipelines guipilot_full \
  --skip-visualize \
  --output-dir ci-artifacts/rq3
```

该命令会：
- 只评估一个屏幕
- 使用单个管线组合（`guipilot_full`）
- 跳过可视化生成
- 输出到 CI  artifacts 目录
