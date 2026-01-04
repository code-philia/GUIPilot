# RQ1：屏幕不一致性检测实验

本实验复现论文中针对屏幕级不一致性的评估。脚本会对真实界面截图应用多种扰动（插入、删除、文本/颜色修改、组件交换等），并统计匹配器与检测器组合的表现。

## 数据准备

### 数据集结构

数据集根目录需包含若干子目录，每个子目录下包含 `*.jpg` 截图与同名 `*.json` 标注文件。

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

1. **图片文件**（`*.jpg`）：
   - 文件名必须为纯数字（如 `1.jpg`、`6.jpg`），脚本会按数字升序遍历
   - 支持任意子目录层级，脚本会递归查找所有符合条件的图片

2. **标注文件**（`*.json`）：
   - 必须与对应图片同名（如 `1.json` 对应 `1.jpg`）
   - 位于同一目录下
   - JSON 格式：包含组件列表，每个组件包含四点坐标（`bbox`）和组件类型（`type`）

**标注文件示例**（`1.json`）：

```json
[
  {
    "type": "button",
    "bbox": [100, 200, 300, 260]
  },
  {
    "type": "text",
    "bbox": [50, 100, 250, 150]
  }
]
```

### 数据路径配置

可通过两种方式指定数据集路径：

1. **环境变量**（推荐）：设置 `DATASET_PATH` 环境变量
   ```bash
   export DATASET_PATH=/path/to/datasets/new
   ```

2. **命令行参数**：使用 `--dataset` 参数覆盖环境变量
   ```bash
   python experiments/rq1_screen_inconsistency/main.py --dataset /path/to/datasets/new
   ```

## 运行方式

### 完整评估（推荐用于论文复现）

```bash
conda activate guipilot  # 或 guipilot-gpu
python experiments/rq1_screen_inconsistency/main.py \
  --dataset datasets/new \
  --output-dir runs/rq1/full
```

脚本会：
- 遍历数据集中所有符合规则的截图（文件名为数字的 `*.jpg`）
- 对每个截图应用 5 种扰动：`insert_row`、`delete_row`、`swap_widgets`、`change_widgets_text`、`change_widgets_color`
- 使用两种匹配器（`gvt`、`guipilot`）和一种检测器（`gvt`）进行组合评估
- 生成详细的评估报告和可视化结果

### 快速验证（推荐用于 CI 烟测）

```bash
python experiments/rq1_screen_inconsistency/main.py \
  --dataset datasets/new \
  --limit 2 \
  --skip-visualize \
  --output-dir runs/rq1/smoke
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `DATASET_PATH` 环境变量 | 数据集根目录路径，包含 `*.jpg/*.json` 文件对 |
| `--limit` | int | `None`（处理全部） | 限制处理的截图数量（按数字排序后取前 N 个），用于快速验证 |
| `--output-dir` | str | `.`（当前目录） | 输出目录，用于存储 CSV 结果和可视化文件 |
| `--skip-visualize` | flag | `False` | 跳过生成可视化图片和日志，加快执行速度（推荐用于 CI） |

### 参数使用建议

- **完整评估**：不设置 `--limit`，保留默认 `--output-dir` 或指定自定义目录，不添加 `--skip-visualize`
- **快速验证**：设置 `--limit 2`，添加 `--skip-visualize`，指定临时输出目录
- **CI 烟测**：设置 `--limit 1`，添加 `--skip-visualize`，输出到 `ci-artifacts/rq1`

## 输出内容

### 1. 评估结果 CSV（`evaluation.csv`）

位于输出目录根目录，包含以下列：

| 列名 | 说明 |
|------|------|
| `id` | 图片文件路径 |
| `mutation` | 扰动类型（`insert_row`、`delete_row`、`swap_widgets`、`change_widgets_text`、`change_widgets_color`） |
| `matcher` | 匹配器名称（`gvt`、`guipilot`） |
| `checker` | 检测器名称（`gvt`） |
| `cls_tp` | 分类真阳性（正确匹配的不一致对及其类型） |
| `tp` | 真阳性（正确匹配的不一致对） |
| `fp` | 假阳性（误报的不一致对） |
| `fn` | 假阴性（漏检的不一致对） |
| `match_time` | 匹配耗时（秒） |
| `check_time` | 检测耗时（秒） |

**示例输出**：

```csv
id,mutation,matcher,checker,cls_tp,tp,fp,fn,match_time,check_time
datasets/new/Adobe/1.jpg,insert_row,gvt,gvt,2,3,1,0,0.12,0.05
datasets/new/Adobe/1.jpg,delete_row,guipilot,gvt,3,3,0,1,0.15,0.06
```

### 2. 可视化目录（`visualize/`）

仅在未使用 `--skip-visualize` 时生成，目录结构：

```
visualize/
  {matcher}_{checker}/
    {mutation}/
      {subdir}/
        {filename}.jpg    # 配对可视化图片
        {filename}.txt    # 详细检测日志
```

**可视化文件说明**：

- `*.jpg`：显示原始屏幕、扰动后屏幕、匹配的组件对、检测到的不一致（用不同颜色标注）
- `*.txt`：包含匹配结果、预测不一致、真实不一致、编辑距离等详细信息

### 3. 控制台输出

脚本运行时会实时输出每个评估任务的进度和指标：

```
insert_row | Adobe/1.jpg | gvt       | gvt       | 0.67 0.75 1.00
delete_row | Adobe/1.jpg | guipilot  | gvt       | 1.00 1.00 0.75
```

格式：`{mutation} | {路径} | {matcher} | {checker} | {cls_precision} {precision} {recall}`

## 环境变量配置

如需启用本地检测器或 OCR，可设置以下环境变量（详见 `docs/environment-setup.md`）：

- `ENABLE_LOCAL_DETECTOR=1`：启用本地 YOLOv8 检测器
- `DETECTOR_WEIGHT_PATH`：自定义检测器权重路径
- `ENABLE_PADDLEOCR=1`：启用本地 PaddleOCR
- `PADDLEOCR_USE_GPU=1`：尝试使用 GPU 加速 OCR
- `DETECTOR_SERVICE_URL`：使用远程检测服务
- `OCR_SERVICE_URL`：使用远程 OCR 服务

## 注意事项

1. **随机种子**：脚本固定使用随机种子 `42`，确保结果可复现
2. **扰动强度**：所有扰动默认使用 `0.05` 的强度参数
3. **文件命名**：图片文件名必须为纯数字，否则会被跳过
4. **后处理**：不同扰动类型会应用不同的后处理过滤策略，以消除误报
5. **错误处理**：如果某个截图处理失败，脚本会跳过并继续处理下一个
