# RQ2：流程不一致性检测实验

该实验复现论文中的流程不一致性评估。脚本会按照人工录制的交互序列回放界面，检测真实界面是否与原设计保持一致，并在发现异常时调用 VLM 助手尝试恢复流程。

## 数据准备

### 数据集结构

数据集根目录需要以应用包名划分子目录，每个子目录下包含若干 `process_*` 录制文件夹。

**示例结构**（参考仓库内的 `datasets/rq2/smoke_app/`）：

```
datasets/rq2/smoke_app/
  process_1/
    1.jpg              # 步骤 1 的设计截图
    1.xml              # 步骤 1 的布局 XML
    2.jpg              # 步骤 2 的设计截图
    2.xml              # 步骤 2 的布局 XML
    record.json        # 流程描述文件
    real/              # （可选）离线重放模式所需的真实截图
      1.jpg
      2.jpg
```

### 文件格式要求

1. **设计截图**（`*.jpg`）：
   - 按步骤编号命名（如 `1.jpg`、`2.jpg`）
   - 表示设计稿或 Mockup 中的界面状态

2. **布局文件**（`*.xml`）：
   - 与截图同名（如 `1.xml` 对应 `1.jpg`）
   - 包含 UI 控件的层次结构和属性信息
   - 用于 `--use-layout` 模式，避免调用检测服务

3. **流程描述文件**（`record.json`）：
   - 必须存在，包含完整的流程信息
   - JSON 格式，包含包名、初始 Activity、步骤列表等

**`record.json` 示例**：

```json
{
  "package_name": "com.example.app",
  "init_activity": "com.example.app.MainActivity",
  "steps": [
    {
      "description": "点击登录按钮",
      "action": {"type": "click", "bounds": [100, 200, 300, 260]},
      "layout": "1.xml"
    },
    {
      "description": "输入用户名",
      "action": {"type": "input", "text": "user@example.com", "bounds": [50, 300, 400, 350]},
      "layout": "2.xml"
    }
  ],
  "inconsistency_index": 0
}
```

**字段说明**：
- `package_name`：应用包名
- `init_activity`：启动 Activity
- `steps`：步骤列表，每个步骤包含描述、动作、布局文件名
- `inconsistency_index`（可选）：指定哪一步应该出现异常（0-based），用于 CI 烟测的确定性

4. **真实截图**（`real/*.jpg`，仅离线重放模式需要）：
   - 位于 `process_*/real/` 目录下
   - 按步骤编号命名（如 `real/1.jpg`、`real/2.jpg`）
   - 表示真实设备上的界面状态，用于离线重放模式

### 数据路径配置

可通过两种方式指定数据集路径：

1. **环境变量**：设置 `DATASET_PATH` 环境变量
2. **命令行参数**：使用 `--dataset` 参数

## 运行方式

### 1. 交互模式（需要真实设备）

```bash
conda activate guipilot
python experiments/rq2_flow_inconsistency/main.py \
  --dataset /path/to/rq2_dataset \
  --mode interactive \
  --output-dir runs/rq2/full
```

**使用场景**：需要连接真实 Android 设备，逐步执行交互并检测不一致性。

**执行流程**：
1. 脚本会通过 uiautomator2 连接默认设备
2. 启动应用并等待人工对齐屏幕
3. 按照 `record.json` 中的步骤执行操作
4. 在指定步骤（`inconsistency_index`）故意执行错误操作
5. 检测到不一致性后，调用 VLM 助手尝试恢复流程
6. 需要人工确认每个步骤的执行结果

**前置条件**：
- Android 设备已连接并启用 USB 调试
- 已安装 uiautomator2 并初始化设备
- 设置了 `OPENAI_KEY` 环境变量（如不跳过 agent）

### 2. 离线重放模式（推荐用于 CI 烟测）

```bash
python experiments/rq2_flow_inconsistency/main.py \
  --dataset datasets/rq2/smoke_app \
  --mode replay \
  --skip-agent \
  --limit 1 \
  --output-dir runs/rq2/smoke
```

**使用场景**：无需真实设备，从磁盘读取真实截图进行离线评估。

**关键特性**：
- `--mode replay`：不连接真机，从 `real/` 目录读取真实截图
- `--use-layout`：直接使用录制时的 XML 构建控件信息，避免调用检测服务
- `--skip-agent`：跳过 VLM 辅助阶段，加快执行速度
- `--limit`：只评估部分流程，控制执行时间

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `DATASET_PATH` 环境变量 | 数据集根目录路径，包含 `process_*` 目录 |
| `--limit` | int | `None`（处理全部） | 限制评估的流程数量（按目录名排序后取前 N 个） |
| `--output-dir` | str | `runs/rq2` | 输出目录，用于存储 CSV 结果和可视化文件 |
| `--mode` | str | `interactive` | 运行模式：`interactive`（需要设备）或 `replay`（离线重放） |
| `--replay-real-subdir` | str | `real` | 离线重放模式中真实截图的子目录名 |
| `--use-layout` | flag | `False` | 使用录制的 XML 布局文件构建控件信息（仅 replay 模式有效） |
| `--skip-agent` | flag | `False` | 跳过 VLM agent 阶段，加快执行速度（推荐用于 CI） |
| `--process-pattern` | str | `process_*` | 流程目录的 glob 模式 |
| `--seed` | int | `42` | 随机种子，用于选择不一致步骤（如果未在 record.json 中指定） |
| `--inconsistency-index` | int | `None` | 强制指定不一致步骤索引（0-based），覆盖 record.json 和随机选择 |
| `--results-file` | str | `results.csv` | 结果 CSV 文件名（位于输出目录内） |
| `--openai-key` | str | `OPENAI_KEY` 环境变量 | OpenAI API 密钥（仅在未跳过 agent 时需要） |

### 参数使用建议

**完整评估（交互模式）**：
```bash
--mode interactive --output-dir runs/rq2/full
# 需要设置 OPENAI_KEY 环境变量，或使用 --openai-key
```

**CI 烟测（离线重放模式）**：
```bash
--mode replay --use-layout --skip-agent --limit 1 --output-dir ci-artifacts/rq2
```

**快速验证（离线重放，部分流程）**：
```bash
--mode replay --use-layout --skip-agent --limit 2 --output-dir runs/rq2/smoke
```

## 输出内容

### 1. 结果 CSV（`results.csv`）

位于输出目录根目录，包含以下列：

| 列名 | 说明 |
|------|------|
| `id` | 流程标识（包名::流程名） |
| `score1` | 第一步的匹配得分 |
| `score2` | 第二步的匹配得分 |
| `score3` | 第三步的匹配得分 |
| `action_time` | 动作执行耗时（秒，仅交互模式） |
| `time1` | 第一步检测耗时（秒） |
| `time2` | 第二步检测耗时（秒） |
| `time3` | 第三步检测耗时（秒） |
| `ground_truth` | 真实标签（是否存在不一致性） |
| `is_completed` | 流程是否完成 |
| `retries` | VLM agent 重试次数（如未跳过 agent） |

**示例输出**：

```csv
id,score1,score2,score3,action_time,time1,time2,time3,ground_truth,is_completed,retries
com.example.app::process_1,0.95,0.82,0.91,1.23,0.15,0.18,0.16,True,True,0
```

### 2. 可视化目录（`visualize/`）

目录结构：

```
visualize/
  {package_name}-{process_name}/
    step_{index}.jpg    # 每一步的匹配可视化结果
```

**可视化内容**：
- 显示设计截图和真实截图的对比
- 标注匹配的组件对
- 高亮检测到的不一致性

### 3. 控制台输出

脚本运行时会实时输出每个流程的进度：

```
[1/2] com.example.app::process_1 | inconsistent step = 0
  - Step 1/2: 点击登录按钮
  - Step 2/2: 输入用户名
```

## 环境变量配置

- `OPENAI_KEY`：OpenAI API 密钥（仅在未使用 `--skip-agent` 时需要）
- `ENABLE_LOCAL_DETECTOR=1`：启用本地检测器（如不使用 `--use-layout`）
- `DETECTOR_WEIGHT_PATH`：自定义检测器权重路径
- `ENABLE_PADDLEOCR=1`：启用本地 OCR
- 其他环境变量详见 `docs/environment-setup.md`

## 注意事项

1. **交互模式**：需要真实 Android 设备，且已初始化 uiautomator2
2. **离线重放模式**：需要提供 `real/` 目录下的真实截图
3. **布局文件**：使用 `--use-layout` 时，必须提供对应的 `*.xml` 文件
4. **不一致步骤**：可通过 `record.json` 中的 `inconsistency_index` 字段或 `--inconsistency-index` 参数指定
5. **VLM Agent**：如未设置 `OPENAI_KEY` 且未使用 `--skip-agent`，脚本会报错退出
6. **随机种子**：用于选择不一致步骤（如果未显式指定），默认值为 `42`

## GitHub Actions 烟测

工作流文件 `.github/workflows/env-matrix-check.yml` 会在 CI 中执行离线重放模式，验证依赖与脚本逻辑是否正确。可通过 `workflow_dispatch` 手动触发进行快速回归。

CI 使用的命令：

```bash
python experiments/rq2_flow_inconsistency/main.py \
  --dataset datasets/rq2/smoke_app \
  --mode replay \
  --use-layout \
  --skip-agent \
  --limit 1 \
  --output-dir ci-artifacts/rq2
```
