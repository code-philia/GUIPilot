# RQ4：案例研究实验

本实验用于复现论文中的案例研究：对比移动应用的 Mockup 设计稿与真实实现，检查界面一致性并验证流程动作的正确性。

## 数据准备

### 数据集结构

数据集根目录需包含若干应用目录，每个应用目录下包含若干流程目录，每个流程目录至少包括 `mockup/` 和 `implementation/` 两个子目录。

**示例结构**：

```
<dataset_root>/
  <app_name>/
    process_1/
      implementation/
        process.json
        1.jpg
        2.jpg
        ...
      mockup/
        1.png
        2.png
        ...
    process_2/
      implementation/
        process.json
        1.jpg
        ...
      mockup/
        1.png
        ...
```

### 文件格式要求

1. **实现截图**（`implementation/*.jpg`）：
   - 按步骤编号命名（如 `1.jpg`、`2.jpg`）
   - 表示真实应用中的界面状态

2. **Mockup 截图**（`mockup/*.png` 或 `*.jpg`）：
   - 与实现截图文件名一一对应（如 `1.png` 对应 `1.jpg`）
   - 表示设计稿中的界面状态
   - 通常为 PNG 格式，但也支持 JPG

3. **流程描述文件**（`implementation/process.json`）：
   - 必须存在，包含流程的屏幕序列和动作信息
   - JSON 格式，按顺序记录每个屏幕的信息

**`process.json` 示例**：

```json
[
  {
    "screen": "1.jpg",
    "mock_actions": ["点击登录按钮"],
    "actions": [
      {
        "action": "click",
        "bounds": [100, 200, 300, 260]
      }
    ]
  },
  {
    "screen": "2.jpg",
    "mock_actions": ["输入用户名"],
    "actions": [
      {
        "action": "input",
        "text": "user@example.com",
        "bounds": [50, 300, 400, 350]
      }
    ]
  }
]
```

**字段说明**：
- `screen`：屏幕文件名（对应 `implementation/` 和 `mockup/` 中的文件）
- `mock_actions`：设计稿中描述的动作（自然语言）
- `actions`：真实实现中的动作列表，每个动作包含：
  - `action`：动作类型（`click`、`input`、`swipe` 等）
  - `bounds`：动作区域的边界框 `[x1, y1, x2, y2]`
  - `text`：（可选）输入文本（仅 `input` 动作）

### 数据路径配置

可通过三种方式指定数据集路径：

1. **环境变量**：设置 `DATASET_PATH` 环境变量
2. **命令行参数**：使用 `--dataset` 参数

### 获取真实数据集

论文中使用的真实案例数据集可从以下链接获取：

- **数据集链接**：[Zenodo](https://zenodo.org/records/15107436)
- **项目页面**：[Project Page](https://sites.google.com/view/guipilot/home)

下载数据集后，解压到本地目录，即可使用 `--dataset` 参数指定路径。

### 准备自己的真实数据

如果需要使用自己的真实案例数据，请按以下步骤准备：

1. **收集 Mockup 设计稿**：
   - 从设计工具（Figma、Sketch、Adobe XD 等）导出设计稿截图
   - 保存为 PNG 或 JPG 格式
   - 按步骤编号命名（如 `1.png`、`2.png`、`3.png`）
   - 放置在 `mockup/` 目录下

2. **收集实现截图**：
   - 从真实应用中截图（可通过 Android Studio 模拟器或真实设备）
   - 保存为 JPG 格式
   - 与 Mockup 文件名一一对应（如 `1.jpg` 对应 `1.png`）
   - 放置在 `implementation/` 目录下

3. **创建 process.json**：
   - 在 `implementation/` 目录下创建 `process.json` 文件
   - 按顺序记录每个屏幕的信息
   - 包含 `mock_actions`（设计稿中的动作描述，自然语言）
   - 包含 `actions`（真实实现中的动作，包含类型和边界框）

4. **数据质量要求**：
   - 截图分辨率建议：宽度 1080px（脚本会自动调整）
   - Mockup 和实现截图应保持相同的屏幕尺寸比例
   - 确保截图清晰，组件边界明确

**示例目录结构**：
```
my_case_study/
  my_app/
    process_1/
      implementation/
        process.json
        1.jpg
        2.jpg
      mockup/
        1.png
        2.png
```

## 运行方式

### 完整评估（推荐用于论文复现）

#### 使用真实数据（启用 VLM Agent）

```bash
conda activate guipilot  # 或 guipilot-gpu

# 设置 OpenAI API 密钥（必需，用于 VLM Agent）
export OPENAI_KEY=your_api_key_here

# 运行完整评估
python experiments/rq4_case_study/main.py \
  --dataset /path/to/case-study \
  --output-dir runs/rq4/full
```

**脚本会执行**：
- 遍历所有应用和流程
- 对比每个屏幕的 Mockup 和实现
- 检测界面不一致性
- **使用 VLM Agent 验证动作的正确性**（将设计稿中的自然语言描述转换为可执行动作并验证）
- 生成详细的报告和可视化结果

**VLM Agent 工作流程**：
1. 读取设计稿中的动作描述（`mock_actions`，如 "点击登录按钮"）
2. 分析当前实现屏幕的截图
3. 生成可执行动作（如 `click(bounds=[100, 200, 300, 260])`）
4. 验证生成的动作是否与真实实现中的动作匹配（类型、边界框重叠等）
5. 如果验证失败，自动重试（最多 `--max-agent-trials` 次）

#### 使用真实数据（跳过 VLM Agent）

如果只想进行界面一致性检测，不需要动作验证：

```bash
conda activate guipilot
python experiments/rq4_case_study/main.py \
  --dataset /path/to/case-study \
  --skip-agent \
  --output-dir runs/rq4/no-agent
```

这种方式会跳过 VLM Agent 阶段，只进行界面一致性检测，无需 API 密钥。

### 限制评估范围

```bash
python experiments/rq4_case_study/main.py \
  --dataset /path/to/case-study \
  --process-limit 2 \
  --screen-limit 3 \
  --output-dir runs/rq4/partial
```

### 快速验证（推荐用于 CI 烟测）

```bash
python experiments/rq4_case_study/main.py \
  --use-demo-data \
  --skip-agent \
  --skip-visualize \
  --output-dir ci-artifacts/rq4
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `DATASET_PATH` 环境变量 | 数据集根目录路径，包含应用和流程目录 |
| `--output-dir` | str | `output` | 输出目录，用于存储报告、图片和汇总 CSV |
| `--process-limit` | int | `None`（处理全部） | 限制评估的流程数量 |
| `--screen-limit` | int | `None`（处理全部） | 限制每个流程评估的屏幕数量 |
| `--skip-agent` | flag | `False` | 跳过 VLM agent 行为推理（CI 必备，可加快执行速度） |
| `--max-agent-trials` | int | `3` | 启用 agent 时的最大重试次数 |
| `--skip-visualize` | flag | `False` | 关闭可视化图片输出，减少运行时开销 |
| `--include-branches` | flag | `False` | 包含分支屏幕（文件名包含 `branch` 的屏幕），默认会忽略 |

### 参数使用建议

**完整评估（启用 VLM Agent）**：
```bash
export OPENAI_KEY=your_api_key_here
python experiments/rq4_case_study/main.py \
  --dataset /path/to/case-study \
  --output-dir runs/rq4/full
# 需要设置 OPENAI_KEY 环境变量，Agent 会验证动作正确性
```

**完整评估（跳过 VLM Agent）**：
```bash
python experiments/rq4_case_study/main.py \
  --dataset /path/to/case-study \
  --skip-agent \
  --output-dir runs/rq4/no-agent
# 只进行界面一致性检测，无需 API 密钥
```

**部分评估**：
```bash
--dataset /path/to/case-study --process-limit 2 --screen-limit 3
# 只评估前 2 个流程，每个流程最多 3 个屏幕
```

**CI 烟测**：
```bash
--use-demo-data --skip-agent --allow-empty --skip-visualize --output-dir ci-artifacts/rq4
# 使用演示数据，跳过 agent，允许空结果，跳过可视化
```

**快速验证**：
```bash
--use-demo-data --skip-agent --output-dir runs/rq4/demo
# 使用演示数据，跳过 agent，生成可视化
```

## 输出内容

### 1. 汇总结果 CSV（`summary.csv`）

位于输出目录根目录，包含每个屏幕的评估结果：

| 列名 | 说明 |
|------|------|
| `process` | 流程标识 |
| `screen` | 屏幕文件名 |
| `match_score` | 匹配得分（0-1） |
| `inconsistencies` | 检测到的不一致数量 |
| `action_verified` | 动作是否验证通过（如未跳过 agent） |
| `agent_retries` | Agent 重试次数（如未跳过 agent） |

**示例输出**：

```csv
process,screen,match_score,inconsistencies,action_verified,agent_retries
demo_app/process_1,1.jpg,0.95,2,True,0
demo_app/process_1,2.jpg,0.88,1,True,1
```

### 2. 详细报告 JSON（`process_*/screen_*/report.json`）

每个屏幕都会在输出目录下创建详细的报告文件：

**目录结构**：

```
output-dir/
  summary.csv
  demo_app/
    process_1/
      screen_1/
        report.json          # 详细检测/匹配结果
        matched.jpg          # 匹配可视化（如未跳过可视化）
        inconsistencies.jpg  # 不一致性标注（如未跳过可视化）
      screen_2/
        report.json
        ...
```

**`report.json` 内容**：

```json
{
  "screen": "1.jpg",
  "match_score": 0.95,
  "matched_pairs": [
    {"widget1": 0, "widget2": 1, "score": 0.98},
    ...
  ],
  "inconsistencies": [
    {"widget1": 2, "widget2": 3, "type": "color"},
    ...
  ],
  "action_attempts": [
    {
      "action": "click",
      "bounds": [100, 200, 300, 260],
      "success": true,
      "retries": 0
    }
  ]
}
```

### 3. 可视化图片（`process_*/screen_*/*.jpg`）

仅在未使用 `--skip-visualize` 时生成：

- `matched.jpg`：显示 Mockup 和实现的对比，标注匹配的组件对
- `inconsistencies.jpg`：高亮检测到的不一致性
- 其他辅助图片（如动作标注等）

### 4. Agent 分析结果（如启用 VLM Agent）

当启用 VLM Agent 时，会生成额外的分析结果：

- `image_agent.jpg`：Agent 标注的屏幕截图，显示 Agent 识别的组件和生成的动作
- `report.json` 中的 `action_trials` 字段：记录所有 Agent 尝试的动作，包括失败的尝试

**查看 Agent 结果示例**：
```json
{
  "action_correct": true,
  "action_trials": [
    "click([100, 200, 300, 260])",
    "click([105, 205, 305, 265])"
  ]
}
```

### 5. 控制台输出

脚本运行时会实时输出每个流程和屏幕的评估进度：

**不使用 Agent 时**：
```
[RQ4] Evaluating 2 processes
  - demo_app/process_1: 2 screens
    Screen 1/2: 1.jpg (match_score=0.95, inconsistencies=2)
    Screen 2/2: 2.jpg (match_score=0.88, inconsistencies=1)
  - demo_app/process_2: 1 screens
    Screen 1/1: 1.jpg (match_score=0.92, inconsistencies=0)
[RQ4] Summary written to runs/rq4/full/summary.csv
```

**使用 Agent 时**（会显示动作验证结果）：
```
[RQ4] Evaluating 2 processes
  - demo_app/process_1: 2 screens
    Screen 1/2: 1.jpg (match_score=0.95, inconsistencies=2, action_verified=True)
    Screen 2/2: 2.jpg (match_score=0.88, inconsistencies=1, action_verified=True, retries=1)
  - demo_app/process_2: 1 screens
    Screen 1/1: 1.jpg (match_score=0.92, inconsistencies=0, action_verified=True)
[RQ4] Summary written to runs/rq4/full/summary.csv
```

## 环境变量配置

### VLM Agent 配置

**必需（启用 Agent 时）**：
- `OPENAI_KEY`：OpenAI API 密钥（仅在未使用 `--skip-agent` 时需要）

**设置方法**：
```bash
# 方法1：临时设置（当前终端会话有效）
export OPENAI_KEY=sk-your-api-key-here

# 方法2：永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export OPENAI_KEY=sk-your-api-key-here' >> ~/.zshrc
source ~/.zshrc

# 方法3：使用 .env 文件（推荐，项目根目录）
echo "OPENAI_KEY=sk-your-api-key-here" >> .env
# 脚本会自动加载 .env 文件
```

**获取 API 密钥**：
1. 访问 [OpenAI API Keys](https://platform.openai.com/api-keys)
2. 登录或注册账号
3. 创建新的 API 密钥
4. 复制密钥并设置到环境变量

### 检测器和 OCR 配置

- `ENABLE_LOCAL_DETECTOR=1`：启用本地 YOLOv8 检测器
- `DETECTOR_WEIGHT_PATH`：自定义检测器权重路径
- `ENABLE_PADDLEOCR=1`：启用本地 PaddleOCR（推荐启用，用于文本识别）
- `PADDLEOCR_USE_GPU=1`：尝试使用 GPU 加速 OCR
- `DETECTOR_SERVICE_URL`：使用远程检测服务
- `OCR_SERVICE_URL`：使用远程 OCR 服务

详细说明请参考 `docs/environment-setup.md`。

## 注意事项

1. **文件对应关系**：Mockup 和实现的截图文件名必须一一对应（如 `1.png` 对应 `1.jpg`）
2. **分支屏幕**：默认会忽略文件名包含 `branch` 的屏幕，使用 `--include-branches` 可强制包含
3. **VLM Agent**：
   - 如未设置 `OPENAI_KEY` 且未使用 `--skip-agent`，脚本会报错退出
   - Agent 调用会产生 API 费用，建议先用 `--skip-agent` 测试数据格式
   - Agent 重试会增加 API 调用次数，可通过 `--max-agent-trials` 控制
4. **空结果处理**：如果筛选后没有可评估的屏幕，默认会报错；使用 `--allow-empty` 可允许空结果
5. **演示数据**：使用 `--use-demo-data` 时，会在输出目录下创建 `_demo_dataset` 目录
6. **动作验证**：只有在未使用 `--skip-agent` 时才会验证动作的正确性
7. **重试机制**：Agent 失败时会自动重试，最多重试 `--max-agent-trials` 次（默认 3 次）
8. **数据质量**：确保 Mockup 和实现截图清晰，组件边界明确，否则可能影响检测和匹配效果
9. **OCR 依赖**：建议启用 `ENABLE_PADDLEOCR=1`，用于提取组件文本信息，提高检测准确性

## 与论文实验的关系

本实验复现论文中的案例研究部分：

- **界面一致性检查**：对比 Mockup 和实现的界面，检测不一致性
- **动作验证**：验证真实实现中的动作是否符合设计稿的描述
- **恢复机制**：使用 VLM agent 在发现不一致时尝试恢复流程（如未跳过 agent）

## CI 烟测示例

CI 工作流使用的命令：

```bash
python experiments/rq4_case_study/main.py \
  --use-demo-data \
  --skip-agent \
  --allow-empty \
  --skip-visualize \
  --output-dir ci-artifacts/rq4
```

该命令会：
- 使用内置演示数据（无需外部下载）
- 跳过 VLM agent 阶段（无需 API 密钥）
- 允许空结果（不会因无可评估屏幕而失败）
- 跳过可视化生成（加快执行速度）
- 输出到 CI artifacts 目录

适用于验证脚本依赖和基础逻辑是否正确。
