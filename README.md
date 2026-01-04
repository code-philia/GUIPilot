# GUIPilot

[![arXiv](https://img.shields.io/badge/Paper-green)](http://linyun.info/publications/issta25.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://sites.google.com/view/guipilot/home)] [[Datasets](https://zenodo.org/records/15107436)] [[Models](https://huggingface.co/code-philia/GUIPilot)]

This is the official repository for the paper *"GUIPilot: A Consistency-based Mobile GUI Testing Approach for Detecting Application-specific Bugs"*, published at ISSTA 2025.

**GUIPilot** detects inconsistencies between mobile app designs and their implementations. It addresses two main types of inconsistencies: screen and process inconsistencies, using a combination of widget alignment and vision-language models. We’re continuously updating this repository. Stay tuned for more developments!

- Screen Inconsistency Detection:
    - Detects differences between the actual and expected UI appearance.
    - Converts the screen-matching problem into an optimizable widget alignment task.

- Process Inconsistency Detection:
    - Detects discrepancies between the actual and expected UI transitions after an action.
    - Translates natural language descriptions of transitions in mockups into stepwise actions (e.g., clicks, long-presses, text inputs).
    - Utilizes a vision-language model to infer actions on the real screen, ensuring that the expected transitions occur in the app.

## 📂 Structure

This repository contains three components:
1. The **core** module (`/guipilot`).
3. The **datasets** module (`/dataset`), which records the dataset repositories.
2. The **experiments** module (`/experiments`), which supports the research questions 1-4 as presented in the paper.

The core GUIPilot module is organized as follows:

- `/agent`: Handles the action completion using a Vision-Language Model (VLM) agent
- `/matcher`: Pairs widgets across two different screens for comparison
- `/checker`: Detects bounding box, color, and text inconsistencies between widget pairs
- `/entities`: Defines Process, Screen, Widget, and Inconsistency entities used throughout the module
- `/models`: Contains OCR and widget detection models

## ⚙️ Setup
### Setup GUIPilot

Clone the repository and follow the steps below:

运行统一脚本创建/更新 Conda 环境（脚本会自动安装所有依赖并安装 `guipilot` 包）：
```bash
python scripts/setup_env.py
```

- macOS/Windows 默认环境名为 `guipilot`
- Linux GPU 机器可以显式指定：`python scripts/setup_env.py --platform linux-gpu --name guipilot-gpu`
- 更多命令选项见 `docs/environment-setup.md`

脚本会自动完成以下步骤：
1. 创建/更新 Conda 环境（根据平台选择对应的 `envs/environment-*.yml`）
2. 安装 pip 依赖（`requirements-pip.txt`，Linux GPU 还会安装 `requirements-pip-gpu.txt`）
3. **安装 `guipilot` 包本身**（通过 `pip install -e .`，使用可编辑模式安装）

安装完成后，激活环境即可使用：
```bash
conda activate guipilot          # 或 guipilot-gpu
```

> **说明**：
> - `setup.py` 的作用是将 `guipilot` 包安装到 Python 环境中，使其可以在任何位置被导入（如 `from guipilot.matcher import ...`）
> - 实验脚本中只保留了必要的 `sys.path` 修改，用于导入实验目录下的本地模块（`utils`、`mutate`、`actions`）
> - `guipilot` 包已安装到环境中，不再需要通过修改 `sys.path` 来导入
> - 这种设计更符合 Python 包管理的最佳实践，也便于在其他项目中使用 GUIPilot

### Setup Experiments

Each directory within `/experiments` includes a `README.md` file that provides detailed instructions on setting up the environment, preparing datasets, and running the experiment.

## 🏃 Usage

更多实验脚本：

- 屏幕不一致性（RQ1）：`experiments/rq1_screen_inconsistency/main.py`
- 流程不一致性（RQ2）：`experiments/rq2_flow_inconsistency/main.py`
- 组件级评估（RQ3）：`experiments/rq3_component_wise_evaluation/main.py`
- 案例研究（RQ4）：`experiments/rq4_case_study/main.py`

- 默认情况下不启用离线模型。若需本地推理，可设置 `ENABLE_LOCAL_DETECTOR=1`（可配合 `DETECTOR_WEIGHT_PATH` 指定权重）与/或 `ENABLE_PADDLEOCR=1`；也可以通过 `DETECTOR_SERVICE_URL`、`OCR_SERVICE_URL` 使用远程服务，`PADDLEOCR_USE_GPU=1` 则尝试开启 GPU。

### Step 1: Load Screenshots as `Screen` Instances

Each `Screen` instance requires:

* an RGB screenshot (`numpy.ndarray`)
* a dictionary of widget ID → `Widget` instances (`dict[int, Widget]`)

You can either load widgets externally or use GUIPilot’s built-in widget detector.

#### Option 1: Load Widgets from JSON

```python
import cv2
import json
from guipilot.entities import Bbox, Widget, WidgetType, Screen

# Load screenshot images
screenA_image = cv2.imread(screenA_path)
screenB_image = cv2.imread(screenB_path)

# Load widgets from JSON file
# Example: [{"type": ..., "bbox": [xmin, ymin, xmax, ymax}, ...]
def load_widgets(path):
    raw = json.load(open(path, encoding="utf-8"))
    return {
        id: Widget(type=WidgetType(item["type"]), bbox=Bbox(*item["bbox"]))
        for id, item in enumerate(raw)
    }

screenA = Screen(screenA_image, load_widgets(widgetsA_path))
screenB = Screen(screenB_image, load_widgets(widgetsB_path))
```

#### Option 2: Auto-detect Widgets with GUIPilot

```python
screenA = Screen(screenA_image)
screenB = Screen(screenB_image)

# Automatically detect widgets and run OCR
screenA.detect()
screenA.ocr()
screenB.detect()
screenB.ocr()
```

---

### Step 2: Run Widget Matching and Consistency Checking

```python
from guipilot.matcher import GUIPilotV2 as Matcher
from guipilot.checker import GVT as Checker

matcher = Matcher()
checker = Checker()

# Match widgets between the two screens
pairs, _, match_time = matcher.match(screenA, screenB)

# Identify widget-level inconsistencies
y_pred, check_time = checker.check(screenA, screenB, pairs)
```

## 📚 Citation
If you find our work useful, please consider citing our work.
```
@article{liu2025guipilot,
  title={GUIPilot: A Consistency-Based Mobile GUI Testing Approach for Detecting Application-Specific Bugs},
  author={Liu, Ruofan and Teoh, Xiwen and Lin, Yun and Chen, Guanjie and Ren, Ruofei and Poshyvanyk, Denys and Dong, Jin Song},
  journal={Proceedings of the ACM on Software Engineering},
  volume={2},
  number={ISSTA},
  pages={753--776},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```
