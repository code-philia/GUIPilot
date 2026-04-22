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

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- NVIDIA GPU with CUDA 12.6-compatible driver (driver ≥ 525.85)

### Install

```bash
git clone https://github.com/code-philia/GUIPilot.git
cd GUIPilot
uv sync
```

`uv sync` creates `.venv/` and installs all dependencies, including GPU-enabled builds of PyTorch (cu126) and PaddlePaddle (cu126) from their respective wheel indexes configured in `pyproject.toml`.

GPU setup is fully automatic. `guipilot/__init__.py` pre-loads all NVIDIA CUDA shared libraries from the venv at import time, so torch and paddle find them without any `LD_LIBRARY_PATH` configuration.

### Setup Experiments

Each directory within `/experiments` includes a `README.md` file with instructions for setting up datasets and running the experiment.

## 🏃 Usage

### Quick start — MVP check on your own image pairs

1. Place your mock/real screenshot pairs in `output_images/`, named `<name>_mock.<ext>` and `<name>_real.<ext>` (e.g. `login_mock.jpg` / `login_real.jpg`).

2. Run:
    ```bash
    uv run run_checks.py --input output_images/ --output results/
    ```

3. Results are written to `<output>/`:
    - `results.csv` — per-pair inconsistency table
    - `visualizations/<name>.jpg` — side-by-side annotated images (mock | real)
        - **Green** boxes: matched widget pairs with no inconsistency
        - **Yellow** boxes: matched pairs with a detected inconsistency
        - **Red** boxes: unmatched widgets (missing in real / excess in real)

The YOLO widget detector weights are downloaded automatically from HuggingFace on first run.

The OCR language defaults to `ch` (Chinese & English). To change it, set `OCR_LANG` in `.env` or the shell before running. See PaddleOCR documentation on supported languages.

### API usage

Refer to [`/experiments/rq1_screen_inconsistency/main.py`](experiments/rq1_screen_inconsistency/main.py) for a complete working example.

#### Step 1: Load screenshots as `Screen` instances

Each `Screen` instance requires an RGB screenshot (`numpy.ndarray`). Widgets can be loaded from a JSON annotation file or auto-detected.

**Option A — auto-detect with GUIPilot’s built-in models:**

```python
import cv2
from guipilot.entities import Screen

screenA = Screen(cv2.imread(pathA))
screenB = Screen(cv2.imread(pathB))

screenA.detect(); screenA.ocr()
screenB.detect(); screenB.ocr()
```

**Option B — load widgets from a JSON annotation file:**

```python
import cv2, json
from guipilot.entities import Bbox, Widget, WidgetType, Screen

def load_widgets(path):
    raw = json.load(open(path, encoding="utf-8"))
    return {
        i: Widget(type=WidgetType(item["type"]), bbox=Bbox(*item["bbox"]))
        for i, item in enumerate(raw)
    }

screenA = Screen(cv2.imread(pathA), load_widgets(widgetsA_path))
screenB = Screen(cv2.imread(pathB), load_widgets(widgetsB_path))
```

#### Step 2: Match widgets and check consistency

```python
from guipilot.matcher import GUIPilotV2 as Matcher
from guipilot.checker import GVT as Checker

pairs, _, match_time = Matcher().match(screenA, screenB)
inconsistencies, check_time = Checker().check(screenA, screenB, pairs)
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
