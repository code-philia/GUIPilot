# Contributing to GUIPilot

Thank you for your interest in contributing to GUIPilot. This document explains how to report issues, contribute code, and run core experiments.

## Quick start

1. Create and activate the development environment:
   - Using Conda:
     ```bash
     conda env create -f environment.yml
     conda activate guipilot
     ```
2. Install the package locally:
   ```bash
   pip install .
   ```
3. Recommended entry points:
   - Screen inconsistency experiment: experiments/rq1_screen_inconsistency/main.py
   - Flow inconsistency experiment: experiments/rq2_flow_inconsistency/main.py
   - Case studies: experiments/rq4_case_study/main.py

## Project structure & key components

- Screen entity and OCR/detection: guipilot/entities/screen.py (Screen)
- Checker examples (e.g., GVT): guipilot/checker/gvt.py
- Widget matcher interface: guipilot/matcher/matcher.py
- Automation & recording tools used by flow experiments: experiments/rq2_flow_inconsistency/actions/

When adding or changing features, prefer to extend or reuse these modules.

## Branching and pull requests

1. Fork the repository and create a branch: feature/your-feature or fix/issue-id.
2. Keep commits focused and atomic. Write clear commit messages.
3. In your PR description include:
   - Purpose and summary of changes
   - Main files or classes modified
   - Reproduction steps or commands for experiments if applicable
4. Maintainters will review PRs and may request changes or tests.

## Reporting issues

- Bug reports should include reproduction steps, environment, and error logs.
- Feature requests should describe motivation, proposed design, and alternatives.
- For experiment issues, include dataset paths and the exact command used.

## Reproducing experiments & data

- Experiment scripts may rely on environment variables or dataset paths; check each experiment folder for README or .env.example.
- Ensure dependencies are installed and datasets are prepared before running experiments.

## Code style and tests

- Use Black and isort for formatting. Consider flake8 for linting.
- Add unit tests or reproducible experiment scripts for significant changes.

## Community & conduct

Be respectful and professional in discussions. This project uses the MIT license — see LICENSE for details.

Thank you for helping improve GUIPilot!
