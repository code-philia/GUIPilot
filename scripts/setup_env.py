#!/usr/bin/env python3
"""Helper script to create or update the GUIPilot conda environment(s).

Usage examples:
    python scripts/setup_env.py              # auto-detect platform and create env
    python scripts/setup_env.py --platform macos
    python scripts/setup_env.py --platform linux-gpu --update
    python scripts/setup_env.py --platform windows --name guipilot-win
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = {
    "macos": ROOT / "envs" / "environment-macos.yml",
    "linux-gpu": ROOT / "envs" / "environment-linux-gpu.yml",
    "windows": ROOT / "envs" / "environment-windows.yml",
}
ENV_NAMES = {
    "macos": "guipilot",
    "linux-gpu": "guipilot-gpu",
    "windows": "guipilot",
}
PIP_REQUIREMENTS = {
    "macos": [ROOT / "requirements-pip.txt"],
    "windows": [ROOT / "requirements-pip-windows.txt"],
    "linux-gpu": [
        ROOT / "requirements-pip.txt",
        ROOT / "requirements-pip-gpu.txt",
    ],
}


def env_exists(conda_exe: str, env_name: str) -> bool:
    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False

    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False

    for path in data.get("envs", []):
        try:
            if Path(path).name == env_name:
                return True
        except Exception:
            continue
    return False


def detect_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    if system == "Linux":
        return "linux-gpu"
    raise RuntimeError(f"Unsupported operating system: {system}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update GUIPilot conda environment.")
    parser.add_argument(
        "--platform",
        choices=sorted(ENV_FILES),
        help="Target platform profile. Defaults to current OS.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Use 'conda env update' instead of 'conda env create'.",
    )
    parser.add_argument(
        "--name",
        help="Override the conda environment name.",
    )
    parser.add_argument(
        "--conda",
        default="conda",
        help="Conda executable to invoke (defaults to 'conda').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.platform or detect_platform()
    env_file = ENV_FILES[target]

    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    conda_candidate = os.environ.get("CONDA_EXE") or shutil.which(args.conda)
    if not conda_candidate:
        raise FileNotFoundError(
            "Conda executable not found. Ensure Conda is installed and available on PATH, "
            "or provide --conda with an absolute path."
        )
    conda_exe = conda_candidate

    env_name = args.name or ENV_NAMES[target]
    should_update = args.update
    if not should_update and env_exists(conda_exe, env_name):
        print(f"[setup_env] Environment '{env_name}' already exists, switching to 'conda env update'.")
        should_update = True

    command = [conda_exe, "env", "update" if should_update else "create", "--file", str(env_file)]
    command.extend(["--name", env_name])

    print(f"[setup_env] Running: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("[setup_env] Conda command failed.", file=sys.stderr)
        return result.returncode

    action = "updated" if should_update else "created"
    print(f"[setup_env] Successfully {action} environment using {env_file.name}.")

    pip_files = PIP_REQUIREMENTS[target]
    for req_file in pip_files:
        if not req_file.exists():
            print(f"[setup_env] Warning: requirements file not found, skipping: {req_file}", file=sys.stderr)
            continue
        pip_cmd = [
            conda_exe,
            "run",
            "--name",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            str(req_file),
        ]
        print(f"[setup_env] Running: {' '.join(pip_cmd)}")
        pip_result = subprocess.run(pip_cmd, check=False)
        if pip_result.returncode != 0:
            print(f"[setup_env] Pip install failed for {req_file.name}.", file=sys.stderr)
            return pip_result.returncode

    print(f"[setup_env] Pip requirements installed into '{env_name}'.")

    # Install guipilot package itself
    setup_py = ROOT / "setup.py"
    if setup_py.exists():
        install_cmd = [
            conda_exe,
            "run",
            "--name",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "-e",
            str(ROOT),
        ]
        print(f"[setup_env] Installing guipilot package: {' '.join(install_cmd)}")
        install_result = subprocess.run(install_cmd, check=False)
        if install_result.returncode != 0:
            print(f"[setup_env] Warning: Failed to install guipilot package.", file=sys.stderr)
            print(f"[setup_env] You can manually install it later with: pip install -e .", file=sys.stderr)
        else:
            print(f"[setup_env] Successfully installed guipilot package.")
    else:
        print(f"[setup_env] Warning: setup.py not found, skipping guipilot package installation.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

