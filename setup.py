import re
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent


def load_long_description() -> str:
    readme = HERE / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


def parse_requirements_txt(path: Path):
    reqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # ignore -e, git+ urls are preserved
        if line.startswith("-e"):
            continue
        reqs.append(line)
    return reqs


def _is_conda_system_entry(s: str) -> bool:
    s_low = s.lower()
    # common markers of conda/system entries or platform libs
    if s_low.startswith("python") or s_low.startswith("_"):
        return True
    if any(
        tok in s_low
        for tok in (
            "linux",
            "linux-64",
            "win",
            "osx",
            "cuda",
            "cudnn",
            "nvidia",
            "glibc",
        )
    ):
        return True
    return False


def load_install_requires():
    req_file = HERE / "requirements.txt"
    if req_file.exists():
        return parse_requirements_txt(req_file)

    # Fallback common dependencies (conservative)
    fallback_deps = [
        "numpy",
        "opencv-python",
        "Pillow",
        "python-dotenv",
        "pydantic",
        "ultralytics",
        "supervision",
        "PyYAML",
        "requests",
        "scikit-learn",
        "matplotlib",
        "openai",
        "paddleocr",
    ]

    env_file = HERE / "environment.yml"
    if env_file.exists():
        try:
            # Delay import of yaml so setup import won't fail if PyYAML missing in some contexts
            try:
                import yaml  # type: ignore
            except Exception:
                yaml = None

            if yaml is None:
                # If PyYAML not available in build environment, return fallback
                return fallback_deps

            data = yaml.safe_load(env_file.read_text(encoding="utf-8"))
            deps = data.get("dependencies", []) or []
            install = []
            for d in deps:
                if isinstance(d, str):
                    d = d.strip()
                    if not d or d.startswith("#"):
                        continue
                    # Skip conda/system entries like 'python=3.12.4' or platform libs
                    if _is_conda_system_entry(d):
                        continue
                    # convert single '=' (conda style) to pip-compatible '>=' as a fallback
                    if re.match(r"^[^=]+=[^=]+$", d) and ":" not in d:
                        install.append(re.sub(r"=+", ">=", d, count=1))
                    else:
                        install.append(d)
                elif isinstance(d, dict) and "pip" in d:
                    for p in d["pip"]:
                        p = p.strip()
                        if not p or p.startswith("#") or p.startswith("-e"):
                            continue
                        install.append(p)
            return install
        except Exception:
            pass

    return fallback_deps


setup(
    name="guipilot",
    version="0.1.0",
    description="GUIPilot: Consistency-based Mobile GUI Testing",
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/guipilot",
    author="GUIPilot Authors",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=load_install_requires(),
    extras_require={
        "dev": ["black", "isort", "flake8", "pre-commit"],
        "test": ["pytest", "pytest-cov", "coverage"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
