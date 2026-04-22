import ctypes
import os


def _preload_cuda_libs() -> None:
    """Pre-load NVIDIA CUDA shared libraries from the active venv.

    torch and paddle expect CUDA libs on LD_LIBRARY_PATH, but when installed
    as Python packages (nvidia-cudnn-cu12, etc.) they live inside site-packages
    and are not on that path by default.  Loading them here with RTLD_GLOBAL
    puts their symbols into the process's global namespace so that subsequent
    dlopen calls (inside torch/paddle C extensions) resolve without any
    LD_LIBRARY_PATH configuration from the user.

    This runs once at import time, before any submodule that pulls in torch or
    paddle, so it covers all entry points automatically.
    """
    site_packages = os.path.join(os.path.dirname(__file__), "..", "site-packages")
    site_packages = os.path.normpath(site_packages)
    if not os.path.isdir(site_packages):
        # Fall back to locating site-packages via sysconfig.
        import sysconfig
        site_packages = sysconfig.get_path("purelib")

    nvidia_base = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_base):
        return  # not a GPU venv, nothing to do

    # Load in dependency order so each library can find its own prerequisites
    # when the dynamic linker resolves transitive deps.
    ordered = [
        "cuda_runtime", "cuda_nvrtc", "nvjitlink",
        "cublas", "curand", "cufft", "cusparse", "cusolver",
        "nccl", "cudnn", "cuda_cupti", "cufile", "nvtx", "cuda_cccl",
    ]
    remaining = sorted(p for p in os.listdir(nvidia_base) if p not in ordered)

    seen: set[str] = set()

    def _load_dir(lib_dir: str) -> None:
        if not os.path.isdir(lib_dir):
            return
        for name in sorted(os.listdir(lib_dir)):
            if ".so." not in name:  # versioned libs only, skip plain .so symlinks
                continue
            path = os.path.join(lib_dir, name)
            if path in seen:
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                seen.add(path)
            except OSError:
                pass

    for pkg in ordered + remaining:
        _load_dir(os.path.join(nvidia_base, pkg, "lib"))

    # Paddle bundles its own CUDA-adjacent libs (flash-attention, etc.)
    _load_dir(os.path.join(site_packages, "paddle", "libs"))


_preload_cuda_libs()
