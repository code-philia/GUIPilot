import os
import sys
import requests
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    _HAS_HF = True
except Exception:
    _HAS_HF = False


def _download_http(url: str, dest: str, chunk_size: int = 32768) -> None:
    """Download a file via HTTP(S) with a simple progress indicator."""
    print(f"Downloading from {url} -> {dest}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        if total is None:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        else:
            total = int(total)
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded * 100 / total
                        sys.stdout.write(f"\r{percent:.1f}%")
                        sys.stdout.flush()
            sys.stdout.write("\n")


def get_models_dir() -> str:
    """Return a default models directory: $XDG_CACHE_HOME/guipilot/models or ~/.cache/guipilot/models."""
    home = Path.home()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", str(home / ".cache"))
    models_dir = Path(xdg_cache) / "guipilot" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir)


def ensure_detector_weights(target_path: str = None, hf_repo: str = None, hf_filename: str = None,
                            hf_revision: str = "main", hf_token: str = None) -> str:
    """Download a model file from Hugging Face and save it to `target_path`.

    Behavior:
      - If `target_path` is provided, save the downloaded model there (create parents).
      - If not provided, will use env `GUIPILOT_DETECTOR_TARGET_PATH`, else default cache/best.pt.
      - `hf_repo` can be provided or set via `GUIPILOT_DETECTOR_HF_REPO`.
      - `hf_filename` can be provided or set via `GUIPILOT_DETECTOR_HF_FILE`. If omitted, the repo snapshot
        will be downloaded and the first file with extension .pt/.pth/.bin will be used.
    """

    # resolve target path priority: arg -> env -> default cache
    env_target = os.environ.get("GUIPILOT_DETECTOR_TARGET_PATH")
    if target_path:
        dest = Path(target_path)
    elif env_target:
        dest = Path(env_target)
    else:
        dest = Path(get_models_dir()) / "best.pt"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # if already exists and non-empty, return immediately
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    if not _HAS_HF:
        raise RuntimeError("huggingface_hub is required to download from Hugging Face. Install it: pip install huggingface_hub")

    # allow direct URL download via env var (keeps compatibility with .env which may contain a resolve URL)
    url = os.environ.get("GUIPILOT_DETECTOR_DOWNLOAD_URL") or os.environ.get("GUIPILOT_DETECTOR_URL")
    if url:
        try:
            _download_http(url, str(dest))
            return str(dest)
        except Exception as e:
            print("HTTP download failed:", e)

    hf_repo = hf_repo or os.environ.get("GUIPILOT_DETECTOR_HF_REPO")
    hf_filename = hf_filename or os.environ.get("GUIPILOT_DETECTOR_HF_FILE")
    hf_revision = hf_revision or os.environ.get("GUIPILOT_DETECTOR_HF_REV", "main")
    hf_token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not _HAS_HF:
        raise RuntimeError("huggingface_hub is required to download from Hugging Face. Install it: pip install huggingface_hub")

    if not hf_repo:
        raise ValueError("hf_repo must be provided via argument or GUIPILOT_DETECTOR_HF_REPO env var, or set GUIPILOT_DETECTOR_DOWNLOAD_URL to an HTTP URL")

    try:
        if hf_filename:
            print(f"Downloading {hf_repo}/{hf_filename} (rev={hf_revision}) from Hugging Face...")
            local_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, revision=hf_revision, token=hf_token)
            shutil.copy(local_path, str(dest))
            return str(dest)

        # snapshot entire repo and pick a likely model file
        print(f"Snapshotting Hugging Face repo {hf_repo} (rev={hf_revision})...")
        local_dir = snapshot_download(repo_id=hf_repo, revision=hf_revision, token=hf_token)
        found = None
        for p in Path(local_dir).rglob('*'):
            if p.suffix.lower() in ('.pt', '.pth', '.bin'):
                found = p
                break
        if not found:
            raise FileNotFoundError(f"No model file (.pt/.pth/.bin) found in Hugging Face repo snapshot {local_dir}")
        shutil.copy(str(found), str(dest))
        return str(dest)
    except Exception:
        # cleanup partial
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise

    else:
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    # if already exists and non-empty, return
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    # 1) huggingface
    hf_repo = hf_repo or os.environ.get("GUIPILOT_DETECTOR_HF_REPO")
    hf_filename = hf_filename or os.environ.get("GUIPILOT_DETECTOR_HF_FILE")
    hf_revision = hf_revision or os.environ.get("GUIPILOT_DETECTOR_HF_REV", "main")
    hf_token = hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if _HAS_HF and hf_repo and hf_filename:
        try:
            print(f"Downloading from Hugging Face repo {hf_repo}/{hf_filename}...")
            local_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename,
                                         revision=hf_revision, token=hf_token)
            # copy to dest
            shutil.copy(local_path, str(dest))
            return str(dest)
        except Exception as e:
            print("Hugging Face download failed:", e)

    # 2) direct URL
    url = url or os.environ.get("GUIPILOT_DETECTOR_DOWNLOAD_URL")
    if url:
        try:
            _download_http(url, str(dest))
            return str(dest)
        except Exception as e:
            print("HTTP download failed:", e)

    # 3) Google Drive
    gdrive_id = gdrive_id or os.environ.get("GUIPILOT_DETECTOR_GDRIVE_ID")
    if gdrive_id:
        try:
            _download_from_gdrive(gdrive_id, str(dest))
            return str(dest)
        except Exception as e:
            print("GDrive download failed:", e)

    # 4) fallback to package-local best.pt (legacy)
    base_path = Path(__file__).resolve().parent
    legacy = base_path / "detector" / "best.pt"
    if legacy.exists():
        try:
            shutil.copy(str(legacy), str(dest))
            return str(dest)
        except Exception:
            pass

    raise FileNotFoundError("Detector weights not found and download attempts failed")

