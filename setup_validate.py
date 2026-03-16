"""Auto-detect and prepare missing CatVTON worker requirements.

This script is designed to run in Docker build/runtime to:
1) validate required Python packages/imports
2) validate required CatVTON files and AutoMasker checkpoints
3) pre-download missing model repos only when absent
4) print a structured report for diagnostics
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

CATVTON_SRC = Path(os.environ.get("CATVTON_SRC", "/app/CatVTON"))
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "booksforcharlie/stable-diffusion-inpainting")
ATTN_REPO = os.environ.get("ATTN_REPO", "zhengchong/CatVTON")
VAE_REPO = os.environ.get("VAE_REPO", "stabilityai/sd-vae-ft-mse")

REQUIRED_IMPORTS = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "scipy",
    "skimage",
    "tqdm",
    "matplotlib",
    "fvcore",
    "iopath",
    "omegaconf",
    "pycocotools",
]

REQUIRED_FILES = [
    CATVTON_SRC / "model" / "pipeline.py",
    CATVTON_SRC / "model" / "cloth_masker.py",
    CATVTON_SRC / "model" / "DensePose" / "__init__.py",
    CATVTON_SRC / "model" / "SCHP" / "__init__.py",
]


def _check_imports() -> tuple[list[str], list[str]]:
    ok: list[str] = []
    missing: list[str] = []
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
            ok.append(mod)
        except Exception:
            missing.append(mod)
    return ok, missing


def _check_files() -> tuple[list[str], list[str]]:
    ok: list[str] = []
    missing: list[str] = []
    for f in REQUIRED_FILES:
        if f.exists():
            ok.append(str(f))
        else:
            missing.append(str(f))
    return ok, missing


def _ensure_repo(repo_id: str) -> str:
    # Use HF cache; snapshot_download reuses existing cache and avoids re-fetch.
    return snapshot_download(repo_id=repo_id)


def main() -> int:
    report: dict[str, object] = {
        "catvton_src": str(CATVTON_SRC),
        "imports_ok": [],
        "imports_missing": [],
        "files_ok": [],
        "files_missing": [],
        "repos": {},
        "automasker_check": {},
        "unresolved": [],
    }

    imports_ok, imports_missing = _check_imports()
    files_ok, files_missing = _check_files()
    report["imports_ok"] = imports_ok
    report["imports_missing"] = imports_missing
    report["files_ok"] = files_ok
    report["files_missing"] = files_missing
    if imports_missing:
        report["unresolved"].append("missing_python_imports")
    if files_missing:
        report["unresolved"].append("missing_catvton_files")

    repos: dict[str, str] = {}
    for repo in [BASE_MODEL_PATH, VAE_REPO, ATTN_REPO]:
        try:
            repos[repo] = _ensure_repo(repo)
        except Exception as exc:
            repos[repo] = f"ERROR: {exc}"
            report["unresolved"].append(f"repo_download_failed:{repo}")
    report["repos"] = repos

    # Validate AutoMasker assets in ATTN_REPO snapshot.
    auto_info: dict[str, object] = {}
    attn_path = repos.get(ATTN_REPO, "")
    if isinstance(attn_path, str) and not attn_path.startswith("ERROR:"):
        densepose = Path(attn_path) / "DensePose"
        schp = Path(attn_path) / "SCHP"
        auto_info["densepose_path"] = str(densepose)
        auto_info["schp_path"] = str(schp)
        auto_info["densepose_exists"] = densepose.exists()
        auto_info["schp_exists"] = schp.exists()
        if not densepose.exists() or not schp.exists():
            report["unresolved"].append("automasker_ckpt_missing")
    else:
        report["unresolved"].append("attn_repo_unavailable")
    report["automasker_check"] = auto_info

    # Informational note: full AutoMasker runtime (DensePose forward) requires CUDA runtime.
    report["automasker_runtime_note"] = "AutoMasker full init/inference is validated at runtime on CUDA worker."

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
