"""
scripts/package_release.py
Automated packaging script for CUDA GPU Acceleration Engine (v1.0.0-rc1).
Builds a clean, reproducible lightweight source distribution zip archive
strictly excluding large binary model weights, virtual environments, node_modules,
caches, and temporary files.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dist"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1.0.0-rc4"
ARCHIVE_NAME = f"cuda-acceleration-engine-{VERSION}-source.zip"
ARCHIVE_PATH = OUTPUT_DIR / ARCHIVE_NAME

# Directories and files to include
INCLUDE_DIRS = [
    "backend",
    "engine",
    "custom_kernels",
    "src",
    "include",
    "config",
    "configs",
    "results",
    "tests",
    "scripts",
]

INCLUDE_ROOT_FILES = [
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
    "MODEL_CARD.md",
    "README.md",
    "requirements.txt",
    "environment.yml",
    "start_platform.bat",
    "start_platform.ps1",
    ".gitignore",
    "pytest.ini",
    "CMakeLists.txt",
]

# Patterns or directory names to strictly exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".venv*",
    "node_modules",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.obj",
    "*.o",
    "*.pdb",
    "*.idb",
    "*.log",
    "*.tmp",
    "*.bin",
    ".git",
    ".vs",
]


def should_exclude(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    # Preserve frontend production distribution assets
    if len(parts) >= 2 and parts[0] == "frontend" and parts[1] == "dist":
        if any(rel_path.lower().endswith(ext) for ext in (".tmp", ".log", ".map")):
            return True
        return False
    for p in parts:
        if p.startswith(".venv") or p in ("__pycache__", ".pytest_cache", "node_modules", "dist", "build", ".vs", "scratch"):
            return True
        if p == ".digest_cache.json":
            return True
    lower = rel_path.lower()
    if any(lower.endswith(ext) for ext in (".pyc", ".pyo", ".pyd", ".obj", ".o", ".pdb", ".tmp", ".log")):
        return True
    return False


def build_release_package():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        raise FileExistsError(f"Target release candidate {ARCHIVE_PATH} already exists. Never overwrite an existing release candidate.")

    print(f"[*] Packaging release archive: {ARCHIVE_PATH}")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 1. Root files
        for rf in INCLUDE_ROOT_FILES:
            fp = PROJECT_ROOT / rf
            if fp.exists():
                arcname = f"cuda-acceleration-engine/{rf}"
                zipf.write(fp, arcname=arcname)
                file_count += 1
                total_uncompressed_bytes += fp.stat().st_size

        # 2. Source directories
        for d in INCLUDE_DIRS:
            dp = PROJECT_ROOT / d
            if dp.exists():
                for root, _, files in os.walk(dp):
                    for file in files:
                        fp = Path(root) / file
                        rel = fp.relative_to(PROJECT_ROOT).as_posix()
                        if not should_exclude(rel):
                            arcname = f"cuda-acceleration-engine/{rel}"
                            zipf.write(fp, arcname=arcname)
                            file_count += 1
                            total_uncompressed_bytes += fp.stat().st_size

        # 3. Frontend sources (excluding node_modules and dist)
        frontend_dir = PROJECT_ROOT / "frontend"
        if frontend_dir.exists():
            for root, _, files in os.walk(frontend_dir):
                for file in files:
                    fp = Path(root) / file
                    rel = fp.relative_to(PROJECT_ROOT).as_posix()
                    if not should_exclude(rel):
                        arcname = f"cuda-acceleration-engine/{rel}"
                        zipf.write(fp, arcname=arcname)
                        file_count += 1
                        total_uncompressed_bytes += fp.stat().st_size

        # 4. Checkpoints documentation, manifests & lightweight seed fixture (NO heavy weights)
        ckpt_meta_files = [
            "checkpoints/MODEL_MANIFEST.json",
            "checkpoints/README.md",
            "checkpoints/.gitkeep",
            "checkpoints/test_checkpoint.pt",
        ]
        for cm in ckpt_meta_files:
            fp = PROJECT_ROOT / cm
            if fp.exists():
                arcname = f"cuda-acceleration-engine/{cm}"
                zipf.write(fp, arcname=arcname)
                file_count += 1
                total_uncompressed_bytes += fp.stat().st_size

        # 5. Lightweight seed datasets & manifests
        seed_data_files = [
            "data/text/tiny_shakespeare.txt",
            "data/text/README.md",
            "data/text/expanded_pretraining/dataset_manifest.json",
            "data/text/production_10m/dataset_manifest.json",
            "data/text/pretraining_campaign/dataset_manifest.json",
        ]
        for sd in seed_data_files:
            fp = PROJECT_ROOT / sd
            if fp.exists():
                arcname = f"cuda-acceleration-engine/{sd}"
                zipf.write(fp, arcname=arcname)
                file_count += 1
                total_uncompressed_bytes += fp.stat().st_size

    archive_size_bytes = ARCHIVE_PATH.stat().st_size
    archive_size_mb = archive_size_bytes / (1024 * 1024)
    uncomp_mb = total_uncompressed_bytes / (1024 * 1024)

    # Compute SHA-256 of created archive
    hasher = hashlib.sha256()
    with open(ARCHIVE_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    archive_sha256 = hasher.hexdigest().upper()

    print(f"[+] Successfully created release archive:")
    print(f"    Path:            {ARCHIVE_PATH}")
    print(f"    Files Packaged:  {file_count}")
    print(f"    Uncompressed:    {uncomp_mb:.2f} MB")
    print(f"    Compressed Size: {archive_size_mb:.2f} MB ({archive_size_bytes} bytes)")
    print(f"    SHA-256:         {archive_sha256}")

    # Write checksum manifest
    checksum_file = OUTPUT_DIR / f"cuda-acceleration-engine-{VERSION}-checksums.sha256"
    checksum_file.write_text(f"{archive_sha256}  {ARCHIVE_NAME}\n", encoding="utf-8")
    print(f"[+] Checksum manifest written to: {checksum_file}")


if __name__ == "__main__":
    build_release_package()

