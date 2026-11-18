"""
tests/test_stage1_packaging_compliance.py
Stage 1 Automated Compliance and Verification Test Battery.
Verifies BLK-01 (Root LICENSE inclusion in release archive)
and BLK-04 (Root physical legal notice files NOTICE and THIRD_PARTY_LICENSES.md).
Conforms to NEXORA_RC3_PACKAGING_REMEDIATION_SPEC.md Section 5.
"""

import os
import hashlib
import zipfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
RC3_ARCHIVE_PATH = DIST_DIR / "cuda-acceleration-engine-v1.0.0-rc3-source.zip"
RC3_CHECKSUM_PATH = DIST_DIR / "cuda-acceleration-engine-v1.0.0-rc3-checksums.sha256"


def compute_sha256(data_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(data_bytes)
    return h.hexdigest().upper()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestStage1PackagingCompliance:
    """Test battery for BLK-01 and BLK-04 resolution."""

    def test_blk04_root_legal_files_exist(self):
        """Verify that root legal notice files exist physically and are non-empty."""
        files = {
            "LICENSE": PROJECT_ROOT / "LICENSE",
            "NOTICE": PROJECT_ROOT / "NOTICE",
            "THIRD_PARTY_LICENSES.md": PROJECT_ROOT / "THIRD_PARTY_LICENSES.md",
            "MODEL_CARD.md": PROJECT_ROOT / "MODEL_CARD.md",
        }
        for name, p in files.items():
            assert p.exists(), f"Required legal file '{name}' does not exist at {p}"
            assert p.stat().st_size > 0, f"File '{name}' is empty"

    def test_blk04_notice_content_integrity(self):
        """Verify key legal notices in NOTICE file."""
        notice_path = PROJECT_ROOT / "NOTICE"
        content = notice_path.read_text(encoding="utf-8")
        assert "NEXORA" in content
        assert "CUDA GPU Acceleration Engine" in content
        assert "Copyright (c) 2026 CUDA GPU Acceleration Engine Contributors" in content
        assert "NVIDIA CUDA Toolkit Attribution Notice" in content
        assert "out-of-band" in content
        assert "Pre-compiled binaries and upstream sources for these packages are NOT" in content
        assert "bundled within the NEXORA source distribution archive" in content

    def test_blk04_third_party_licenses_content_integrity(self):
        """Verify third party disclosure contents and unverified status markings."""
        tpl_path = PROJECT_ROOT / "THIRD_PARTY_LICENSES.md"
        content = tpl_path.read_text(encoding="utf-8")
        assert "Third-Party Software Licenses & Notices" in content
        assert "torch" in content
        assert "numpy" in content
        assert "fastapi" in content
        assert "requests" in content
        assert "UNVERIFIED" in content
        assert "Model Checkpoint Weights Redistribution" in content
        assert "Contributor Title Affirmation Execution" in content

    def test_blk01_packaging_script_includes_root_files(self):
        """Verify scripts/package_release.py explicitly includes legal files."""
        script_path = PROJECT_ROOT / "scripts" / "package_release.py"
        content = script_path.read_text(encoding="utf-8")
        assert '"LICENSE"' in content
        assert '"NOTICE"' in content
        assert '"THIRD_PARTY_LICENSES.md"' in content
        assert '"MODEL_CARD.md"' in content

    def test_blk01_rc3_archive_presence_and_checksum(self):
        """Verify RC3 archive exists and matches its manifest."""
        assert RC3_ARCHIVE_PATH.exists(), f"RC3 archive missing at {RC3_ARCHIVE_PATH}"
        assert RC3_CHECKSUM_PATH.exists(), f"RC3 checksum missing at {RC3_CHECKSUM_PATH}"

        archive_sha = compute_file_sha256(RC3_ARCHIVE_PATH)
        checksum_text = RC3_CHECKSUM_PATH.read_text(encoding="utf-8").strip()
        manifest_sha = checksum_text.split()[0].upper()
        assert archive_sha == manifest_sha, (
            f"Archive hash {archive_sha} does not match manifest hash {manifest_sha}"
        )

    def test_blk01_rc3_archive_contains_legal_files_bit_for_bit(self):
        """Verify that RC3 archive contains root legal files identical to disk."""
        with zipfile.ZipFile(RC3_ARCHIVE_PATH, "r") as zf:
            namelist = zf.namelist()
            expected_files = [
                ("cuda-acceleration-engine/LICENSE", PROJECT_ROOT / "LICENSE"),
                ("cuda-acceleration-engine/NOTICE", PROJECT_ROOT / "NOTICE"),
                ("cuda-acceleration-engine/THIRD_PARTY_LICENSES.md", PROJECT_ROOT / "THIRD_PARTY_LICENSES.md"),
                ("cuda-acceleration-engine/MODEL_CARD.md", PROJECT_ROOT / "MODEL_CARD.md"),
            ]
            for archive_rel_path, disk_path in expected_files:
                assert archive_rel_path in namelist, (
                    f"Expected '{archive_rel_path}' inside zip archive"
                )
                archive_bytes = zf.read(archive_rel_path)
                disk_bytes = disk_path.read_bytes()
                assert compute_sha256(archive_bytes) == compute_sha256(disk_bytes), (
                    f"Hash mismatch between disk and archive for {archive_rel_path}"
                )

    def test_rc3_archive_zero_unintended_cuda_redistributables(self):
        """Verify Section 5 Step 4: Zero proprietary binary runtimes or virtualenvs leaked into RC3."""
        banned_exts = [".dll", ".so", ".lib", ".pyd", ".dylib"]
        prohibited_dirs = {"node_modules", ".venv", "__pycache__", ".pytest_cache", ".git"}

        with zipfile.ZipFile(RC3_ARCHIVE_PATH, "r") as zf:
            for entry in zf.namelist():
                parts = entry.split("/")
                for d in prohibited_dirs:
                    assert d not in parts, f"Prohibited directory '{d}' found in archive entry: {entry}"
                
                assert not any(entry.lower().endswith(ext) for ext in banned_exts), (
                    f"Unauthorized binary runtime '{entry}' found in archive"
                )
