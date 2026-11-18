"""
Pytest configuration for CUDA GPU Acceleration Engine test suite.
Guarantees strict isolation of production database (data/training_jobs.db)
by redirecting tests to a temporary SQLite test database.
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure tests NEVER mutate the production database
_test_db_dir = Path(tempfile.gettempdir()) / "cuda_engine_test_suite"
_test_db_dir.mkdir(parents=True, exist_ok=True)
_test_db_path = _test_db_dir / "pytest_isolated_training_jobs.db"

# Set environment variable before any modules import JobStore
os.environ["CUDA_ENGINE_DB_PATH"] = str(_test_db_path)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure clean test environment and clean up test database after session."""
    yield
    # Cleanup test db files if desired
    try:
        if _test_db_path.exists():
            _test_db_path.unlink(missing_ok=True)
    except Exception:
        pass
