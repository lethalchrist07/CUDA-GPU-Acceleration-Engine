import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.data import ArtifactStore, collect_hardware, read_json

ROOT = Path(__file__).resolve().parents[2]


def test_actual_phase2_results_and_manifest():
    store = ArtifactStore(ROOT)
    result = store.benchmarks()
    assert result["status"] == "available"
    assert result["source"]["provenance"] == "phase2_hash_verified"
    assert len(result["rows"]) == 12
    assert all(row["verified"] for row in result["rows"])
    assert store.verification()["manifest_status"] == "matched"


def test_actual_historical_experiments_remain_labeled_unverified():
    store = ArtifactStore(ROOT)
    for kind, count in [("training", 3), ("inference", 6)]:
        result = store.experiments(kind)
        assert len(result["rows"]) == count
        assert result["source"]["provenance"] == "historical_unverified"
    assert "do not independently prove GPU offload" in store.experiments("inference")["warnings"][1]


def test_missing_files_are_unavailable_not_fake_measurements(tmp_path):
    store = ArtifactStore(tmp_path)
    for result in [store.benchmarks(), store.experiments("training"), store.experiments("inference")]:
        assert result["status"] == "unavailable"
        assert result["rows"] == []
        assert result["source"] is None
    assert store.overview()["status"] == "degraded"
    assert store.verification()["manifest_status"] == "unavailable"


@pytest.mark.parametrize("text", ['{"benchmarks": NaN}', '{"benchmarks": Infinity}', '[]', '{broken'])
def test_bad_json_is_rejected(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_nonpositive_timing_rejected(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "benchmark_results.json").write_text(json.dumps({"benchmarks": [
        {"cpu_time_ms": 1, "gpu_kernel_ms": 0, "gpu_end_to_end_ms": 2, "verified": True}
    ]}), encoding="utf-8")
    assert ArtifactStore(tmp_path).benchmarks()["status"] == "unavailable"


def test_manifest_mismatch_is_visible(tmp_path):
    directory = tmp_path / "results/phase2_verification_test"
    (directory / "final_run").mkdir(parents=True)
    original = ROOT / "results/phase2_verification_20261016/final_run/benchmark_results.json"
    path = directory / "final_run/benchmark_results.json"
    path.write_bytes(original.read_bytes())
    (directory / "manifest.json").write_text(json.dumps({
        path.relative_to(tmp_path).as_posix(): {"sha256": "incorrect"}
    }), encoding="utf-8")
    store = ArtifactStore(tmp_path)
    assert store.benchmarks()["source"]["provenance"] == "hash_mismatch"
    assert store.verification()["manifest_status"] == "mismatch"


def test_artifact_paths_are_allowlisted_and_confined():
    store = ArtifactStore(ROOT)
    paths = store.artifact_paths()
    assert "benchmark-json" in paths
    assert all(path.resolve().is_relative_to(ROOT) for path in paths.values())
    assert "../requirements.txt" not in paths
    with pytest.raises(ValueError):
        store._inside_root(ROOT.parent / "outside.txt")


def test_telemetry_unavailable_does_not_report_zero_performance():
    with patch("backend.data.shutil.which", return_value=None):
        result = collect_hardware()
    assert result["status"] == "unavailable"
    assert result["gpus"] == []


def test_nvidia_unavailable_metric_remains_null():
    # Explicit parser fixture, never returned by the real dashboard API.
    from subprocess import CompletedProcess
    output = "0, Test GPU, test-uuid, 1.0, 6144, 10, 6000, 0, 45, 11.2, [N/A]\n"
    with patch("backend.data.shutil.which", return_value="nvidia-smi"), patch(
        "backend.data.subprocess.run", return_value=CompletedProcess([], 0, output, "")
    ):
        result = collect_hardware()
    assert result["gpus"][0]["power_limit_w"] is None
    assert result["gpus"][0]["utilization_percent"] == 0
    assert result["gpus"][0]["memory_total_mib"] == 6144
