"""
scripts/run_milestone5_regression.py
Executes the comprehensive Phase 13 Milestone 5 regression test suite
across all 12 core test suites with clean process isolation,
and generates the machine-readable regression report.
"""

import sys
import time
import json
import subprocess
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_SUITES = [
    "tests/test_milestone13_milestone5_production_readiness.py",
    "tests/test_milestone13_milestone4_audit_readiness.py",
    "tests/test_milestone13_extended_pretraining.py",
    "tests/test_milestone13_pretraining_diagnostic.py",
    "tests/test_milestone13_dataset_engineering.py",
    "tests/test_milestone12_pretraining_campaign.py",
    "tests/test_milestone11_platform_integration.py",
    "tests/test_milestone10_flagship_transformer.py",
    "tests/test_milestone9_transformer.py",
    "tests/test_milestone8_training_reliability.py",
    "tests/test_stage2_milestone7_resnet.py",
    "tests/test_custom_kernels.py",
]

def run_suite(suite_rel_path: str):
    suite_path = ROOT_DIR / suite_rel_path
    cmd = [sys.executable, "-m", "pytest", str(suite_path), "-v"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    elapsed = time.time() - t0

    # Parse stdout for summary
    # Format e.g. "== 9 passed, 1 warning in 12.00s ==" or "== 10 passed in 2.09s =="
    passed = 0
    failed = 0
    skipped = 0

    match_passed = re.search(r"(\d+)\s+passed", proc.stdout)
    if match_passed:
        passed = int(match_passed.group(1))

    match_failed = re.search(r"(\d+)\s+failed", proc.stdout)
    if match_failed:
        failed = int(match_failed.group(1))

    match_skipped = re.search(r"(\d+)\s+skipped", proc.stdout)
    if match_skipped:
        skipped = int(match_skipped.group(1))

    return {
        "suite": suite_rel_path,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "exit_code": proc.returncode,
        "stdout": proc.stdout if proc.returncode != 0 else ""
    }

def main():
    print("=" * 80)
    print("PHASE 13 MILESTONE 5: COMPREHENSIVE REGRESSION TEST SWEEP (ISOLATED)")
    print("=" * 80)

    results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    start_total_time = time.time()

    for suite in TEST_SUITES:
        res = run_suite(suite)
        results.append(res)
        total_passed += res["passed"]
        total_failed += res["failed"]
        total_skipped += res["skipped"]

        status_str = "PASS" if res["exit_code"] == 0 else "FAIL"
        print(f"  [{status_str}] {suite:<55} {res['passed']:>3} passed, {res['failed']:>2} failed, {res['skipped']:>2} skipped ({res['elapsed_seconds']:.2f}s)")
        if res["exit_code"] != 0:
            print(f"[-] ERROR IN {suite}:\n{res['stdout']}")

    total_elapsed = time.time() - start_total_time

    print("=" * 80)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped across {len(TEST_SUITES)} suites in {total_elapsed:.2f}s")
    print("=" * 80)

    report_data = {
        "milestone": "Phase 13 Milestone 5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_suites": len(TEST_SUITES),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "all_passed": (total_failed == 0),
        "suites": [
            {k: v for k, v in r.items() if k != "stdout"} for r in results
        ]
    }

    out_path = ROOT_DIR / "results" / "phase13_milestone5_regression_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[+] Saved regression report to: {out_path}")
    if total_failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
