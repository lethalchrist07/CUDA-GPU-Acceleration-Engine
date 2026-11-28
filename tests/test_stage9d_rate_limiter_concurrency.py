"""
NEXORA v1.0.0 — Stage 9D: Authentication Rate-Limiter Concurrency Regression Tests
Deterministic verification of thread-safe atomic check-and-record operations
preventing TOCTOU race conditions under high-concurrency authentication bursts.
"""
import concurrent.futures
import threading
from collections import deque
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status

from backend.auth import (
    AUTH_RATE_LIMITER,
    AuthRateLimiter,
    is_owner_authorized,
)


TEST_KEY = "test-secret-owner-key-stage9d-concurrency-32char"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Ensure a clean rate limiter state for every test."""
    AUTH_RATE_LIMITER.reset()
    yield
    AUTH_RATE_LIMITER.reset()


def make_mock_request(client_ip: str, token: str) -> MagicMock:
    """Create a mock Request with specified client IP and X-Owner-Key."""
    req = MagicMock()
    req.client.host = client_ip
    req.headers = {
        "x-owner-key": token,
    }
    return req


def make_mock_settings(owner_key: str = TEST_KEY) -> MagicMock:
    """Create mock settings with specified owner key."""
    settings = MagicMock()
    settings.owner_key = owner_key
    return settings


class TestRateLimiterConcurrency:
    """Deterministic concurrency tests for atomic check-and-record rate limiter."""

    def test_01_concurrent_failed_attempts_cannot_exceed_limit(self):
        """
        Verify that 30 simultaneous failed auth attempts from the same IP cannot
        bypass the 10-attempt limit through a check-then-record race.
        Precisely 10 attempts must be recorded as failures (returning False),
        and precisely 20 attempts must be blocked with HTTP 429.
        """
        num_threads = 30
        client_ip = "198.51.100.50"
        barrier = threading.Barrier(num_threads)
        settings = make_mock_settings()

        results = []
        lock = threading.Lock()

        def worker(thread_idx: int):
            req = make_mock_request(client_ip, f"invalid-token-{thread_idx}")
            barrier.wait()  # Synchronize release of all threads simultaneously
            try:
                outcome = is_owner_authorized(req, settings=settings)
                with lock:
                    results.append(("OK", outcome))
            except HTTPException as exc:
                with lock:
                    results.append(("429", exc.status_code, exc.headers.get("Retry-After")))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Categorize outcomes
        ok_results = [r for r in results if r[0] == "OK"]
        rate_limited_results = [r for r in results if r[0] == "429"]

        # Assert exactly 10 failed attempts were evaluated and recorded
        assert len(ok_results) == 10, f"Expected 10 failed attempts, got {len(ok_results)}"
        assert all(outcome is False for _, outcome in ok_results)

        # Assert exactly 20 attempts were immediately blocked by HTTP 429
        assert len(rate_limited_results) == 20, f"Expected 20 429s, got {len(rate_limited_results)}"
        for _, status_code, retry_after in rate_limited_results:
            assert status_code == 429
            assert retry_after is not None
            assert int(retry_after) > 0 and int(retry_after) <= 300

        # Verify internal history count is strictly capped at 10
        assert len(AUTH_RATE_LIMITER._failures[client_ip]) == 10

    def test_02_subsequent_attempts_remain_rate_limited(self):
        """
        After 10 failures have been recorded within the window, all subsequent
        attempts (whether sequential or concurrent) receive HTTP 429.
        """
        client_ip = "198.51.100.51"
        settings = make_mock_settings()

        # Exhaust 10 attempts
        for i in range(10):
            req = make_mock_request(client_ip, f"wrong-{i}")
            assert is_owner_authorized(req, settings=settings) is False

        # Next sequential attempt must receive 429
        with pytest.raises(HTTPException) as exc_info:
            req_11 = make_mock_request(client_ip, "wrong-11")
            is_owner_authorized(req_11, settings=settings)

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers
        assert int(exc_info.value.headers["Retry-After"]) > 0

    def test_03_separate_client_ips_remain_independently_limited_under_concurrency(self):
        """
        Concurrent authentication attempts from independent client IPs
        must be tracked and limited independently without cross-IP interference.
        """
        ip1 = "198.51.100.61"
        ip2 = "198.51.100.62"
        threads_per_ip = 25
        total_threads = threads_per_ip * 2
        barrier = threading.Barrier(total_threads)
        settings = make_mock_settings()

        results_ip1 = []
        results_ip2 = []
        lock = threading.Lock()

        def worker(ip: str, idx: int):
            req = make_mock_request(ip, f"bad-key-{idx}")
            barrier.wait()
            try:
                outcome = is_owner_authorized(req, settings=settings)
                with lock:
                    if ip == ip1:
                        results_ip1.append(("OK", outcome))
                    else:
                        results_ip2.append(("OK", outcome))
            except HTTPException as exc:
                with lock:
                    if ip == ip1:
                        results_ip1.append(("429", exc.status_code))
                    else:
                        results_ip2.append(("429", exc.status_code))

        threads = []
        for i in range(threads_per_ip):
            threads.append(threading.Thread(target=worker, args=(ip1, i)))
            threads.append(threading.Thread(target=worker, args=(ip2, i)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Both IPs must record exactly 10 failures and 15 rate-limits
        ok_ip1 = [r for r in results_ip1 if r[0] == "OK"]
        ratelimit_ip1 = [r for r in results_ip1 if r[0] == "429"]
        assert len(ok_ip1) == 10
        assert len(ratelimit_ip1) == 15

        ok_ip2 = [r for r in results_ip2 if r[0] == "OK"]
        ratelimit_ip2 = [r for r in results_ip2 if r[0] == "429"]
        assert len(ok_ip2) == 10
        assert len(ratelimit_ip2) == 15

        assert len(AUTH_RATE_LIMITER._failures[ip1]) == 10
        assert len(AUTH_RATE_LIMITER._failures[ip2]) == 10

    def test_04_successful_owner_authentication_clears_history_atomically(self):
        """
        Legitimate owner authentication succeeds and atomically resets failure
        history, allowing subsequent operations without hindrance.
        """
        client_ip = "198.51.100.70"
        settings = make_mock_settings()

        # Record 7 failures
        for i in range(7):
            req = make_mock_request(client_ip, f"wrong-{i}")
            assert is_owner_authorized(req, settings=settings) is False

        assert len(AUTH_RATE_LIMITER._failures[client_ip]) == 7

        # Successful authentication with valid key
        req_valid = make_mock_request(client_ip, TEST_KEY)
        assert is_owner_authorized(req_valid, settings=settings) is True

        # Failure history cleared
        assert client_ip not in AUTH_RATE_LIMITER._failures

    def test_05_locked_out_ip_blocks_even_valid_owner_key(self):
        """
        Once an IP is locked out (10 failures), subsequent requests — even with
        the valid owner key — must be blocked with HTTP 429 until the window expires.
        """
        client_ip = "198.51.100.80"
        settings = make_mock_settings()

        for i in range(10):
            req = make_mock_request(client_ip, f"wrong-{i}")
            assert is_owner_authorized(req, settings=settings) is False

        # Attempt with valid key from locked-out IP
        req_valid = make_mock_request(client_ip, TEST_KEY)
        with pytest.raises(HTTPException) as exc_info:
            is_owner_authorized(req_valid, settings=settings)

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    def test_06_fastapi_app_client_concurrency(self):
        """
        Test end-to-end HTTP request concurrency through FastAPI TestClient.
        """
        from fastapi.testclient import TestClient
        from backend.app import app
        import os

        old_key = os.environ.get("CUDA_ENGINE_OWNER_KEY")
        os.environ["CUDA_ENGINE_OWNER_KEY"] = TEST_KEY

        try:
            client = TestClient(app)
            num_requests = 20
            barrier = threading.Barrier(num_requests)
            status_codes = []
            lock = threading.Lock()

            def send_req(i: int):
                barrier.wait()
                resp = client.get(
                    "/api/health",
                    headers={"X-Owner-Key": f"concurrent-bad-{i}"},
                )
                with lock:
                    status_codes.append(resp.status_code)

            threads = [threading.Thread(target=send_req, args=(i,)) for i in range(num_requests)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            count_200 = status_codes.count(200)
            count_429 = status_codes.count(429)

            assert count_200 == 10, f"Expected 10 200s (public demo mode), got {count_200}"
            assert count_429 == 10, f"Expected 10 429s (rate limited), got {count_429}"
        finally:
            if old_key is None:
                os.environ.pop("CUDA_ENGINE_OWNER_KEY", None)
            else:
                os.environ["CUDA_ENGINE_OWNER_KEY"] = old_key
