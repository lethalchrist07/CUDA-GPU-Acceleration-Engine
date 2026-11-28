"""
NEXORA v1.0.0 — Stage 8H-R3: Secure Owner Mode Authentication Module
Provides stateless, constant-time validation of owner credentials against
CUDA_ENGINE_OWNER_KEY with thread-safe sliding-window rate limiting on
failed authentication attempts (max 10 failures / 300s) and spoof-resistant
client IP resolution. Fail-safe closed if owner key is not configured.
"""
from __future__ import annotations

import ipaddress
import math
import secrets
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from starlette.requests import Request

from backend.config import get_settings


class AuthRateLimiter:
    """
    Thread-safe sliding-window rate limiter for failed authentication attempts.
    Tracks timestamps of failed attempts per client IP.
    Enforces maximum `max_failures` within `window_seconds`.
    Exceeding limit raises HTTP 429 Too Many Requests with Retry-After header.
    Guarantees atomic check-and-record under a single lock to eliminate TOCTOU races.
    """

    def __init__(self, max_failures: int = 10, window_seconds: float = 300.0):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._failures: Dict[str, deque[float]] = {}

    def _purge_expired_locked(self, client_ip: str, now: float) -> Optional[deque[float]]:
        """Purge timestamps older than window_seconds for client_ip. Assumes self._lock is held."""
        history = self._failures.get(client_ip)
        if not history:
            return None

        cutoff = now - self.window_seconds
        while history and history[0] < cutoff:
            if isinstance(history, deque):
                history.popleft()
            else:
                history.pop(0)

        if not history:
            del self._failures[client_ip]
            return None

        return history

    def _raise_rate_limited(self, history: deque[float], now: float) -> None:
        """Helper to compute Retry-After and raise HTTP 429. Assumes self._lock is held."""
        retry_after = max(1, min(int(self.window_seconds), math.ceil(self.window_seconds - (now - history[0]))))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed authentication attempts. Access locked for {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    def check_rate_limit(self, client_ip: str) -> None:
        """Raise HTTP 429 if client_ip has exceeded max_failures within window_seconds."""
        now = time.time()
        with self._lock:
            history = self._purge_expired_locked(client_ip, now)
            if history and len(history) >= self.max_failures:
                self._raise_rate_limited(history, now)

    def record_failure(self, client_ip: str) -> None:
        """Record a failed authentication attempt."""
        now = time.time()
        with self._lock:
            history = self._purge_expired_locked(client_ip, now)
            if history is None:
                history = deque()
                self._failures[client_ip] = history
            history.append(now)

    def check_and_record_failure(self, client_ip: str) -> None:
        """
        Atomically check whether client_ip has exceeded max_failures within window_seconds,
        and if not, record the failed authentication attempt under the same lock.
        Raises HTTP 429 Too Many Requests if the limit is reached/exceeded.
        """
        now = time.time()
        with self._lock:
            history = self._purge_expired_locked(client_ip, now)
            if history and len(history) >= self.max_failures:
                self._raise_rate_limited(history, now)
            if history is None:
                history = deque()
                self._failures[client_ip] = history
            history.append(now)

    def check_and_record_success(self, client_ip: str) -> None:
        """
        Atomically check whether client_ip is currently rate-limited before confirming success.
        If locked out, raises HTTP 429 Too Many Requests.
        Otherwise, clears any failure history for client_ip under the same lock.
        """
        now = time.time()
        with self._lock:
            history = self._purge_expired_locked(client_ip, now)
            if history and len(history) >= self.max_failures:
                self._raise_rate_limited(history, now)
            if client_ip in self._failures:
                del self._failures[client_ip]

    def process_attempt(self, client_ip: str, is_valid: bool) -> bool:
        """
        Atomically evaluate and record an authentication attempt under a single lock.
        If is_valid is False: records failure or raises 429 if limit already reached.
        If is_valid is True: clears failure history or raises 429 if already locked out.
        """
        if is_valid:
            self.check_and_record_success(client_ip)
            return True
        else:
            self.check_and_record_failure(client_ip)
            return False

    def record_success(self, client_ip: str) -> None:
        """Clear failure history upon successful owner authentication."""
        with self._lock:
            if client_ip in self._failures:
                del self._failures[client_ip]

    def reset(self) -> None:
        """Reset state across all client IPs (used for testing and lifecycle reloads)."""
        with self._lock:
            self._failures.clear()


AUTH_RATE_LIMITER = AuthRateLimiter(max_failures=10, window_seconds=300.0)


def get_trusted_client_ip(request: Request) -> str:
    """
    Safely extract client IP, preventing spoofed forwarding headers from bypassing rate limiting.

    Rules:
    1. If direct connection (request.client.host) is not a trusted local loopback/proxy,
       use request.client.host directly. External clients cannot spoof their TCP peer IP.
    2. If direct connection is local loopback (127.0.0.1, ::1) indicating a local reverse proxy
       (such as ngrok or cloudflared), inspect trusted proxy headers:
       a. CF-Connecting-IP (set by Cloudflare edge).
       b. Rightmost valid entry in X-Forwarded-For (since proxies append to the end,
          preventing client-injected prefixes from being accepted).
    3. Validate IP format using ipaddress; fallback to direct host if malformed.
    """
    direct_ip = request.client.host if request.client else "127.0.0.1"

    # Check if direct connection is loopback
    is_loopback = False
    try:
        ip_obj = ipaddress.ip_address(direct_ip)
        is_loopback = ip_obj.is_loopback
    except ValueError:
        is_loopback = direct_ip in ("127.0.0.1", "localhost", "::1", "testclient")

    if not is_loopback:
        # Untrusted external peer; ignore forwarding headers to prevent header spoofing
        return direct_ip

    # If coming through local tunnel proxy (cloudflared / ngrok / testclient)
    # Check CF-Connecting-IP first
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip and cf_ip.strip():
        val = cf_ip.strip()
        try:
            ipaddress.ip_address(val)
            return val
        except ValueError:
            pass

    # Next check X-Forwarded-For: take the rightmost valid entry before local proxy
    xff = request.headers.get("x-forwarded-for")
    if xff and xff.strip():
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        for part in reversed(parts):
            try:
                parsed = ipaddress.ip_address(part)
                return str(parsed)
            except ValueError:
                continue

    return direct_ip


def get_owner_token_from_request(request: Request) -> Optional[str]:
    """Extract owner key/token from X-Owner-Key or Authorization: Bearer headers."""
    # 1. Custom header
    x_key = request.headers.get("x-owner-key")
    if x_key and x_key.strip():
        return x_key.strip()

    # 2. Authorization header
    auth = request.headers.get("authorization")
    if auth and auth.strip():
        auth_str = auth.strip()
        if auth_str.lower().startswith("bearer "):
            token = auth_str[7:].strip()
            if token:
                return token
        return auth_str

    return None


def is_owner_authorized(
    request: Request,
    settings: Optional[Any] = None,
    enforce_rate_limit: bool = True,
) -> bool:
    """
    Check if the incoming request is authorized as project owner.
    - Validates client IP against failed authentication rate limiter.
    - Fails safe (returns False) if CUDA_ENGINE_OWNER_KEY is not configured or empty.
    - Uses secrets.compare_digest for constant-time comparison to prevent timing attacks.
    - Records failed authentication attempts and enforces HTTP 429 after 10 failures / 300s.
    """
    if settings is None:
        settings = get_settings()
    configured_key = (getattr(settings, "owner_key", None) or "").strip()

    provided_token = get_owner_token_from_request(request)
    if not provided_token:
        # Anonymous public request; not an attempted owner authentication
        return False

    client_ip = get_trusted_client_ip(request)

    if not configured_key:
        if enforce_rate_limit:
            AUTH_RATE_LIMITER.check_and_record_failure(client_ip)
        return False

    is_valid = secrets.compare_digest(
        provided_token.encode("utf-8"),
        configured_key.encode("utf-8"),
    )

    if enforce_rate_limit:
        return AUTH_RATE_LIMITER.process_attempt(client_ip, is_valid)
    return is_valid
