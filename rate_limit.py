from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Tuple

from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class RateLimitRule:
    requests: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, rule: RateLimitRule) -> None:
        if rule.requests <= 0:
            return
        now = time.monotonic()
        cutoff = now - rule.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= rule.requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please wait and try again.",
                )
            events.append(now)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


limiter = InMemoryRateLimiter()


def parse_limit(value: str, fallback: RateLimitRule) -> RateLimitRule:
    raw = value.strip().lower()
    if not raw:
        return fallback
    if "/" not in raw:
        return fallback
    count_raw, window_raw = raw.split("/", 1)
    try:
        count = int(count_raw)
    except ValueError:
        return fallback
    units = {"s": 1, "m": 60, "h": 3600}
    suffix = window_raw[-1]
    if suffix in units:
        number = window_raw[:-1]
        try:
            window = int(number) * units[suffix]
        except ValueError:
            return fallback
    else:
        try:
            window = int(window_raw)
        except ValueError:
            return fallback
    return RateLimitRule(count, max(window, 1))


def rule_for(name: str, fallback: RateLimitRule) -> RateLimitRule:
    return parse_limit(os.environ.get(f"RATE_LIMIT_{name.upper()}", ""), fallback)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def rate_limit(request: Request, name: str, fallback: RateLimitRule, user_id: str | None = None) -> None:
    if os.environ.get("RATE_LIMIT_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
        return
    identity = user_id or client_ip(request)
    rule = rule_for(name, fallback)
    limiter.check(f"{name}:{identity}", rule)
