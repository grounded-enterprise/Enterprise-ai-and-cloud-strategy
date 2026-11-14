# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 6.3 — Redis atomic token-bucket enforcement plus the attribution wrapper that emits the priced event.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

import time
import redis.asyncio as redis

# Atomic token-bucket: refill by rate, then try to spend 'cost'. One round trip.
_BUCKET_LUA = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])   -- refill per second (budget units)
local burst = tonumber(ARGV[2])   -- bucket capacity (hard cap headroom)
local cost  = tonumber(ARGV[3])   -- estimated request cost
local now   = tonumber(ARGV[4])
local st    = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(st[1]) or burst
local last   = tonumber(st[2]) or now
tokens = math.min(burst, tokens + (now - last) * rate)   -- refill
local allowed = tokens >= cost
if allowed then tokens = tokens - cost end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 3600)
return { allowed and 1 or 0, tokens }
"""


class BudgetGate:
    def __init__(self, r: redis.Redis) -> None:
        self._r = r
        self._sha: str | None = None

    async def _load(self) -> str:
        if self._sha is None:
            self._sha = await self._r.script_load(_BUCKET_LUA)
        return self._sha

    async def admit(self, key: str, rate: float, burst: float,
                    est_cost: float) -> tuple[bool, float]:
        sha = await self._load()
        allowed, remaining = await self._r.evalsha(
            sha, 1, key, rate, burst, est_cost, time.time())
        return bool(allowed), float(remaining)


async def enforce_and_emit(gate: BudgetGate, emit, event: "TelemetryEvent",
                           soft: float, hard: float, est_cost: float):
    # hard-cap enforcement against the fast Redis counter (Sec 6.3)
    ok, remaining = await gate.admit(
        f"budget:{event.tenant}:{event.feature}",
        rate=hard / 3600.0, burst=hard, est_cost=est_cost)
    if not ok:
        raise BudgetExhausted(tenant=event.tenant, reset_hint="hourly")  # -> 429
    if remaining < (hard - soft):
        alert_owner(event.tenant, remaining)          # soft-cap: notify only
    await emit(event)     # async, off hot path -> stream -> ClickHouse (Sec 6.2)


class BudgetExhausted(Exception):
    def __init__(self, tenant: str, reset_hint: str) -> None:
        self.tenant, self.reset_hint = tenant, reset_hint
