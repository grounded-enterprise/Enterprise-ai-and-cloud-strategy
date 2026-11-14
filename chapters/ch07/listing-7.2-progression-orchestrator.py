# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 7.2 — Progression orchestrator: metric-gated promotion through tiers with automated failback on any breach.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class Tier(int, Enum):
    SHADOW = 0        # mirror traffic, discard output
    CANARY = 1        # 5% live
    ACTIVE_SPLIT = 2  # 50% live
    PRIMARY = 3       # 100% live; incumbent armed as fallback

TRAFFIC = {Tier.SHADOW: 0.0, Tier.CANARY: 0.05,
           Tier.ACTIVE_SPLIT: 0.50, Tier.PRIMARY: 1.0}


@dataclass(frozen=True)
class Gate:
    max_p95_ttft_s: float = 0.5
    max_error_pct: float = 1.0
    min_quality: float = 0.90      # eval score SLA
    max_cogs_ratio: float = 1.10   # <=110% of incumbent run-rate
    bake_seconds: float = 3600.0


@dataclass
class Metrics:
    p95_ttft_s: float
    error_pct: float
    quality: float
    cogs_ratio: float              # candidate / incumbent run-rate


@dataclass
class Candidate:
    name: str
    tier: Tier = Tier.SHADOW
    healthy_since: float = field(default_factory=time.monotonic)


class ProgressionOrchestrator:
    def __init__(self, gate: Gate) -> None:
        self.gate = gate

    def _passes(self, m: Metrics) -> bool:
        g = self.gate
        return (m.p95_ttft_s <= g.max_p95_ttft_s
                and m.error_pct <= g.max_error_pct
                and m.quality >= g.min_quality
                and m.cogs_ratio <= g.max_cogs_ratio)

    def evaluate(self, c: Candidate, m: Metrics) -> Candidate:
        now = time.monotonic()
        if not self._passes(m):
            # sustained breach on ANY axis -> immediate failback
            c.tier = Tier.SHADOW
            c.healthy_since = now
            return c                       # incumbent keeps serving
        # healthy: promote only after the full bake window at this tier
        if now - c.healthy_since >= self.gate.bake_seconds and c.tier < Tier.PRIMARY:
            c.tier = Tier(c.tier + 1)
            c.healthy_since = now
        return c

    def traffic_share(self, c: Candidate) -> float:
        return TRAFFIC[c.tier]


if __name__ == "__main__":
    orch = ProgressionOrchestrator(Gate())
    cand = Candidate("gpt-next")
    good = Metrics(p95_ttft_s=0.42, error_pct=0.3, quality=0.94, cogs_ratio=0.98)
    bad  = Metrics(p95_ttft_s=0.42, error_pct=0.3, quality=0.81, cogs_ratio=0.98)
    # a healthy bake promotes Shadow -> Canary; a quality breach fails back:
    cand.healthy_since -= 4000            # simulate a completed bake window
    print("promote ->", orch.evaluate(cand, good).tier.name,
          orch.traffic_share(cand))
    print("breach  ->", orch.evaluate(cand, bad).tier.name,
          orch.traffic_share(cand))
