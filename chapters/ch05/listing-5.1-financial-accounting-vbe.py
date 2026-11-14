# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 5.1 — Financial accounting and V_be optimization engine.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class InfrastructureCostParameters(BaseModel):
    on_demand_rate_per_1m: float
    provisioned_monthly_base_fee: float
    provisioned_included_tokens_1m: float
    provisioned_overage_rate_per_1m: float

class CrossoverAnalysisResult(BaseModel):
    break_even_volume_1m_tokens: float
    current_monthly_volume_1m_tokens: float
    recommended_mode: str
    projected_monthly_savings_usd: float

class FinancialOptimizationEngine:
    """Calculates V_be metrics and routes traffic between provisioned and on-demand endpoints."""

    def __init__(self, params: InfrastructureCostParameters):
        self.params = params

    def compute_break_even_volume(self) -> float:
        """Calculates the break-even volume threshold (V_be) in millions of tokens."""
        rate_diff = self.params.on_demand_rate_per_1m - self.params.provisioned_overage_rate_per_1m
        if rate_diff <= 0:
            return float("inf")  # Provisioned is never cheaper if overage equals on-demand

        return self.params.provisioned_monthly_base_fee / rate_diff

    def analyze_cost_efficiency(self, current_volume_1m_tokens: float) -> CrossoverAnalysisResult:
        v_be = self.compute_break_even_volume()

        on_demand_total = current_volume_1m_tokens * self.params.on_demand_rate_per_1m

        if current_volume_1m_tokens <= self.params.provisioned_included_tokens_1m:
            provisioned_total = self.params.provisioned_monthly_base_fee
        else:
            overage = current_volume_1m_tokens - self.params.provisioned_included_tokens_1m
            provisioned_total = (
                self.params.provisioned_monthly_base_fee +
                (overage * self.params.provisioned_overage_rate_per_1m)
            )

        if current_volume_1m_tokens >= v_be:
            recommendation = "PROVISIONED_CAPACITY"
            savings = on_demand_total - provisioned_total
        else:
            recommendation = "ON_DEMAND_APIS"
            savings = provisioned_total - on_demand_total

        return CrossoverAnalysisResult(
            break_even_volume_1m_tokens=v_be,
            current_monthly_volume_1m_tokens=current_volume_1m_tokens,
            recommended_mode=recommendation,
            projected_monthly_savings_usd=abs(savings)
        )

@dataclass
class ProvisionedCapacityTracker:
    total_allocated_ptu_capacity: int
    currently_active_ptu_utilization: int

    def possesses_available_capacity(self) -> bool:
        return self.currently_active_ptu_utilization < self.total_allocated_ptu_capacity

    def allocate_slot(self) -> bool:
        if self.possesses_available_capacity():
            self.currently_active_ptu_utilization += 1
            return True
        return False

    def release_slot(self) -> None:
        self.currently_active_ptu_utilization = max(0, self.currently_active_ptu_utilization - 1)

class HybridCapacityRouter:
    """Directs traffic to provisioned capacity allocations, bursting overflow to on-demand APIs."""

    def __init__(self, tracker: ProvisionedCapacityTracker):
        self.tracker = tracker

    def resolve_execution_target(self) -> Dict[str, Any]:
        if self.tracker.allocate_slot():
            return {"target_tier": "PROVISIONED_CAPACITY", "billing_mode": "FIXED_BASELINE"}
        else:
            return {"target_tier": "ON_DEMAND_OVERFLOW", "billing_mode": "VARIABLE_BURST"}
