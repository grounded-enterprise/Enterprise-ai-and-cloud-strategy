# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 10.3 — crossover_calculator.py : V_be, U_be, V_t and the cost-optimal decision matrix.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

"""Capacity Crossover & Volatility Calculator (Chapter 5, reference form).

A stand-alone module implementing the Capacity Crossover Equation (V_be) and
the Traffic Volatility Index (V_t), and emitting a cost-optimal decision matrix
between pay-as-you-go serverless endpoints and provisioned throughput units
(PTUs / Dedicated AI Clusters). Pure standard library; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Strategy(str, Enum):
    SERVERLESS = "serverless"
    PROVISIONED_PEAK = "provisioned_peak"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ServerlessPricing:
    """Pay-per-token pricing, USD per million tokens."""
    usd_per_m_input: float
    usd_per_m_output: float

    def blended(self, completion_fraction: float) -> float:
        """Blended price given alpha = output-token fraction in [0, 1]."""
        if not 0.0 <= completion_fraction <= 1.0:
            raise ValueError("completion_fraction must be in [0, 1]")
        a = completion_fraction
        return (1.0 - a) * self.usd_per_m_input + a * self.usd_per_m_output


@dataclass(frozen=True)
class ProvisionedPricing:
    """Committed-capacity pricing for a provisioned unit (PTU / model unit / DAC)."""
    usd_per_unit_month: float
    m_tokens_day_per_unit: float           # throughput a unit delivers, M tok/day

    def monthly_cost(self, units: int) -> float:
        return units * self.usd_per_unit_month

    def monthly_capacity(self, units: int) -> float:
        return units * self.m_tokens_day_per_unit * 30.0


@dataclass
class VolumeMetrics:
    """Observed demand, in millions of tokens per day."""
    mu_day: float
    sigma_day: float
    completion_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.mu_day <= 0:
            raise ValueError("mu_day must be positive")
        if self.sigma_day < 0:
            raise ValueError("sigma_day must be non-negative")

    @property
    def monthly(self) -> float:
        return self.mu_day * 30.0

    @property
    def volatility_index(self) -> float:
        """Traffic Volatility Index V_t = sigma / mu."""
        return self.sigma_day / self.mu_day


@dataclass
class DecisionMatrix:
    """The cost-optimal routing decision and its supporting diagnostics."""
    strategy: Strategy
    provisioned_units: int
    blended_price: float
    v_be_month: float
    u_be: float
    v_t: float
    cost_serverless: float
    cost_provisioned_peak: float
    cost_hybrid: float

    @property
    def best_cost(self) -> float:
        return min(self.cost_serverless, self.cost_provisioned_peak, self.cost_hybrid)

    @property
    def savings_vs_serverless(self) -> float:
        if self.cost_serverless == 0:
            return 0.0
        return 1.0 - self.best_cost / self.cost_serverless

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "strategy": self.strategy.value,
            "provisioned_units": self.provisioned_units,
            "blended_price": round(self.blended_price, 4),
            "v_be_month": round(self.v_be_month, 1),
            "u_be": round(self.u_be, 4),
            "v_t": round(self.v_t, 4),
            "cost_serverless": round(self.cost_serverless, 2),
            "cost_provisioned_peak": round(self.cost_provisioned_peak, 2),
            "cost_hybrid": round(self.cost_hybrid, 2),
            "best_cost": round(self.best_cost, 2),
            "savings_vs_serverless": round(self.savings_vs_serverless, 4),
        }


class CrossoverCalculator:
    """Computes V_be, U_be, V_t and the cost-minimizing hybrid split."""

    def __init__(self, serverless: ServerlessPricing, provisioned: ProvisionedPricing) -> None:
        self.serverless = serverless
        self.provisioned = provisioned

    def _units_for(self, m_tokens_day: float) -> int:
        return max(1, math.ceil(m_tokens_day / self.provisioned.m_tokens_day_per_unit))

    @staticmethod
    def _expected_daily_burst(sigma_day: float) -> float:
        """E[max(0, D - mu)] for D ~ Normal(mu, sigma) with baseline K = mu."""
        return sigma_day / math.sqrt(2.0 * math.pi)

    def capacity_crossover_volume(self, c_fixed: float, blended_price: float) -> float:
        """V_be = C_fixed / P_blended (break-even monthly token volume)."""
        if blended_price <= 0:
            raise ValueError("blended_price must be positive")
        return c_fixed / blended_price

    def break_even_utilization(self, c_fixed: float, blended_price: float,
                               capacity_month: float) -> float:
        """U_be = C_fixed / (P_blended * K)."""
        denom = blended_price * capacity_month
        if denom <= 0:
            raise ValueError("blended_price and capacity must be positive")
        return c_fixed / denom

    def evaluate(self, metrics: VolumeMetrics) -> DecisionMatrix:
        """Return the cost-optimal strategy and full decision matrix."""
        price = self.serverless.blended(metrics.completion_fraction)

        # Pure serverless: linear in volume.
        cost_serverless = metrics.monthly * price

        # Provision to peak (mu + 2 sigma): safe but idle-heavy.
        peak_units = self._units_for(metrics.mu_day + 2.0 * metrics.sigma_day)
        cost_peak = self.provisioned.monthly_cost(peak_units)

        # Hybrid: provision the mean, burst the expected overflow to serverless.
        base_units = self._units_for(metrics.mu_day)
        c_fixed = self.provisioned.monthly_cost(base_units)
        capacity_month = self.provisioned.monthly_capacity(base_units)
        burst_month = self._expected_daily_burst(metrics.sigma_day) * 30.0
        cost_hybrid = c_fixed + burst_month * price

        v_be = self.capacity_crossover_volume(c_fixed, price)
        u_be = self.break_even_utilization(c_fixed, price, capacity_month)

        costs = {
            Strategy.SERVERLESS: cost_serverless,
            Strategy.PROVISIONED_PEAK: cost_peak,
            Strategy.HYBRID: cost_hybrid,
        }
        best = min(costs, key=costs.get)
        units = {
            Strategy.SERVERLESS: 0,
            Strategy.PROVISIONED_PEAK: peak_units,
            Strategy.HYBRID: base_units,
        }[best]

        return DecisionMatrix(
            strategy=best, provisioned_units=units, blended_price=price,
            v_be_month=v_be, u_be=u_be, v_t=metrics.volatility_index,
            cost_serverless=cost_serverless, cost_provisioned_peak=cost_peak,
            cost_hybrid=cost_hybrid,
        )


if __name__ == "__main__":
    calc = CrossoverCalculator(
        ServerlessPricing(usd_per_m_input=3.0, usd_per_m_output=15.0),
        ProvisionedPricing(usd_per_unit_month=260.0, m_tokens_day_per_unit=3.6),
    )
    decision = calc.evaluate(VolumeMetrics(mu_day=50.0, sigma_day=20.0,
                                           completion_fraction=0.30))
    for key, value in decision.as_dict().items():
        print(f"{key:24s}: {value}")
