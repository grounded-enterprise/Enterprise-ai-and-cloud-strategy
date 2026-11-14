# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Golden-value tests for the Capacity Crossover Equation.

Chapter 5 prints every intermediate of one worked scenario. That makes the
whole calculation a regression fixture: if a refactor moves any of these
numbers, the book and the code have diverged and one of them is wrong.

These are the tests that would have caught the crossover equation carrying its
unknown on both sides. Cheap, and they need no cloud account.
"""
import math
import unittest

import _paths  # noqa: F401  (import for the sys.path side effect)

from crossover_calculator import (
    CrossoverCalculator,
    DecisionMatrix,
    ProvisionedPricing,
    ServerlessPricing,
    Strategy,
    VolumeMetrics,
)

# The scenario worked in Chapter 5.
SERVERLESS = ServerlessPricing(usd_per_m_input=3.0, usd_per_m_output=15.0)
PROVISIONED = ProvisionedPricing(usd_per_unit_month=260.0, m_tokens_day_per_unit=3.6)
METRICS = VolumeMetrics(mu_day=50.0, sigma_day=20.0, completion_fraction=0.30)


class ChapterFiveWorkedExample(unittest.TestCase):
    """Every number the chapter prints, asserted against the implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.calc = CrossoverCalculator(SERVERLESS, PROVISIONED)
        cls.decision: DecisionMatrix = cls.calc.evaluate(METRICS)

    def test_blended_price_is_6_60_per_million(self):
        # 70% input at $3 + 30% output at $15.
        self.assertAlmostEqual(self.decision.blended_price, 6.60, places=6)

    def test_monthly_volume_is_1_5_billion_tokens(self):
        self.assertAlmostEqual(METRICS.monthly, 1500.0, places=6)

    def test_serverless_month_is_9900(self):
        self.assertAlmostEqual(self.decision.cost_serverless, 9900.0, places=2)

    def test_volatility_index_is_0_40(self):
        # V_t = sigma / mu = 20 / 50.
        self.assertAlmostEqual(self.decision.v_t, 0.40, places=6)

    def test_committed_cost_is_3640(self):
        # ceil(50 / 3.6) = 14 units at $260.
        base_units = math.ceil(METRICS.mu_day / PROVISIONED.m_tokens_day_per_unit)
        self.assertEqual(base_units, 14)
        self.assertAlmostEqual(PROVISIONED.monthly_cost(base_units), 3640.0, places=2)

    def test_break_even_utilization_is_36_5_percent(self):
        self.assertAlmostEqual(self.decision.u_be, 0.3648, places=4)

    def test_hybrid_month_is_5220(self):
        self.assertAlmostEqual(self.decision.cost_hybrid, 5219.81, places=2)

    def test_hybrid_saves_47_3_percent(self):
        self.assertAlmostEqual(self.decision.savings_vs_serverless, 0.4727, places=4)

    def test_hybrid_is_the_chosen_strategy(self):
        self.assertIs(self.decision.strategy, Strategy.HYBRID)
        self.assertEqual(self.decision.provisioned_units, 14)


class CrossoverEquationInvariants(unittest.TestCase):
    """Properties that must hold for any inputs, not just the worked scenario.

    The printed form of V_be once had the unknown on both sides. An identity
    test is the guard against that class of error returning.
    """

    def setUp(self) -> None:
        self.calc = CrossoverCalculator(SERVERLESS, PROVISIONED)

    def test_v_be_is_the_volume_where_serverless_equals_committed(self):
        # At V_be, serverless spend must equal the committed fixed cost.
        c_fixed, price = 3640.0, 6.60
        v_be = self.calc.capacity_crossover_volume(c_fixed, price)
        self.assertAlmostEqual(v_be * price, c_fixed, places=6)

    def test_v_be_depends_only_on_the_commitment_it_sizes(self):
        # V_be is a function of (C_fixed, price) — not of observed volume
        # directly. `evaluate` sizes the commitment from demand, so the V_be it
        # reports moves with demand; what must hold is that it always describes
        # the commitment actually being priced.
        for mu in (10.0, 50.0, 250.0):
            with self.subTest(mu_day=mu):
                d = self.calc.evaluate(VolumeMetrics(mu_day=mu, sigma_day=mu * 0.4))
                base_units = math.ceil(mu / PROVISIONED.m_tokens_day_per_unit)
                c_fixed = PROVISIONED.monthly_cost(base_units)
                self.assertAlmostEqual(d.v_be_month * d.blended_price, c_fixed, places=6)

    def test_v_be_is_independent_of_volume_for_a_fixed_commitment(self):
        # Held at one commitment size, the break-even volume is constant.
        v_be = {self.calc.capacity_crossover_volume(3640.0, 6.60) for _ in range(3)}
        self.assertEqual(len(v_be), 1)
        self.assertAlmostEqual(v_be.pop(), 3640.0 / 6.60, places=9)

    def test_utilization_and_volume_break_even_agree(self):
        # U_be is V_be expressed as a fraction of provisioned capacity.
        c_fixed, price, capacity = 3640.0, 6.60, 14 * 3.6 * 30.0
        v_be = self.calc.capacity_crossover_volume(c_fixed, price)
        u_be = self.calc.break_even_utilization(c_fixed, price, capacity)
        self.assertAlmostEqual(u_be, v_be / capacity, places=9)

    def test_chosen_cost_is_never_worse_than_serverless(self):
        for mu in (1.0, 10.0, 50.0, 250.0, 1000.0):
            with self.subTest(mu_day=mu):
                d = self.calc.evaluate(VolumeMetrics(mu_day=mu, sigma_day=mu * 0.4))
                self.assertLessEqual(d.best_cost, d.cost_serverless + 1e-9)
                self.assertGreaterEqual(d.savings_vs_serverless, -1e-9)

    def test_low_volume_prefers_serverless(self):
        d = self.calc.evaluate(VolumeMetrics(mu_day=1.0, sigma_day=0.2))
        self.assertIs(d.strategy, Strategy.SERVERLESS)
        self.assertEqual(d.provisioned_units, 0)

    def test_high_volume_stops_preferring_serverless(self):
        d = self.calc.evaluate(VolumeMetrics(mu_day=2000.0, sigma_day=100.0))
        self.assertIsNot(d.strategy, Strategy.SERVERLESS)

    def test_flat_traffic_has_zero_volatility(self):
        d = self.calc.evaluate(VolumeMetrics(mu_day=50.0, sigma_day=0.0))
        self.assertAlmostEqual(d.v_t, 0.0, places=9)
        # With no burst, hybrid degenerates to the committed cost alone.
        self.assertAlmostEqual(d.cost_hybrid, 3640.0, places=2)

    def test_volatility_raises_the_hybrid_cost_monotonically(self):
        costs = [
            self.calc.evaluate(VolumeMetrics(mu_day=50.0, sigma_day=s)).cost_hybrid
            for s in (0.0, 10.0, 20.0, 40.0)
        ]
        self.assertEqual(costs, sorted(costs))


class InputValidation(unittest.TestCase):
    def test_completion_fraction_must_be_a_fraction(self):
        for bad in (-0.1, 1.1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                SERVERLESS.blended(bad)

    def test_mean_volume_must_be_positive(self):
        with self.assertRaises(ValueError):
            VolumeMetrics(mu_day=0.0, sigma_day=1.0)

    def test_standard_deviation_must_not_be_negative(self):
        with self.assertRaises(ValueError):
            VolumeMetrics(mu_day=50.0, sigma_day=-1.0)

    def test_blended_price_must_be_positive(self):
        calc = CrossoverCalculator(SERVERLESS, PROVISIONED)
        with self.assertRaises(ValueError):
            calc.capacity_crossover_volume(3640.0, 0.0)


if __name__ == "__main__":
    unittest.main()
