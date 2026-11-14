# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Every Python listing in `chapters/`, executed as shipped.

The other test modules cover the five `reference/` modules and the extensions.
This one loads each chapter listing by path and runs it, so that "the code in
the book runs" is something the suite demonstrates listing by listing.

A listing is an excerpt, and three of them stop short in ways a reader would
hit on first use. Those are marked `expectedFailure` with the reason, in the
same way the suite already tracks the redaction defect: the code is printed, so
the fix is a manuscript change. When one reports *unexpected success* the fix
has landed and the decorator comes off.

Listings that need pydantic are skipped without it. Listings 3.1 and 6.3 import
FastAPI and `redis.asyncio`; without those packages a stand-in for the framework
is registered (see `_listings.py`) and the listing's own logic still runs. The
Lua script in Listing 6.3 is never executed here: that needs a Redis server.
"""
import asyncio
import json
import time
import unittest
from decimal import Decimal

import _listings

HAVE_PYDANTIC = _listings.have("pydantic")
HAVE_OTEL = _listings.have("opentelemetry")
needs_pydantic = unittest.skipUnless(HAVE_PYDANTIC, "listing imports pydantic")
needs_otel = unittest.skipUnless(HAVE_OTEL, "listing imports opentelemetry-api")


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  Chapter 1
# --------------------------------------------------------------------------- #
@needs_pydantic
class Listing_1_2_DynamicRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch01/listing-1.2-dynamic-router.py")

    def _binding(self, provider, cost_in, cost_out, p95, **kw):
        m = self.m
        return m.ModelBinding(provider=provider, model_id=f"{provider.value}-model", region="us-east-1",
                              usd_per_1m_input=cost_in, usd_per_1m_output=cost_out,
                              measured_p95_ms=p95, supports_zdr=kw.pop("zdr", True), **kw)

    def _request(self, **constraints):
        m = self.m
        return m.UnifiedInferenceRequest(
            model_role="chat", messages=[m.Message(role="user", content="hi")],
            routing_constraints=m.RoutingConstraints(**constraints))

    def _router(self, bindings, adapters=None):
        m = self.m
        return m.PolicyRouter({"chat": m.ModelRole(role="chat", bindings=bindings)}, adapters or {})

    def test_a_request_names_a_role_and_no_provider(self):
        fields = set(self.m.UnifiedInferenceRequest.model_fields)
        self.assertIn("model_role", fields)
        self.assertFalse({"provider", "model_id", "deployment"} & fields)

    def test_an_empty_message_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.m.UnifiedInferenceRequest(model_role="chat", messages=[])

    def test_candidates_are_ranked_cheapest_first_when_cost_is_all_that_counts(self):
        P = self.m.Provider
        router = self._router([self._binding(P.AWS, 3.0, 15.0, 400), self._binding(P.GCP, 1.0, 4.0, 900)])
        ranked = router.resolve_candidates(self._request(cost_weight=1.0))
        self.assertEqual([b.provider for b in ranked], [P.GCP, P.AWS])

    def test_candidates_are_ranked_fastest_first_when_latency_is_all_that_counts(self):
        P = self.m.Provider
        router = self._router([self._binding(P.AWS, 3.0, 15.0, 400), self._binding(P.GCP, 1.0, 4.0, 900)])
        ranked = router.resolve_candidates(self._request(cost_weight=0.0))
        self.assertEqual([b.provider for b in ranked], [P.AWS, P.GCP])

    def test_policy_filters_remove_ineligible_bindings(self):
        P = self.m.Provider
        router = self._router([
            self._binding(P.AWS, 1, 1, 300, zdr=False),          # fails zero-data-retention
            self._binding(P.AZURE, 1, 1, 2500),                  # too slow
            self._binding(P.GCP, 1, 1, 300, healthy=False),      # unhealthy
            self._binding(P.OCI, 9, 9, 300, quota_remaining=0),  # out of quota
        ])
        with self.assertRaises(self.m.RoutingException):
            router.resolve_candidates(self._request())
        relaxed = router.resolve_candidates(self._request(zero_data_retention=False))
        self.assertEqual([b.provider for b in relaxed], [P.AWS])

    def test_an_unknown_role_is_refused(self):
        m = self.m
        request = m.UnifiedInferenceRequest(model_role="nope", messages=[m.Message(role="user", content="x")])
        with self.assertRaises(m.RoutingException):
            self._router([]).resolve_candidates(request)

    def test_a_failing_provider_fails_over_to_the_next(self):
        m, P = self.m, self.m.Provider
        calls = []

        class Adapter:
            def __init__(self, provider, fail):
                self.provider, self.fail = provider, fail

            async def invoke(self, req, binding):
                calls.append(binding.provider)
                if self.fail:
                    raise RuntimeError("upstream 500")
                return m.UnifiedInferenceResponse(
                    content="ok", finish_reason="stop",
                    usage=m.Usage(input_tokens=1, output_tokens=1, computed_cost_usd=0.0),
                    resolved_binding=m.ResolvedBinding(provider=binding.provider, model_id=binding.model_id,
                                                       region=binding.region))

        cheap, dear = self._binding(P.GCP, 1, 1, 300), self._binding(P.AWS, 5, 5, 300)
        router = self._router([cheap, dear], {P.GCP: Adapter(P.GCP, True), P.AWS: Adapter(P.AWS, False)})
        response = _run(router.execute_route(self._request(cost_weight=1.0)))
        self.assertEqual(calls, [P.GCP, P.AWS])
        self.assertEqual(response.resolved_binding.provider, P.AWS)
        self.assertFalse(cheap.healthy)

    def test_when_every_provider_fails_the_cause_is_chained(self):
        m, P = self.m, self.m.Provider

        class Down:
            provider = P.AWS

            async def invoke(self, req, binding):
                raise RuntimeError("upstream 500")

        router = self._router([self._binding(P.AWS, 1, 1, 300)], {P.AWS: Down()})
        with self.assertRaises(m.RoutingException) as caught:
            _run(router.execute_route(self._request()))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)


# --------------------------------------------------------------------------- #
#  Chapter 2
# --------------------------------------------------------------------------- #
@needs_pydantic
class Listing_2_1_ResponseNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch02/listing-2.1-response-normalizer.py")

    def test_bedrock_shape(self):
        m = self.m
        raw = {"output": {"message": {"content": [
                   {"text": "Hello "}, {"text": "world"},
                   {"toolUse": {"toolUseId": "t1", "name": "lookup", "input": {"q": "x"}}}]}},
               "stopReason": "tool_use", "usage": {"inputTokens": 11, "outputTokens": 7}}
        out = m.ResponseNormalizer.normalize_aws_bedrock(raw, "claude")
        self.assertEqual((out.provider, out.content), ("aws", "Hello world"))
        self.assertEqual(out.finish_reason, m.FinishReason.TOOL_CALL)
        self.assertEqual((out.tool_calls[0].id, out.tool_calls[0].name, out.tool_calls[0].arguments),
                         ("t1", "lookup", {"q": "x"}))
        self.assertEqual((out.usage.input_tokens, out.usage.output_tokens), (11, 7))

    def test_azure_shape_with_reasoning_and_cached_tokens(self):
        m = self.m
        raw = {"choices": [{"finish_reason": "stop", "message": {
                   "content": "answer",
                   "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                         "completion_tokens_details": {"reasoning_tokens": 25},
                         "prompt_tokens_details": {"cached_tokens": 60}}}
        out = m.ResponseNormalizer.normalize_azure_openai(raw, "gpt")
        self.assertEqual(out.content, "answer")
        self.assertEqual(out.tool_calls[0].arguments, {"a": 1})
        self.assertEqual(out.usage.reasoning_tokens, 25)
        self.assertEqual(out.usage.cached_input_tokens, 60)
        self.assertEqual(out.usage.total_billable_tokens, 165)   # reasoning tokens are billed

    def test_azure_null_content_becomes_empty_text(self):
        raw = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None}}]}
        self.assertEqual(self.m.ResponseNormalizer.normalize_azure_openai(raw, "gpt").content, "")

    def test_vertex_shape_infers_a_tool_call(self):
        m = self.m
        raw = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
                   {"text": "calling"}, {"functionCall": {"name": "lookup", "args": {"q": "x"}}}]}}],
               "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3, "thoughtsTokenCount": 2}}
        out = m.ResponseNormalizer.normalize_gcp_vertex(raw, "gemini")
        self.assertEqual(out.finish_reason, m.FinishReason.TOOL_CALL)
        self.assertEqual(out.tool_calls[0].name, "lookup")
        self.assertEqual(out.usage.reasoning_tokens, 2)

    def test_truncation_never_collapses_into_stop(self):
        N, F = self.m.ResponseNormalizer, self.m.FinishReason
        cases = [
            N.normalize_aws_bedrock({"stopReason": "max_tokens"}, "x"),
            N.normalize_azure_openai({"choices": [{"finish_reason": "length", "message": {}}]}, "x"),
            N.normalize_gcp_vertex({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "x"),
        ]
        for out in cases:
            self.assertEqual(out.finish_reason, F.TRUNCATED)
            self.assertTrue(out.was_truncated)

    def test_content_filters_map_to_one_reason(self):
        N, F = self.m.ResponseNormalizer, self.m.FinishReason
        self.assertEqual(N.normalize_aws_bedrock({"stopReason": "content_filtered"}, "x").finish_reason,
                         F.CONTENT_FILTER)
        self.assertEqual(N.normalize_gcp_vertex({"candidates": [{"finishReason": "SAFETY"}]}, "x").finish_reason,
                         F.CONTENT_FILTER)

    def test_empty_payloads_do_not_raise(self):
        N = self.m.ResponseNormalizer
        for out in (N.normalize_aws_bedrock({}, "x"), N.normalize_azure_openai({}, "x"),
                    N.normalize_gcp_vertex({}, "x")):
            self.assertEqual(out.content, "")


# --------------------------------------------------------------------------- #
#  Chapter 3
# --------------------------------------------------------------------------- #
class Listing_3_1_CircuitBreaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch03/listing-3.1-sse-proxy-circuit-breaker.py")

    def _breaker(self, **kw):
        return self.m.EndpointCircuitBreaker(failure_threshold=2, cooldown_seconds=0.02, **kw)

    def test_it_opens_at_the_failure_threshold(self):
        breaker = self._breaker()
        breaker.record_failure("connectivity")
        self.assertTrue(breaker.can_execute())
        breaker.record_failure("connectivity")
        self.assertFalse(breaker.can_execute())

    def test_rate_limits_and_content_filters_do_not_trip_it(self):
        breaker = self._breaker()
        for _ in range(10):
            breaker.record_failure("rate_limit")
            breaker.record_failure("content_filter")
        self.assertTrue(breaker.can_execute())

    def test_it_half_opens_after_the_cooldown_and_limits_probes(self):
        breaker = self._breaker()
        breaker.record_failure("connectivity")
        breaker.record_failure("connectivity")
        time.sleep(0.03)
        self.assertTrue(breaker.can_execute())      # probe 1
        self.assertTrue(breaker.can_execute())      # probe 2
        self.assertFalse(breaker.can_execute())     # probe budget spent

    def test_a_successful_probe_closes_it(self):
        breaker = self._breaker()
        breaker.record_failure("connectivity")
        breaker.record_failure("connectivity")
        time.sleep(0.03)
        breaker.can_execute()
        breaker.record_success()
        self.assertEqual(breaker._state, self.m.BreakerState.CLOSED)
        self.assertTrue(all(breaker.can_execute() for _ in range(5)))

    def test_a_failed_probe_reopens_it(self):
        breaker = self._breaker()
        breaker.record_failure("connectivity")
        breaker.record_failure("connectivity")
        time.sleep(0.03)
        breaker.can_execute()
        breaker.record_failure("connectivity")
        self.assertEqual(breaker._state, self.m.BreakerState.OPEN)

    def test_the_router_prefers_the_better_score_and_skips_open_endpoints(self):
        m = self.m
        fast = m.ManagedEndpoint("fast", "azure", "x", cost_per_1k_tokens=0.004, ewma_ttft_seconds=0.2)
        slow = m.ManagedEndpoint("slow", "aws", "x", cost_per_1k_tokens=0.004, ewma_ttft_seconds=0.9)
        router = m.LatencyAwareRouter({"chat": [slow, fast]})
        self.assertEqual(router.select_best_endpoint("chat").endpoint_id, "fast")
        for _ in range(5):
            fast.circuit_breaker.record_failure("connectivity")
        self.assertEqual(router.select_best_endpoint("chat").endpoint_id, "slow")

    def test_no_available_endpoint_is_a_503(self):
        m = self.m
        with self.assertRaises(Exception) as caught:
            m.LatencyAwareRouter({}).select_best_endpoint("chat")
        self.assertEqual(getattr(caught.exception, "status_code", None), 503)

    def test_time_to_first_token_is_smoothed(self):
        endpoint = self.m.ManagedEndpoint("e", "aws", "x", 0.001, ewma_ttft_seconds=0.5, alpha=0.2)
        endpoint.update_ttft(1.0)
        self.assertAlmostEqual(endpoint.ewma_ttft_seconds, 0.6)

    @unittest.expectedFailure
    def test_a_tripped_endpoint_recovers_while_a_healthy_peer_exists(self):
        # KNOWN DEFECT, found 2026-11-13. `select_best_endpoint` calls
        # `can_execute()` on every candidate to build its list, and for a
        # half-open breaker that call spends a probe. A half-open endpoint
        # scores 100 worse than a closed one, so while a healthy peer exists it
        # is never chosen: its two probes are spent by two selections that went
        # elsewhere, no request ever reaches it to report success, and it stays
        # half-open with no probes for good. Eligibility needs to be checked
        # without side effects, and the probe spent only on the endpoint chosen.
        m = self.m
        tripped = m.ManagedEndpoint("a", "azure", "x", 0.002,
                                    circuit_breaker=m.EndpointCircuitBreaker(failure_threshold=2,
                                                                             cooldown_seconds=0.02))
        healthy = m.ManagedEndpoint("b", "aws", "x", 0.004)
        router = m.LatencyAwareRouter({"chat": [tripped, healthy]})
        tripped.circuit_breaker.record_failure("connectivity")
        tripped.circuit_breaker.record_failure("connectivity")
        time.sleep(0.03)
        for _ in range(10):                                   # traffic continues, all to `b`
            router.select_best_endpoint("chat")
        self.assertTrue(tripped.circuit_breaker.can_execute(),
                        "the endpoint can never be probed again")

    def test_provider_frames_become_unified_frames(self):
        m = self.m
        cases = {
            "azure": [b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
                      b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'],
            "aws": [b'data: {"delta":{"text":"Hel"}}\n\n', b'data: {"delta":{"text":"lo"}}\n\n'],
            "gcp": [b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n',
                    b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n\n'],
        }
        for provider, frames in cases.items():
            async def upstream(frames=frames):
                for frame in frames:
                    yield frame
                yield b"data: [DONE]\n\n"

            async def run(provider=provider, upstream=upstream):
                return [f async for f in m.transform_provider_stream(upstream(), provider)]

            out = _run(run())
            self.assertEqual(out[-1], b"data: [DONE]\n\n")
            text = "".join(json.loads(f.decode()[6:])["delta"] for f in out[:-1])
            self.assertEqual(text, "Hello", provider)

    def test_an_event_split_across_network_chunks_is_reassembled(self):
        m = self.m

        async def upstream():
            yield b'data: {"delta":{"te'
            yield b'xt":"whole"}}\n\ndata: [DONE]\n\n'

        async def run():
            return [f async for f in m.transform_provider_stream(upstream(), "aws")]

        self.assertEqual(_run(run())[0], b'data: {"delta": "whole"}\n\n')

    def test_a_malformed_frame_is_skipped(self):
        m = self.m

        async def upstream():
            yield b"data: {broken\n\n"
            yield b'data: {"delta":{"text":"ok"}}\n\n'

        async def run():
            return [f async for f in m.transform_provider_stream(upstream(), "aws")]

        self.assertEqual(_run(run()), [b'data: {"delta": "ok"}\n\n'])


# --------------------------------------------------------------------------- #
#  Chapter 4  (the streaming window is covered in test_redaction_window.py)
# --------------------------------------------------------------------------- #
@needs_pydantic
class Listing_4_1_RedactorCacheToolGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch04/listing-4.1-redactor-cache-tool-gate.py")

    def test_whole_text_redaction(self):
        out = self.m.apply_redaction("ssn 123-45-6789, card 4111111111111111, key AKIAIOSFODNN7EXAMPLE.")
        self.assertEqual(out, "ssn [REDACTED-SSN], card [REDACTED-CREDIT_CARD], key [REDACTED-API_KEY].")

    def test_a_digit_run_failing_luhn_survives(self):
        self.assertEqual(self.m.apply_redaction("order 4111111111111112."), "order 4111111111111112.")

    def test_luhn(self):
        self.assertTrue(self.m.verify_luhn_checksum("4111 1111 1111 1111"))
        self.assertFalse(self.m.verify_luhn_checksum("4111 1111 1111 1112"))
        self.assertFalse(self.m.verify_luhn_checksum("1234"))

    def test_cache_key_ignores_case_and_spacing_but_not_scope(self):
        m = self.m
        key = m.CacheScopeKey(tenant_id="acme", model_role="chat")
        self.assertEqual(key.compute_exact_hash("What  is\tTTFT?"), key.compute_exact_hash("what is ttft?"))
        other_tenant = m.CacheScopeKey(tenant_id="globex", model_role="chat")
        other_region = m.CacheScopeKey(tenant_id="acme", model_role="chat", residency="EU")
        self.assertNotEqual(key.compute_exact_hash("q"), other_tenant.compute_exact_hash("q"))
        self.assertNotEqual(key.compute_exact_hash("q"), other_region.compute_exact_hash("q"))

    def test_a_completion_cached_for_one_tenant_is_not_served_to_another(self):
        m = self.m
        cache = m.ProxySemanticCache(m.MockVectorCacheStore())
        acme = m.CacheScopeKey(tenant_id="acme", model_role="chat")
        globex = m.CacheScopeKey(tenant_id="globex", model_role="chat")
        self.assertIsNone(cache.resolve_cache(acme, "what is our margin?"))
        cache.store_completion(acme, "what is our margin?", "41%")
        self.assertEqual(cache.resolve_cache(acme, "What is our margin?"), "41%")
        self.assertIsNone(cache.resolve_cache(globex, "what is our margin?"))

    def _gate(self):
        m = self.m
        from pydantic import BaseModel

        class Lookup(BaseModel):
            account_id: str

        class Refund(BaseModel):
            account_id: str
            amount_usd: float

        def spec(name, tier, model):
            return m.ToolDefinitionSpec(name=name, risk_tier=tier, schema_model=model)

        return m.ToolExecutionGate({
            "lookup": spec("lookup", m.ToolRiskTier.T0_READ, Lookup),
            "refund": spec("refund", m.ToolRiskTier.T2_SENSITIVE, Refund),
            "wire": spec("wire", m.ToolRiskTier.T3_CRITICAL, Refund),
        })

    def _call(self, name, **params):
        return self.m.ValidatedToolCall(tool_name=name, parameters=params)

    def test_a_read_tool_needs_no_approval(self):
        self.assertTrue(self._gate().evaluate_and_authorize(self._call("lookup", account_id="a1"), []))

    def test_a_sensitive_tool_needs_one_approval_and_a_critical_tool_two(self):
        gate = self._gate()
        refund = self._call("refund", account_id="a1", amount_usd=20.0)
        wire = self._call("wire", account_id="a1", amount_usd=20.0)
        self.assertFalse(gate.evaluate_and_authorize(refund, []))
        self.assertTrue(gate.evaluate_and_authorize(refund, ["manager"]))
        self.assertFalse(gate.evaluate_and_authorize(wire, ["manager"]))
        self.assertTrue(gate.evaluate_and_authorize(wire, ["manager", "controller"]))

    def test_an_unregistered_tool_is_refused(self):
        with self.assertRaises(PermissionError):
            self._gate().evaluate_and_authorize(self._call("drop_table"), ["a", "b"])

    def test_parameters_are_validated_before_any_approval_counts(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._gate().evaluate_and_authorize(self._call("refund", account_id="a1"), ["manager"])


# --------------------------------------------------------------------------- #
#  Chapter 5
# --------------------------------------------------------------------------- #
@needs_pydantic
class Listing_5_1_FinancialAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch05/listing-5.1-financial-accounting-vbe.py")

    def _engine(self, included=0.0, on_demand=10.0, base=6000.0, overage=4.0):
        m = self.m
        return m.FinancialOptimizationEngine(m.InfrastructureCostParameters(
            on_demand_rate_per_1m=on_demand, provisioned_monthly_base_fee=base,
            provisioned_included_tokens_1m=included, provisioned_overage_rate_per_1m=overage))

    def test_break_even_with_no_included_volume(self):
        self.assertAlmostEqual(self._engine().compute_break_even_volume(), 1000.0)

    def test_at_break_even_the_two_modes_cost_the_same(self):
        engine = self._engine()
        v_be = engine.compute_break_even_volume()
        self.assertAlmostEqual(v_be * 10.0, 6000.0 + v_be * 4.0)

    def test_provisioned_never_wins_when_overage_matches_on_demand(self):
        self.assertEqual(self._engine(overage=10.0).compute_break_even_volume(), float("inf"))

    def test_recommendation_either_side_of_break_even(self):
        engine = self._engine()
        below, above = engine.analyze_cost_efficiency(400), engine.analyze_cost_efficiency(1500)
        self.assertEqual(below.recommended_mode, "ON_DEMAND_APIS")
        self.assertEqual(above.recommended_mode, "PROVISIONED_CAPACITY")
        self.assertAlmostEqual(above.projected_monthly_savings_usd, 15000 - (6000 + 1500 * 4))

    @unittest.expectedFailure
    def test_the_recommended_mode_is_the_cheaper_one_when_volume_is_included(self):
        # KNOWN DEFECT, found 2026-11-13. `compute_break_even_volume` divides the
        # base fee by the rate difference and ignores `provisioned_included_
        # tokens_1m`. With 500M tokens included, provisioned capacity becomes
        # cheaper at 667M tokens, and the listing reports break-even at 1,000M.
        # Between the two it recommends on-demand while pricing provisioned as
        # cheaper, and reports the loss as a saving. The equation is exact only
        # when nothing is included. (`reference/crossover_calculator.py`, the
        # Chapter 10 module, does not have this fault.)
        engine = self._engine(included=500.0)
        for volume in (700, 800, 900):
            on_demand = volume * 10.0
            provisioned = 6000.0 + (volume - 500.0) * 4.0
            cheaper = "PROVISIONED_CAPACITY" if provisioned < on_demand else "ON_DEMAND_APIS"
            self.assertEqual(engine.analyze_cost_efficiency(volume).recommended_mode, cheaper, volume)

    def test_hybrid_routing_fills_provisioned_capacity_then_bursts(self):
        m = self.m
        tracker = m.ProvisionedCapacityTracker(total_allocated_ptu_capacity=2, currently_active_ptu_utilization=0)
        router = m.HybridCapacityRouter(tracker)
        tiers = [router.resolve_execution_target()["target_tier"] for _ in range(3)]
        self.assertEqual(tiers, ["PROVISIONED_CAPACITY", "PROVISIONED_CAPACITY", "ON_DEMAND_OVERFLOW"])
        tracker.release_slot()
        self.assertEqual(router.resolve_execution_target()["target_tier"], "PROVISIONED_CAPACITY")

    def test_releasing_an_empty_tracker_does_not_go_negative(self):
        tracker = self.m.ProvisionedCapacityTracker(1, 0)
        tracker.release_slot()
        self.assertEqual(tracker.currently_active_ptu_utilization, 0)


# --------------------------------------------------------------------------- #
#  Chapter 6
# --------------------------------------------------------------------------- #
@needs_pydantic
class Listing_6_1_TelemetryEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch06/listing-6.1-unified-telemetry-event.py")

    def _event(self, **overrides):
        fields = dict(tenant="acme", department="support", feature="chat", user="u1", provider="aws",
                      model="m", region="us-east-1", prompt_tokens=1_000_000, completion_tokens=500_000,
                      cached_tokens=200_000, reasoning_tokens=100_000, retrieval_ops=3, compute_ms=2000,
                      price_book_version="unpriced")
        fields.update(overrides)
        return self.m.TelemetryEvent(**fields)

    def test_cost_is_exact_decimal_arithmetic_over_every_measure(self):
        book = self.m.PriceBook(version="2026-11", usd_per_m_input=Decimal("3"), usd_per_m_output=Decimal("15"),
                                usd_per_m_cached=Decimal("0.30"), usd_per_m_reasoning=Decimal("15"),
                                usd_per_retrieval_op=Decimal("0.002"), usd_per_compute_s=Decimal("0.01"))
        event = self._event().price(book)
        # 3.00 + 7.50 + 0.06 + 1.50 + 0.006 + 0.02
        self.assertEqual(event.cost_usd, Decimal("12.086"))
        self.assertIsInstance(event.cost_usd, Decimal)

    def test_pricing_stamps_the_price_book_version(self):
        book = self.m.PriceBook(version="2026-11", usd_per_m_input=Decimal("1"), usd_per_m_output=Decimal("1"))
        self.assertEqual(self._event().price(book).price_book_version, "2026-11")

    def test_an_unpriced_event_costs_nothing(self):
        self.assertEqual(self._event().cost_usd, Decimal("0"))

    def test_cost_is_part_of_the_serialized_event(self):
        book = self.m.PriceBook(version="v", usd_per_m_input=Decimal("2"), usd_per_m_output=Decimal("0"))
        dumped = self._event(completion_tokens=0, cached_tokens=0, reasoning_tokens=0,
                             retrieval_ops=0, compute_ms=0).price(book).model_dump()
        self.assertEqual(dumped["cost_usd"], Decimal("2"))

    def test_a_stray_field_is_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._event(colour="blue")

    def test_every_event_gets_its_own_id(self):
        self.assertNotEqual(self._event().event_id, self._event().event_id)


class _ScriptedRedis:
    """Stands in for the Redis client. `evalsha` applies the same arithmetic as
    the listing's Lua token bucket, in Python. The Lua itself does not run."""

    def __init__(self):
        self.buckets, self.loads = {}, 0

    async def script_load(self, script):
        self.loads += 1
        self.script = script
        return "sha1"

    async def evalsha(self, sha, numkeys, key, rate, burst, cost, now):
        tokens, last = self.buckets.get(key, (burst, now))
        tokens = min(burst, tokens + (now - last) * rate)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
        self.buckets[key] = (tokens, now)
        return [1 if allowed else 0, tokens]


class Listing_6_3_BudgetEnforcement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch06/listing-6.3-redis-budget-enforcement.py")

    class _Event:
        tenant, feature = "acme", "chat"

    def test_the_script_is_loaded_once_and_reused(self):
        client = _ScriptedRedis()
        gate = self.m.BudgetGate(client)

        async def run():
            return [await gate.admit("k", rate=0.0, burst=100.0, est_cost=10.0) for _ in range(3)]

        results = _run(run())
        self.assertEqual(client.loads, 1)
        self.assertEqual([allowed for allowed, _ in results], [True, True, True])
        self.assertEqual(results[-1][1], 70.0)

    def test_the_lua_script_names_what_the_python_passes(self):
        script = self.m._BUCKET_LUA
        for token in ("KEYS[1]", "ARGV[1]", "ARGV[2]", "ARGV[3]", "ARGV[4]", "HMGET", "HMSET", "EXPIRE"):
            self.assertIn(token, script)

    def test_a_request_past_the_hard_cap_is_refused(self):
        m = self.m
        gate, sent = m.BudgetGate(_ScriptedRedis()), []

        async def emit(event):
            sent.append(event)

        async def run():
            await m.enforce_and_emit(gate, emit, self._Event(), soft=100.0, hard=100.0, est_cost=60.0)
            await m.enforce_and_emit(gate, emit, self._Event(), soft=100.0, hard=100.0, est_cost=60.0)

        with self.assertRaises(m.BudgetExhausted) as caught:
            _run(run())
        self.assertEqual((caught.exception.tenant, caught.exception.reset_hint), ("acme", "hourly"))
        self.assertEqual(len(sent), 1)

    def test_budgets_are_keyed_by_tenant_and_feature(self):
        m = self.m
        client = _ScriptedRedis()

        async def emit(event):
            pass

        _run(m.enforce_and_emit(m.BudgetGate(client), emit, self._Event(), soft=100.0, hard=100.0, est_cost=1.0))
        self.assertEqual(list(client.buckets), ["budget:acme:chat"])

    @unittest.expectedFailure
    def test_crossing_the_soft_cap_notifies_and_still_serves(self):
        # KNOWN DEFECT, found 2026-11-13. The soft-cap branch calls
        # `alert_owner(...)`, which the listing neither defines nor imports. The
        # first request to cross the soft cap raises NameError, so the path the
        # comment describes as "notify only" fails the request it meant to
        # serve. The listing needs the notifier passed in, as `emit` is.
        m = self.m
        gate, sent = m.BudgetGate(_ScriptedRedis()), []

        async def emit(event):
            sent.append(event)

        async def run():
            for _ in range(9):          # 90 of 100 spent: past the soft cap of 80, under the hard cap
                await m.enforce_and_emit(gate, emit, self._Event(), soft=80.0, hard=100.0, est_cost=10.0)

        _run(run())
        self.assertEqual(len(sent), 9)


# --------------------------------------------------------------------------- #
#  Chapter 7
# --------------------------------------------------------------------------- #
@needs_otel
class Listing_7_1_OtelInstrumentation(unittest.TestCase):
    """Runs against the OpenTelemetry API. With no SDK configured the API records
    into no-op instruments, so this proves the calls are well-formed, not that
    anything is exported."""

    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch07/listing-7.1-otel-genai-instrumentation.py")

    def test_attributes_use_the_genai_semantic_convention_keys(self):
        attrs = self.m.GenAISpanAttributes("aws", "claude", "chat", finish_reasons=("stop",)).as_otel()
        self.assertEqual(attrs["gen_ai.provider.name"], "aws")
        self.assertEqual(attrs["gen_ai.response.model"], "claude")        # falls back to the request model
        self.assertEqual(attrs["gen_ai.response.finish_reasons"], ["stop"])
        self.assertTrue(all(key.startswith("gen_ai.") for key in attrs))

    def test_recording_an_inference_does_not_raise(self):
        attrs = self.m.GenAISpanAttributes("azure", "gpt", "chat", response_model="gpt-2026")
        self.m.record_inference(attrs, in_tok=120, out_tok=48, ttft_s=0.31, dur_s=1.9)


class Listing_7_2_ProgressionOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch07/listing-7.2-progression-orchestrator.py")

    def setUp(self):
        m = self.m
        self.orchestrator = m.ProgressionOrchestrator(m.Gate(bake_seconds=3600.0))
        self.good = m.Metrics(p95_ttft_s=0.42, error_pct=0.3, quality=0.94, cogs_ratio=0.98)

    def _baked(self, candidate):
        candidate.healthy_since -= 4000
        return candidate

    def test_a_candidate_starts_in_shadow_with_no_live_traffic(self):
        candidate = self.m.Candidate("next")
        self.assertEqual(candidate.tier, self.m.Tier.SHADOW)
        self.assertEqual(self.orchestrator.traffic_share(candidate), 0.0)

    def test_healthy_metrics_do_not_promote_before_the_bake_window(self):
        candidate = self.orchestrator.evaluate(self.m.Candidate("next"), self.good)
        self.assertEqual(candidate.tier, self.m.Tier.SHADOW)

    def test_promotion_climbs_one_tier_per_bake_and_stops_at_primary(self):
        m, candidate = self.m, self.m.Candidate("next")
        shares = []
        for _ in range(5):
            self.orchestrator.evaluate(self._baked(candidate), self.good)
            shares.append(self.orchestrator.traffic_share(candidate))
        self.assertEqual(shares, [0.05, 0.50, 1.0, 1.0, 1.0])
        self.assertEqual(candidate.tier, m.Tier.PRIMARY)

    def test_a_breach_on_any_one_axis_fails_back_to_shadow(self):
        m = self.m
        breaches = [dict(p95_ttft_s=0.9), dict(error_pct=2.0), dict(quality=0.81), dict(cogs_ratio=1.3)]
        for breach in breaches:
            candidate = m.Candidate("next", tier=m.Tier.ACTIVE_SPLIT)
            metrics = m.Metrics(**{**dict(p95_ttft_s=0.42, error_pct=0.3, quality=0.94, cogs_ratio=0.98), **breach})
            self.orchestrator.evaluate(self._baked(candidate), metrics)
            self.assertEqual(candidate.tier, m.Tier.SHADOW, breach)

    def test_a_breach_restarts_the_bake_clock(self):
        m = self.m
        candidate = self._baked(m.Candidate("next", tier=m.Tier.CANARY))
        bad = m.Metrics(p95_ttft_s=0.42, error_pct=0.3, quality=0.5, cogs_ratio=0.98)
        self.orchestrator.evaluate(candidate, bad)
        self.orchestrator.evaluate(candidate, self.good)
        self.assertEqual(candidate.tier, m.Tier.SHADOW)


# --------------------------------------------------------------------------- #
#  Chapter 8
# --------------------------------------------------------------------------- #
class Listing_8_2_TamperEvidentAuditLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch08/listing-8.2-tamper-evident-audit-log.py")

    def _log(self, entries=3):
        m = self.m
        log = m.AuditLog(hmac_key=b"k" * 32)
        for i in range(entries):
            log.append(m.salted_hash(f"prompt {i}", b"salt"), m.salted_hash(f"answer {i}", b"salt"),
                       {"pii_redacted": True, "residency": "EU"}, user="u-42", tenant="acme")
        return log

    def test_an_intact_chain_verifies(self):
        self.assertTrue(self._log().verify())
        self.assertTrue(self.m.AuditLog(hmac_key=b"k").verify())      # empty log

    def test_entries_are_chained_from_the_genesis_hash(self):
        log = self._log()
        self.assertEqual(log._chain[0].prev_hash, self.m.AuditLog.GENESIS)
        self.assertEqual(log._chain[2].prev_hash, log._chain[1].entry_hash)

    def test_the_log_stores_hashes_not_content(self):
        entry = self._log(1)._chain[0]
        self.assertNotIn("prompt 0", entry.body().decode())
        self.assertEqual(len(entry.input_hash), 64)

    def test_a_salt_changes_the_hash(self):
        self.assertNotEqual(self.m.salted_hash("x", b"a"), self.m.salted_hash("x", b"b"))

    def test_altering_a_past_entry_is_detected(self):
        log = self._log()
        log._chain[1].policy_results["residency"] = "US"
        self.assertFalse(log.verify())

    def test_removing_or_reordering_an_entry_is_detected(self):
        removed = self._log()
        del removed._chain[1]
        self.assertFalse(removed.verify())
        swapped = self._log()
        swapped._chain[0], swapped._chain[1] = swapped._chain[1], swapped._chain[0]
        self.assertFalse(swapped.verify())

    def test_a_chain_signed_with_another_key_does_not_verify(self):
        log = self._log()
        log._key = b"another-key"
        self.assertFalse(log.verify())


@needs_pydantic
class Listing_8_3_HashChainedAuditLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("chapters/ch08/listing-8.3-hash-chained-audit-ledger.py")

    KEY = b"hsm-derived-key-for-tests-only!!"

    def _payload(self, n=0):
        return self.m.AuditRecordPayload(
            event_id=f"e{n}", tenant_id="acme", model_role="chat", resolved_provider="aws",
            resolved_model_id="m", resolved_region="us-east-1", zero_data_retention=True,
            input_hash="a" * 64, output_hash="b" * 64, input_tokens=10 + n, output_tokens=5)

    def _chain(self, count=3):
        ledger = self.m.CryptographicAuditLedger(self.KEY)
        return [ledger.append_event(self._payload(i)) for i in range(count)]

    def test_each_entry_verifies_against_its_predecessor(self):
        entries = self._chain()
        verify = self.m.CryptographicAuditLedger.verify_entry
        self.assertTrue(verify(entries[1], entries[0].block_hash, self.KEY))
        self.assertTrue(verify(entries[2], entries[1].block_hash, self.KEY))

    def test_sequence_numbers_and_links_advance(self):
        entries = self._chain()
        self.assertEqual([e.sequence_number for e in entries], [1, 2, 3])
        self.assertEqual(entries[2].previous_block_hash, entries[1].block_hash)

    def test_an_altered_payload_fails_verification(self):
        entries = self._chain()
        entries[1].payload.input_tokens = 999_999
        self.assertFalse(self.m.CryptographicAuditLedger.verify_entry(entries[1], entries[0].block_hash, self.KEY))

    def test_the_wrong_predecessor_fails_verification(self):
        entries = self._chain()
        self.assertFalse(self.m.CryptographicAuditLedger.verify_entry(entries[2], entries[0].block_hash, self.KEY))

    def test_the_wrong_key_fails_verification(self):
        entries = self._chain()
        self.assertFalse(self.m.CryptographicAuditLedger.verify_entry(entries[1], entries[0].block_hash, b"other"))

    def test_a_forged_signature_fails_verification(self):
        entries = self._chain()
        entries[1].hmac_signature = "0" * 64
        self.assertFalse(self.m.CryptographicAuditLedger.verify_entry(entries[1], entries[0].block_hash, self.KEY))


# --------------------------------------------------------------------------- #
#  Chapter 10, telemetry module (the other four have their own test modules)
# --------------------------------------------------------------------------- #
@needs_otel
class Reference_OtelInstrumentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _listings.load("reference/otel_instrumentation.py")

    def _observation(self, **overrides):
        fields = dict(provider="gcp", request_model="gemini", response_model="gemini-2026", operation="chat",
                      tenant="acme", feature="search", input_tokens=900, output_tokens=300,
                      reasoning_tokens=50, ttft_s=0.4, duration_s=2.4, cost_usd=Decimal("0.0123"),
                      finish_reason="stop")
        fields.update(overrides)
        return self.m.InferenceObservation(**fields)

    def test_tokens_per_second_excludes_time_to_first_token(self):
        self.assertAlmostEqual(self._observation().tokens_per_second, 150.0)

    def test_a_zero_length_decode_window_does_not_divide_by_zero(self):
        self.assertGreater(self._observation(ttft_s=1.0, duration_s=1.0).tokens_per_second, 0)

    def test_metric_attributes_stay_low_cardinality(self):
        attrs = self.m._base_attributes(self._observation())
        self.assertEqual(set(attrs), {"gen_ai.provider.name", "gen_ai.request.model",
                                      "gen_ai.operation.name", "tenant", "feature"})

    def test_recording_does_not_raise(self):
        self.m.record_inference(self._observation())
        self.m.record_inference(self._observation(reasoning_tokens=0))


if __name__ == "__main__":
    unittest.main()
