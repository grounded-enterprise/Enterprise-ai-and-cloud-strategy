# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""Contract tests for the unified request/response schema.

The schema is the seam the whole gateway is built on, so its guarantees are
worth pinning: a role is never a model id, a truncated answer is never
reported as a complete one, and a forced tool must actually exist.

Requires `pydantic`. Skipped when it is absent.
"""
import unittest

import _paths  # noqa: F401  (import for the sys.path side effect)

try:
    import pydantic  # noqa: F401
    from pydantic import ValidationError

    from unified_schema import (
        FinishReason,
        Message,
        Provider,
        ReasoningConfig,
        ReasoningEffort,
        ResolvedBinding,
        ResponseFormat,
        Role,
        RoutingConstraints,
        TokenUsage,
        ToolCall,
        ToolChoice,
        ToolParameter,
        ToolSpec,
        UnifiedRequest,
        UnifiedResponse,
    )

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False


def _message(text: str = "hello") -> "Message":
    return Message(role=Role.USER, content=text)


def _tool(name: str = "get_invoice") -> "ToolSpec":
    return ToolSpec(
        name=name,
        description="Fetch an invoice by id.",
        parameters={"invoice_id": ToolParameter(type="string")},
        required=["invoice_id"],
    )


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class RequestContract(unittest.TestCase):
    def test_a_minimal_request_validates(self):
        req = UnifiedRequest(model_role="reasoning", messages=[_message()])
        self.assertEqual(req.model_role, "reasoning")
        self.assertFalse(req.stream)

    def test_messages_must_not_be_empty(self):
        with self.assertRaises(ValidationError):
            UnifiedRequest(model_role="reasoning", messages=[])

    def test_model_role_must_not_be_empty(self):
        with self.assertRaises(ValidationError):
            UnifiedRequest(model_role="", messages=[_message()])

    def test_unknown_fields_are_rejected(self):
        # extra="forbid" is what stops a provider-specific field being smuggled
        # through the neutral schema and quietly creating a coupling.
        with self.assertRaises(ValidationError):
            UnifiedRequest(model_role="reasoning", messages=[_message()],
                           anthropic_beta="whatever")

    def test_defaults_are_conservative(self):
        req = UnifiedRequest(model_role="reasoning", messages=[_message()])
        self.assertTrue(req.routing_constraints.zero_data_retention)
        self.assertEqual(req.temperature, 0.2)
        self.assertIs(req.reasoning.effort, ReasoningEffort.NONE)

    def test_temperature_and_token_bounds_are_enforced(self):
        for field, value in [("temperature", 2.5), ("temperature", -0.1),
                             ("max_output_tokens", 0), ("max_output_tokens", 500_000)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                UnifiedRequest(model_role="r", messages=[_message()], **{field: value})


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class ToolContract(unittest.TestCase):
    def test_forced_tool_must_be_declared(self):
        with self.assertRaises(ValidationError):
            UnifiedRequest(
                model_role="reasoning",
                messages=[_message()],
                tools=[_tool("get_invoice")],
                tool_choice=ToolChoice(mode="required", forced_tool="get_refund"),
            )

    def test_forced_tool_that_is_declared_validates(self):
        req = UnifiedRequest(
            model_role="reasoning",
            messages=[_message()],
            tools=[_tool("get_invoice")],
            tool_choice=ToolChoice(mode="required", forced_tool="get_invoice"),
        )
        self.assertEqual(req.tool_choice.forced_tool, "get_invoice")

    def test_forcing_a_tool_requires_required_mode(self):
        with self.assertRaises(ValidationError):
            ToolChoice(mode="auto", forced_tool="get_invoice")

    def test_required_arguments_must_be_declared_parameters(self):
        with self.assertRaises(ValidationError):
            ToolSpec(name="t", description="d",
                     parameters={"a": ToolParameter(type="string")}, required=["b"])

    def test_tool_names_are_constrained(self):
        with self.assertRaises(ValidationError):
            ToolSpec(name="bad name!", description="d", parameters={})

    def test_tool_call_arguments_are_an_object_not_a_string(self):
        # Providers differ here; the neutral shape is always parsed.
        call = ToolCall(id="c1", name="get_invoice", arguments={"invoice_id": "INV-1"})
        self.assertIsInstance(call.arguments, dict)


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class ResponseContract(unittest.TestCase):
    def _response(self, finish: "FinishReason") -> "UnifiedResponse":
        return UnifiedResponse(
            content="answer",
            finish_reason=finish,
            resolved_binding=ResolvedBinding(provider=Provider.AWS,
                                             model_id="m", region="us-east-1"),
        )

    def test_truncated_is_distinguishable_from_stop(self):
        # Folding these together makes a client process a partial answer as
        # though it were complete. This is the single most consequential
        # normalization in the schema.
        self.assertTrue(self._response(FinishReason.TRUNCATED).is_truncated)
        self.assertFalse(self._response(FinishReason.STOP).is_truncated)

    def test_finish_reasons_cover_the_normalized_vocabulary(self):
        self.assertEqual(
            {r.value for r in FinishReason},
            {"stop", "truncated", "content_filter", "tool_call", "error"},
        )

    def test_provenance_is_mandatory(self):
        # A response you cannot attribute to an endpoint is not auditable.
        with self.assertRaises(ValidationError):
            UnifiedResponse(content="a", finish_reason=FinishReason.STOP)

    def test_all_four_providers_are_representable(self):
        self.assertEqual({p.value for p in Provider}, {"aws", "azure", "gcp", "oci"})


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class TokenAccounting(unittest.TestCase):
    """Cost attribution needs one definition of what is billable."""

    def test_reasoning_tokens_are_billable_output(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=25)
        self.assertEqual(usage.billable_output, 75)

    def test_total_counts_input_output_and_reasoning(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, reasoning_tokens=25)
        self.assertEqual(usage.total, 175)

    def test_cached_input_is_tracked_separately(self):
        # Cached input is priced differently; it must not silently inflate the
        # total the way a naive sum would.
        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_input_tokens=80)
        self.assertEqual(usage.total, 150)

    def test_usage_defaults_to_zero(self):
        usage = TokenUsage()
        self.assertEqual((usage.total, usage.billable_output), (0, 0))


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class RoutingAndFormat(unittest.TestCase):
    def test_zero_data_retention_defaults_to_on(self):
        self.assertTrue(RoutingConstraints().zero_data_retention)

    def test_cost_weight_is_a_fraction(self):
        with self.assertRaises(ValidationError):
            RoutingConstraints(cost_weight=1.5)

    def test_latency_target_must_be_positive(self):
        with self.assertRaises(ValidationError):
            RoutingConstraints(max_p95_latency_ms=0)

    def test_json_schema_format_requires_a_schema(self):
        with self.assertRaises(ValidationError):
            ResponseFormat(kind="json_schema")

    def test_json_schema_format_with_a_schema_validates(self):
        fmt = ResponseFormat(kind="json_schema", json_schema={"type": "object"})
        self.assertEqual(fmt.kind, "json_schema")

    def test_reasoning_budget_must_be_positive_when_set(self):
        with self.assertRaises(ValidationError):
            ReasoningConfig(effort=ReasoningEffort.HIGH, max_reasoning_tokens=0)


@unittest.skipUnless(HAVE_PYDANTIC, "requires `pydantic`")
class NeutralityInvariants(unittest.TestCase):
    """The property the book is named for."""

    def test_the_request_has_no_provider_field(self):
        # Provider appears only on the response, as resolved provenance. If a
        # caller could name a provider on the way in, the abstraction is gone.
        self.assertNotIn("provider", UnifiedRequest.model_fields)
        self.assertIn("model_role", UnifiedRequest.model_fields)

    def test_no_request_field_names_a_vendor(self):
        vendors = ("aws", "bedrock", "azure", "openai", "gcp", "vertex", "oci", "anthropic")
        for name in UnifiedRequest.model_fields:
            with self.subTest(field=name):
                self.assertFalse(any(v in name.lower() for v in vendors))

    def test_a_request_round_trips_through_json(self):
        req = UnifiedRequest(model_role="reasoning", messages=[_message()],
                             tools=[_tool()], stream=True)
        self.assertEqual(UnifiedRequest.model_validate_json(req.model_dump_json()), req)


if __name__ == "__main__":
    unittest.main()
