# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

"""The non-Python listings, and the places two listings must agree.

A YAML contract, a SQL schema and three Terraform files cannot be executed here,
so they are checked for what can be checked without a cloud account: that they
parse, that their references resolve, and that each agrees with the Python
listing describing the same thing. Drift between the OpenAPI contract and the
request model, or between the telemetry event and the ledger table, is the
failure these exist to catch.

`terraform validate` is not run: it needs the Terraform binary and the provider
plugins. The Terraform checks here are structural only.
"""
import re
import unittest

import _listings

ROOT = _listings.ROOT
HAVE_YAML = _listings.have("yaml")
HAVE_PYDANTIC = _listings.have("pydantic")


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


@unittest.skipUnless(HAVE_YAML, "needs PyYAML")
class Listing_1_1_OpenApiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.doc = yaml.safe_load(_text("chapters/ch01/listing-1.1-gateway-contract.openapi.yaml"))
        cls.schemas = cls.doc["components"]["schemas"]

    def test_it_is_an_openapi_3_1_document(self):
        self.assertTrue(str(self.doc["openapi"]).startswith("3.1"))
        for key in ("info", "paths", "components"):
            self.assertIn(key, self.doc)

    def test_every_reference_resolves(self):
        def refs(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "$ref":
                        yield value
                    else:
                        yield from refs(value)
            elif isinstance(node, list):
                for value in node:
                    yield from refs(value)

        for ref in set(refs(self.doc)):
            node = self.doc
            for part in ref.lstrip("#/").split("/"):
                self.assertIn(part, node, f"unresolved {ref}")
                node = node[part]

    def test_every_required_property_is_declared(self):
        for name, schema in self.schemas.items():
            for required in schema.get("required", []):
                self.assertIn(required, schema.get("properties", {}), f"{name}.{required}")

    def test_every_operation_declares_its_responses(self):
        for path, operations in self.doc["paths"].items():
            for method, operation in operations.items():
                self.assertIn("200", operation["responses"], f"{method} {path}")
                self.assertIn("operationId", operation)

    def test_the_request_names_a_role_and_no_provider(self):
        properties = set(self.schemas["UnifiedInferenceRequest"]["properties"])
        self.assertIn("model_role", properties)
        self.assertFalse({"provider", "model_id", "deployment"} & properties)

    @unittest.skipUnless(HAVE_PYDANTIC, "Listing 1.2 imports pydantic")
    def test_the_contract_and_the_python_models_declare_the_same_fields(self):
        models = _listings.load("chapters/ch01/listing-1.2-dynamic-router.py")
        for name in ("UnifiedInferenceRequest", "Message", "RoutingConstraints",
                     "UnifiedInferenceResponse", "Usage", "ResolvedBinding"):
            self.assertEqual(set(self.schemas[name]["properties"]),
                             set(getattr(models, name).model_fields), name)


@unittest.skipUnless(HAVE_YAML, "needs PyYAML")
class Listing_11_1_MetricRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.metrics = yaml.safe_load(_text("chapters/ch11/listing-11.1-metric-registry.yaml"))

    def test_every_signal_has_an_owner_a_query_and_a_healthy_band(self):
        self.assertTrue(self.metrics)
        for metric in self.metrics:
            for key in ("id", "source", "query", "unit", "healthy", "alarm", "surfaces", "owner"):
                self.assertIn(key, metric, metric.get("id"))
            self.assertTrue(metric["surfaces"])

    def test_signal_ids_are_unique(self):
        ids = [metric["id"] for metric in self.metrics]
        self.assertEqual(len(ids), len(set(ids)))


class Listing_6_2_ClickHouseLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = _text("chapters/ch06/listing-6.2-clickhouse-ledger.sql")

    def _columns(self):
        body = self.sql.split("CREATE TABLE metering.events", 1)[1].split(")\nENGINE", 1)[0]
        return [line.split()[0] for line in body.strip().strip("(").strip().splitlines() if line.strip()]

    def test_the_ledger_is_partitioned_ordered_and_tiered(self):
        for clause in ("ENGINE = MergeTree", "PARTITION BY", "ORDER BY (tenant, feature, ts)", "TTL"):
            self.assertIn(clause, self.sql)

    def test_parentheses_balance(self):
        code = re.sub(r"--.*", "", self.sql)
        self.assertEqual(code.count("("), code.count(")"))

    def test_the_rollup_reads_only_columns_the_ledger_has(self):
        columns = set(self._columns())
        view = self.sql.split("AS SELECT", 1)[1].split("FROM metering.events", 1)[0]
        used = set(re.findall(r"\b([a-z_]+)\b", re.sub(r"\bAS\s+\w+", "", view))) - {"sum", "count", "tostartofhour"}
        self.assertFalse(used - columns, used - columns)

    @unittest.skipUnless(HAVE_PYDANTIC, "Listing 6.1 imports pydantic")
    def test_the_table_has_a_column_for_every_field_of_the_telemetry_event(self):
        event = _listings.load("chapters/ch06/listing-6.1-unified-telemetry-event.py").TelemetryEvent
        fields = set(event.model_fields) | set(event.model_computed_fields)
        self.assertEqual(set(self._columns()), fields)


class TerraformStructure(unittest.TestCase):
    FILES = ("chapters/ch08/listing-8.1-oidc-web-identity-role.tf", "reference/privatelink.tf",
             "extensions/privatelink_oci.tf")

    def _code(self, path):
        code = re.sub(r"#.*", "", _text(path))
        return re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)

    def test_delimiters_balance(self):
        for path in self.FILES:
            code = self._code(path)
            for left, right in ("{}", "[]", "()"):
                self.assertEqual(code.count(left), code.count(right), f"{path}: {left}{right}")

    def _variables(self, path):
        source = _text(path)
        declared = set(re.findall(r'^variable\s+"([^"]+)"', source, flags=re.M))
        used = set(re.findall(r"\bvar\.([A-Za-z0-9_]+)", source))
        return declared, used

    def test_the_private_ingress_files_declare_every_variable_they_use(self):
        for path in ("reference/privatelink.tf", "extensions/privatelink_oci.tf"):
            declared, used = self._variables(path)
            self.assertFalse(used - declared, f"{path}: undeclared {sorted(used - declared)}")

    @unittest.expectedFailure
    def test_listing_8_1_uses_only_variables_declared_in_the_repository(self):
        # KNOWN GAP, found 2026-11-13. Listing 8.1 is an excerpt: it declares no
        # variables. Four of the five it uses (`region`, `vpc_id`,
        # `private_subnet_ids`, `endpoints_only_sg`) are declared in
        # `reference/privatelink.tf`, so the two apply together. The fifth,
        # `oidc_thumbprint`, is declared nowhere, so the set does not pass
        # `terraform validate` until a reader adds it.
        declared = set()
        for path in self.FILES:
            declared |= self._variables(path)[0]
        used = self._variables("chapters/ch08/listing-8.1-oidc-web-identity-role.tf")[1]
        self.assertFalse(used - declared, f"undeclared {sorted(used - declared)}")

    def test_no_resource_address_is_declared_twice(self):
        for path in self.FILES:
            addresses = re.findall(r'^(?:resource|data)\s+"([^"]+)"\s+"([^"]+)"', _text(path), flags=re.M)
            self.assertEqual(len(addresses), len(set(addresses)), path)

    def test_private_ingress_covers_all_four_providers(self):
        combined = _text("reference/privatelink.tf") + _text("extensions/privatelink_oci.tf")
        for resource in ("aws_vpc_endpoint", "azurerm_private_endpoint", "google_compute_forwarding_rule",
                         "oci_generative_ai_generative_ai_private_endpoint"):
            self.assertIn(resource, combined)


if __name__ == "__main__":
    unittest.main()
