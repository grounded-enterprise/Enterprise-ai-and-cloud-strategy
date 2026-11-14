# Enterprise AI Architecture & Cloud Strategy — companion code

A provider-neutral inference gateway and the pieces around it: one request schema that four
hyperscalers can serve, a streaming core that redacts PII mid-frame, a calculator that tells
you when committed capacity beats on-demand, private connectivity, and telemetry that makes
cost and quality observable per request.

---

## Run something in the next sixty seconds

`crossover_calculator.py` has no dependencies at all. It answers the question the rest of the
system exists to make answerable: *at what volume does committed capacity beat paying per
token?*

```bash
python3 reference/crossover_calculator.py
```

That runs the built-in scenario — 50 M tokens/day mean, 20 M standard deviation, 30%
completions, $3/$15 per million in/out, $260 per provisioned unit-month — and prints
break-even volume, break-even utilization, volatility index, and the cost-optimal choice
across pure serverless, provision-to-peak, and baseline-plus-burst hybrid.

To use your own numbers:

```python
from crossover_calculator import (CrossoverCalculator, ServerlessPricing,
                                  ProvisionedPricing, VolumeMetrics)

calc = CrossoverCalculator(
    ServerlessPricing(usd_per_m_input=3.0, usd_per_m_output=15.0),
    ProvisionedPricing(usd_per_unit_month=260.0, m_tokens_day_per_unit=3.6),
)
decision = calc.evaluate(VolumeMetrics(mu_day=50.0, sigma_day=20.0,
                                       completion_fraction=0.30))
print(decision.as_dict())
```

`mu_day` and `sigma_day` come from your metering data — mean and standard deviation of daily
token volume. Everything else comes off a price list.

---

## Install

```bash
pip install -r requirements.txt
```

Only what you use:

| You want | You need |
|---|---|
| Capacity maths | nothing — standard library |
| Request/response schema | `pydantic` |
| Streaming gateway | `pydantic`, `fastapi` (plus an ASGI server, e.g. `uvicorn`) |
| Budget enforcement | `redis` |
| Telemetry | `opentelemetry-api`, `opentelemetry-sdk` |
| Private connectivity | Terraform or OpenTofu |
| Ledger | ClickHouse |
| Asymmetric audit signing | `cryptography` |

---

## How this tree is organized

Two halves, with different rules.

| Directory | Rule |
|---|---|
| `chapters/`, `reference/` | Machine-generated from the manuscript — **do not hand-edit**; changes are overwritten. Below the licence header, the code is byte-identical to the printed listing. Diff it against the page and you get nothing back, which is the point: what you read is what you run. |
| `extensions/`, `tests/` | Written for this repository. Where a printed listing has been deliberately taught in its simplest correct form, the deployable version lives here. |

`reference/` is Chapter 10's five modules under the names Table 10.1 prints;
`chapters/` is everything else, filed by the chapter that introduces it.

---

## Tests

```bash
python3 -m unittest discover -s tests -t tests
```

No installation required and no cloud credentials — the suite runs offline
against recorded shapes and deterministic inputs. `pip install -r
requirements-dev.txt` lifts the schema-test skips and lets the same suite run
under `pytest tests` if you prefer.

What it pins:

- **The capacity maths**, against the worked example printed in Chapter 5 —
  $3,640 committed, U_be 36.5%, V_t 0.40, $5,219.81 hybrid, 47.3% saving. Every
  intermediate is asserted, so a refactor that moves any of them fails loudly.
  Also the identity `V_be × price == C_fixed`, which is the check that catches
  a break-even equation carrying its unknown on both sides.
- **Redaction across frame boundaries** — an SSN split as `…123-45` / `-6789`
  must still be caught. This is the case that makes naive per-frame redaction
  wrong, and it is asserted both two-frame and character-by-character.
- **Delta extraction per provider**, which is the executable form of the
  response-shape tables in Chapter 2.
- **Tamper evidence** — altering, deleting or reordering a committed block is
  detected, and re-signing a rewritten block still fails at the next block.

Two tests are marked `expectedFailure`. They document a known defect rather
than assert around it; see **Known issues** below.

---

## Build order

The schema first — everything else depends on the contract it defines.

### 1. `reference/unified_schema.py` — the contract

One request shape that all four providers can serve, and one response shape they all normalize
into. Pydantic v2 models with the validation that stops bad requests reaching a provider.

```python
from unified_schema import UnifiedRequest, Message, Role, RoutingConstraints

req = UnifiedRequest(
    model_role="reasoning",                     # a role, never a provider model id
    messages=[Message(role=Role.USER, content="…")],
    routing=RoutingConstraints(residency="eu", zero_data_retention=True),
)
```

Bind to a **role**, not a model name. That is the whole point — the router resolves the role to
a concrete endpoint at request time, so swapping providers is a routing decision, not a code
change.

Validators worth knowing: a forced tool choice must name a declared tool, messages cannot be
empty, content cannot be blank, and `TokenUsage` exposes `billable_output` and `total` so cost
attribution has one definition. `UnifiedResponse.is_truncated` distinguishes a max-token stop
from a natural one — collapse those two and clients process a partial answer as complete.

### 2. `reference/streaming_core.py` — the request path

A FastAPI service exposing `POST /v1/inference:stream`. It transforms provider SSE frames into
one shape and redacts PII as the stream passes through.

```bash
uvicorn streaming_core:app --port 8080
```

Three functions carry the work:

- `extract_delta(obj, provider)` — pulls the incremental text out of whichever field that
  provider uses
- `redact(text)` — deterministic PII redaction
- `transform_and_sanitize(upstream, provider)` — the async generator that does both per frame

The redactor holds a bounded suffix as carry, so a value split across two frames — `…SSN is
123-45` then `-6789` — is still caught. Redacting per frame without that carry leaks every
value that straddles a chunk boundary.

Do not buffer the whole body to make redaction easier. That trades away time-to-first-token,
which is the metric users actually judge.

### 3. `reference/otel_instrumentation.py` — make it observable

Standard `gen_ai.*` semantic-convention attributes plus cost, so one dashboard compares all
four providers.

```python
from otel_instrumentation import InferenceObservation, record_inference

record_inference(InferenceObservation(...))   # exposes tokens_per_second
```

Wire this early. Cost and quality you cannot see are cost and quality you cannot govern.

### 4. `reference/privatelink.tf` — close the network

Private endpoints to each provider, no public egress on the inference path.

```bash
terraform init && terraform plan
```

Variables: `vpc_id`, `private_subnet_ids`, `endpoints_only_sg`. Stand this up before a single
production token flows — it is a precondition, not a hardening step you add later.

---

## Then, by concern

`chapters/` holds fifteen further implementations. Take what the problem calls for.

| Concern | File |
|---|---|
| API contract, published | `ch01/listing-1.1-gateway-contract.openapi.yaml` |
| Routing on live latency, cost and health | `ch01/listing-1.2-dynamic-router.py` |
| Normalizing four providers' responses | `ch02/listing-2.1-response-normalizer.py` |
| Circuit breaking and failover | `ch03/listing-3.1-sse-proxy-circuit-breaker.py` |
| Redaction, semantic cache, tool gating | `ch04/listing-4.1-redactor-cache-tool-gate.py` |
| Per-request cost accounting | `ch05/listing-5.1-financial-accounting-vbe.py` |
| Telemetry event shape | `ch06/listing-6.1-unified-telemetry-event.py` |
| Append-only cost ledger | `ch06/listing-6.2-clickhouse-ledger.sql` |
| Hard spend caps under concurrency | `ch06/listing-6.3-redis-budget-enforcement.py` |
| GenAI tracing | `ch07/listing-7.1-otel-genai-instrumentation.py` |
| Automated model promotion and rollback | `ch07/listing-7.2-progression-orchestrator.py` |
| Keyless workload identity | `ch08/listing-8.1-oidc-web-identity-role.tf` |
| Tamper-evident audit logging | `ch08/listing-8.2-tamper-evident-audit-log.py`, `ch08/listing-8.3-hash-chained-audit-ledger.py` |
| Signal-to-owner routing | `ch11/listing-11.1-metric-registry.yaml` |

---

## Beyond the printed form

`extensions/` holds the versions that outgrew what a printed listing should be.

### `extensions/audit_ledger.py` — the ledger with a pluggable signer

```bash
python3 extensions/audit_ledger.py
```

Listing 8.3 signs with HMAC and hard-codes it, which is right for teaching the
hash chain and wrong for deployment, because the choice between tamper evidence
and non-repudiation should not be a rewrite. Here the signer is injected:

```python
from audit_ledger import AuditLedger, HMACSigner, Ed25519Signer, verify_chain

ledger = AuditLedger(HMACSigner(hsm_derived_key))       # tamper evidence
ledger = AuditLedger(Ed25519Signer(kms_private_key))    # non-repudiation

ok, first_bad = verify_chain(entries, verifier)
```

Three differences from the printed listing, each deliberate:

- **The signer is an interface.** Entries record which algorithm signed them,
  so a ledger that changes signers mid-life stays verifiable end to end.
- **Serialization is canonical and defined here**, not inherited from Pydantic's
  field ordering. A record you must verify in seven years should not have its
  hash preimage defined by a dependency you will upgrade next quarter.
- **Verification walks the chain from genesis**, which is what detects a deleted
  or reordered block. Checking one entry against a predecessor hash the caller
  supplies assumes the caller already knows the right answer.

### `extensions/privatelink_oci.tf` — the fourth provider

`reference/privatelink.tf` shows private-only ingress for AWS, Azure and GCP.
This completes it with the OCI Generative AI private endpoint, so all four
providers are reachable over private IP with no public path to a model API.

```bash
terraform init && terraform plan
```

Variables: `oci_compartment_id`, `oci_private_subnet_id`, `oci_endpoints_only_nsg`,
`oci_dns_prefix`. Written against the published
`oci_generative_ai_generative_ai_private_endpoint` schema.

OCI has two private paths and they are not interchangeable. A **private endpoint**
gives a private IP in a subnet you choose with its own FQDN — the analogue of
PrivateLink and Private Endpoint, and what you want when the gateway addresses a
specific endpoint. A **service gateway** grants a subnet private access to the
Oracle Services Network as a whole, which is coarser. Two things to check that
Terraform will not settle for you: on-demand usage must be allowed on the
endpoint if you route to on-demand models, and the default tenancy limit is five
private endpoints.

---

## What you are getting

Reference implementations, not a packaged product. There is no CI and no deployment manifest —
those depend on your environment. The modules are meant to be lifted into your own tree rather
than installed from here, which is why they carry no packaging scaffolding you would have to
strip out later.

**On signing and what it proves.** Listing 8.3 signs with HMAC, which gives tamper evidence and
authenticated integrity. It is **not** non-repudiation: the key is symmetric, so any party who
can verify a tag could equally have produced it — the verifier is a suspect. Moving to
asymmetric signing is not a configuration change in the printed form; the key type, the signing
call and the entry's `hmac_signature` field would all have to change, and that last one is a
schema change affecting any ledger already written. `extensions/audit_ledger.py` is that
refactor already done. Use HMAC when the question is "has this been altered since we wrote it".
Use Ed25519, with the private key held in an HSM or KMS, when it must be "prove to a third party
that you wrote it".

**Provider APIs move**, sometimes more than once a year, sometimes without the underlying
service changing. Names here follow provider documentation and API surfaces at the time of
writing. Where a name has moved, the architecture is unaffected — the seam is the durable part.

## Known issues

**Long API keys can evade redaction when split across stream frames.** The sliding window in
`reference/streaming_core.py` holds back `_HOLDBACK = 32` characters, sized from
`_LONGEST_PATTERN = 24`. The API-key pattern `sk-[A-Za-z0-9]{20,}` has no upper bound, so a key
longer than the holdback is redacted when it arrives in one frame and passes through in the
clear when it straddles a boundary — the exact failure the window exists to prevent, returning
for the one pattern whose length is not capped. SSNs (11 characters) and card numbers are
unaffected.

The fix is to bound the pattern and size the holdback from that bound; raising the constant on
its own moves the hole rather than closing it. Two tests in
`tests/test_streaming_core.py` are marked `expectedFailure` against this. When they start
reporting *unexpected success*, the fix has landed and the decorators should come off.

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Chosen over MIT for the
explicit patent grant in section 3.

## Contact

`thegroundedenterprise.com/activation/`
