# Inventory — companion code

`Enterprise AI Architecture & Cloud Strategy` · 32 files · 4,414 lines · Apache-2.0
Repository: `github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy`

---

## Repository description (for the GitHub "About" field)

> Provider-neutral AI inference gateway: one request schema served by AWS, Azure, GCP and OCI,
> streaming PII redaction, capacity crossover economics, private connectivity and a
> tamper-evident audit ledger. Companion code to the book.

Suggested topics: `multi-cloud` · `llm-gateway` · `finops` · `terraform` · `opentelemetry` ·
`pydantic` · `fastapi` · `enterprise-architecture`

---

## Structure

Two halves with different rules. This distinction is the one thing a contributor or reader
needs to understand before touching anything.

| Half | Directories | Rule |
|---|---|---|
| **Generated** | `chapters/`, `reference/` | Produced by `build/extract_listings.py` from the manuscript. Never hand-edit — changes are overwritten on the next extract. Below the licence header, the code is byte-identical to the printed listing, so a reader can diff it against the page and get nothing back. |
| **Authored** | `extensions/`, `tests/`, manifests | Written for the repository. Where a printed listing is deliberately taught in its simplest correct form, the deployable version lives here. |

---

## `reference/` — the five modules Table 10.1 names (823 lines)

Printed as Listings 10.1–10.4b, filed under the module names the book uses.

| File | Lines | What it is | Needs |
|---|---:|---|---|
| `unified_schema.py` | 252 | The contract. Pydantic v2 request/response models; callers bind to a **role**, never a provider model id. Validators cover forced-tool declaration, non-empty messages, and a `TRUNCATED` finish reason that never collapses into `STOP`. | `pydantic` |
| `streaming_core.py` | 197 | FastAPI service exposing `POST /v1/inference:stream`. Normalizes provider SSE frames and redacts PII with a bounded sliding window, so a value split across two frames is still caught without buffering the body. | `pydantic`, `fastapi` |
| `crossover_calculator.py` | 195 | The Capacity Crossover Equation (V_be), break-even utilization (U_be) and Traffic Volatility Index (V_t), with a cost-optimal decision across serverless, provision-to-peak and hybrid. | nothing — stdlib |
| `otel_instrumentation.py` | 106 | `gen_ai.*` semantic-convention spans plus cost and tokens-per-second, so one dashboard compares all four providers. | `opentelemetry-api`, `-sdk` |
| `privatelink.tf` | 73 | Private-only ingress: AWS PrivateLink, Azure Private Endpoint, GCP Private Service Connect, and a Direct Connect gateway. OCI is in `extensions/`. | Terraform / OpenTofu |

## `chapters/` — the other fifteen listings (1,444 lines)

| File | Lines | Concern |
|---|---:|---|
| `ch01/listing-1.1-gateway-contract.openapi.yaml` | 130 | The published API contract |
| `ch01/listing-1.2-dynamic-router.py` | 130 | Routing on live latency, cost and health |
| `ch02/listing-2.1-response-normalizer.py` | 171 | Collapsing four providers' response shapes into one |
| `ch03/listing-3.1-sse-proxy-circuit-breaker.py` | 175 | Circuit breaking and failover on the streaming path |
| `ch04/listing-4.1-redactor-cache-tool-gate.py` | 144 | Redaction, semantic cache, tool-call gating |
| `ch05/listing-5.1-financial-accounting-vbe.py` | 94 | Per-request cost accounting |
| `ch06/listing-6.1-unified-telemetry-event.py` | 65 | The telemetry event shape |
| `ch06/listing-6.2-clickhouse-ledger.sql` | 51 | Append-only cost ledger schema |
| `ch06/listing-6.3-redis-budget-enforcement.py` | 63 | Hard spend caps under concurrency |
| `ch07/listing-7.1-otel-genai-instrumentation.py` | 53 | GenAI tracing |
| `ch07/listing-7.2-progression-orchestrator.py` | 86 | Automated model promotion and rollback |
| `ch08/listing-8.1-oidc-web-identity-role.tf` | 60 | Keyless workload identity; KMS HMAC signing key |
| `ch08/listing-8.2-tamper-evident-audit-log.py` | 97 | Tamper-evident audit logging |
| `ch08/listing-8.3-hash-chained-audit-ledger.py` | 92 | Hash-chained ledger with HMAC |
| `ch11/listing-11.1-metric-registry.yaml` | 33 | Signal-to-owner routing |

## `extensions/` — beyond the printed form (517 lines)

| File | Lines | Why it exists |
|---|---:|---|
| `audit_ledger.py` | 434 | Listing 8.3 with a **pluggable signer**. `HMACSigner` reproduces the printed behaviour; `Ed25519Signer` gives non-repudiation. Canonical serialization is pinned in the module rather than inherited from Pydantic's field ordering, because a record you must verify in seven years should not have its hash preimage defined by a dependency. Verification walks the chain from genesis, which is what detects a deleted or reordered block. |
| `privatelink_oci.tf` | 83 | The fourth provider. `oci_generative_ai_generative_ai_private_endpoint`, completing the private-ingress pattern the printed Terraform shows for three. Documents why a private endpoint is not a service gateway. |

## `tests/` — 114 tests (1,067 lines)

```bash
python3 -m unittest discover -s tests -t tests     # no installation required
```

83 execute with nothing installed; 29 skip pending `pydantic`; 2 are `expectedFailure` against a
known defect. Written on `unittest` so they run under `pytest tests` as well.

| File | Tests | Pins |
|---|---|---|
| `test_crossover_calculator.py` | 22 | Every intermediate of the Chapter 5 worked example — $3,640 committed, U_be 36.5%, V_t 0.40, $5,219.81 hybrid, 47.3% saving — plus the identity `V_be × price == C_fixed`, which is the check that catches a break-even equation carrying its unknown on both sides. |
| `test_audit_ledger.py` | 36 | Tamper evidence: altering, deleting or reordering a block is detected; re-signing a rewritten block still fails downstream. Canonical serialization stability. Non-repudiation under Ed25519, including that a verifier holding only the public key cannot forge. |
| `test_streaming_core.py` | 27 | Redaction across frame boundaries, both two-frame and character-by-character. Per-provider delta extraction — the executable form of the Chapter 2 response-shape tables. Output begins before the stream ends. |
| `test_unified_schema.py` | 29 | Forced tools must be declared, `TRUNCATED` never collapses to `STOP`, reasoning tokens are billable output, and no request field names a vendor. **Requires `pydantic`; not yet executed.** |

## Manifests and licence

| File | Purpose |
|---|---|
| `requirements.txt` | Runtime, annotated by concern. Nothing is needed for the capacity maths or the ledger. |
| `requirements-dev.txt` | Adds `pytest` and lifts the schema-test skips. |
| `LICENSE` | Apache-2.0, chosen over MIT for the explicit patent grant in section 3 — the clause enterprise legal looks for. |
| `NOTICE` | Attribution required by Apache-2.0 §4(d). |
| `.gitignore` | `__pycache__`, Terraform state and plan files. |

---

## Known gaps

1. **`test_unified_schema.py` has never run.** Its 29 tests skip without `pydantic`. One run with
   dependencies installed before the repo goes public.
2. **`extensions/privatelink_oci.tf` has not been through `terraform validate`.** Written from
   Oracle's published schema and checked for delimiter balance and argument names, which is not
   the same thing.
3. **No configuration examples.** Page ix promises "dependency manifests, test scaffolding,
   configuration examples". The first two exist; the third does not.
4. **No CI.** The suite is runnable but nothing runs it on a schedule. Related: the weekly
   provider smoke suite described in `DISCUSSION-code-repo.md` is the drift detector page ix
   promises, and it needs credentials and a cadence.
5. **One tracked defect** — long API keys can evade redaction when split across stream frames.
   See `ERRATA.md`; two tests are marked `expectedFailure` against it.
