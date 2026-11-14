-- Enterprise AI Architecture & Cloud Strategy — companion code
-- Listing 6.2 — ClickHouse ledger (MergeTree, append-only) with an incremental materialized-view rollup for billing.
--
-- Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
-- SPDX-License-Identifier: Apache-2.0
-- github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

-- Append-only ledger: insert only, never update. Immutable audit trail.
CREATE TABLE metering.events
(
    event_id           UUID,
    ts                 DateTime64(3, 'UTC'),
    tenant             LowCardinality(String),
    department         LowCardinality(String),
    feature            LowCardinality(String),
    user               String,
    provider           LowCardinality(String),
    model              LowCardinality(String),
    region             LowCardinality(String),
    prompt_tokens      UInt32,
    completion_tokens  UInt32,
    cached_tokens      UInt32,
    reasoning_tokens   UInt32,
    retrieval_ops      UInt32,
    compute_ms         UInt32,
    cost_usd           Decimal(18, 8),
    price_book_version LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)
ORDER BY (tenant, feature, ts)
TTL toDateTime(ts) + INTERVAL 90 DAY TO VOLUME 'cold';

-- Incremental rollup maintained at insert time: hourly cost per tenant/feature.
CREATE MATERIALIZED VIEW metering.rollup_hourly
ENGINE = SummingMergeTree
PARTITION BY toYYYYMMDD(hour)
ORDER BY (tenant, feature, provider, hour)
AS SELECT
    tenant, feature, provider,
    toStartOfHour(ts)                 AS hour,
    sum(cost_usd)                     AS cost_usd,
    sum(prompt_tokens + completion_tokens) AS total_tokens,
    sum(cached_tokens)                AS cached_tokens,
    count()                           AS requests
FROM metering.events
GROUP BY tenant, feature, provider, hour;

-- Billing API reads the small pre-aggregated rollup, not raw events:
-- SELECT tenant, sum(cost_usd) FROM metering.rollup_hourly
--   WHERE hour >= :from AND hour < :to GROUP BY tenant;
