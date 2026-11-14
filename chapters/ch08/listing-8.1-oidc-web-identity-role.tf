# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 8.1 — Terraform / OpenTofu: OIDC web-identity role (no static keys), HSM-backed signing key, private endpoint.
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

# --- Keyless workload identity: OIDC federation, no long-lived keys ---
resource "aws_iam_openid_connect_provider" "workloads" {
  url             = "https://oidc.gateway.internal"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [var.oidc_thumbprint]
}

data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.workloads.arn]
    }
    # constrain to the intended workload only (sub + aud pinned)
    condition {
      test     = "StringEquals"
      variable = "oidc.gateway.internal:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "oidc.gateway.internal:sub"
      values   = ["system:serviceaccount:ai-gateway:proxy"]
    }
  }
}

resource "aws_iam_role" "gateway" {
  name                 = "ai-gateway-proxy"
  assume_role_policy   = data.aws_iam_policy_document.trust.json
  max_session_duration = 3600            # short-lived credentials only
}

# --- HSM-backed signing key; rotate by replacement ---
resource "aws_kms_key" "audit_hmac" {
  description             = "HMAC signing key for tamper-evident audit log"
  key_usage               = "GENERATE_VERIFY_MAC"
  customer_master_key_spec = "HMAC_256"
  # HMAC keys have no auto-rotation; keep prior keys to verify old entries
  deletion_window_in_days = 30
}

# --- Private connectivity: no public egress to the model API ---
resource "aws_vpc_endpoint" "bedrock" {
  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.region}.bedrock-runtime"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true             # resolves to private IPs only
  subnet_ids          = var.private_subnet_ids
  security_group_ids  = [var.endpoints_only_sg]
}
