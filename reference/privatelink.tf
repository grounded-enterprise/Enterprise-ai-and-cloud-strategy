# Enterprise AI Architecture & Cloud Strategy — companion code
# Listing 10.4a — privatelink.tf : private cross-cloud connectivity (Terraform / OpenTofu).
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy

# Private cross-cloud connectivity (Terraform / OpenTofu).
# Private-only ingress for AWS, Azure and GCP (OCI: same pattern) so the
# gateway never reaches a model API over the public internet (Chapter 8).

variable "region"             { type = string }
variable "vpc_id"             { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "endpoints_only_sg"  { type = string }   # egress = provider endpoints only

# --- AWS: PrivateLink interface endpoint to Bedrock runtime ---------------- #
resource "aws_vpc_endpoint" "bedrock_runtime" {
  vpc_id              = var.vpc_id
  service_name        = "com.amazonaws.${var.region}.bedrock-runtime"
  vpc_endpoint_type   = "Interface"
  private_dns_enabled = true                        # resolves to private IPs only
  subnet_ids          = var.private_subnet_ids
  security_group_ids  = [var.endpoints_only_sg]

  tags = {
    Name      = "bedrock-privatelink"
    Compliance = "no-public-egress"
  }
}

# --- Azure: Private Endpoint to the Azure OpenAI account ------------------- #
resource "azurerm_private_endpoint" "openai" {
  name                = "openai-private-endpoint"
  location            = var.region
  resource_group_name = var.azure_resource_group
  subnet_id           = var.azure_private_subnet_id

  private_service_connection {
    name                           = "openai-psc"
    private_connection_resource_id = var.azure_openai_account_id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }
}

# --- GCP: Private Service Connect endpoint to Vertex AI online prediction --- #
resource "google_compute_forwarding_rule" "vertex_psc" {
  name                  = "vertex-psc-endpoint"
  region                = var.region
  network               = var.gcp_network
  ip_address            = var.gcp_psc_ip
  load_balancing_scheme = ""                        # required empty for PSC
  target                = var.vertex_service_attachment
}

# --- Dedicated interconnect (no traffic over the public internet) ---------- #
resource "aws_dx_gateway" "backbone" {
  name            = "ai-gateway-dx"
  amazon_side_asn = 64512
}

variable "azure_resource_group"      { type = string }
variable "azure_private_subnet_id"   { type = string }
variable "azure_openai_account_id"   { type = string }
variable "gcp_network"               { type = string }
variable "gcp_psc_ip"                { type = string }
variable "vertex_service_attachment" { type = string }

output "bedrock_endpoint_dns" {
  value       = aws_vpc_endpoint.bedrock_runtime.dns_entry
  description = "Private DNS entries for the Bedrock interface endpoint."
}
