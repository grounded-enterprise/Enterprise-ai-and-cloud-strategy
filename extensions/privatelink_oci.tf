# Enterprise AI Architecture & Cloud Strategy — companion code
#
# Copyright 2026 Alan Hancock and Duane Cash / Chili Pepper Studios, Inc.
# SPDX-License-Identifier: Apache-2.0
# github.com/grounded-enterprise/Enterprise-ai-and-cloud-strategy
#
# Written for this repository, beyond the printed listings.
# Hand-maintained — not generated from the manuscript.

# OCI private endpoint for Generative AI (Terraform / OpenTofu).
#
# The printed Listing 10.4a shows three of the four providers — AWS PrivateLink,
# Azure Private Endpoint, GCP Private Service Connect. This is the fourth, and
# it completes the pattern the book describes: a private IP inside your own
# network, an FQDN that resolves to it, and no public path to a model API.
#
# OCI offers two private paths. They are not interchangeable:
#
#   Private endpoint (this file)  A private IP in a subnet you choose, with its
#                                 own FQDN. Closest analogue to PrivateLink and
#                                 Private Endpoint, and the one to use when the
#                                 gateway must address a specific endpoint.
#   Service gateway               Subnet-level private access to the Oracle
#                                 Services Network as a whole. Coarser; grants
#                                 the subnet reach to Oracle services generally
#                                 rather than binding one endpoint.
#
# Use with reference/privatelink.tf. Variable names follow that file.

variable "oci_compartment_id"     { type = string }
variable "oci_private_subnet_id"  { type = string }   # must be a PRIVATE subnet
variable "oci_endpoints_only_nsg" { type = list(string) }
variable "oci_dns_prefix" {
  type        = string
  description = "Label forming the endpoint FQDN. Stable; changing it moves the name."
}

# --- OCI: private endpoint to Generative AI -------------------------------- #
resource "oci_generative_ai_generative_ai_private_endpoint" "genai" {
  compartment_id = var.oci_compartment_id
  dns_prefix     = var.oci_dns_prefix
  subnet_id      = var.oci_private_subnet_id

  display_name = "genai-private-endpoint"
  description  = "Private-only ingress to OCI Generative AI for the inference gateway."

  # Controls which hosts in the VCN may reach the endpoint's VNIC. This is the
  # OCI counterpart of the endpoints-only security group in the AWS resource:
  # the endpoint existing is not the control, restricting who can call it is.
  nsg_ids = var.oci_endpoints_only_nsg

  freeform_tags = {
    Compliance = "no-public-egress"
  }

  # `resource_type` is also accepted, and scopes what the endpoint may serve.
  # Deliberately unset: Oracle's provider documentation names the argument but
  # does not publish its permitted values, and guessing an enum into
  # infrastructure code is how you get an apply that fails in someone else's
  # account. Check the current provider docs and set it explicitly if you need
  # to constrain the endpoint.
}

output "oci_genai_endpoint_fqdn" {
  value       = oci_generative_ai_generative_ai_private_endpoint.genai.fqdn
  description = "FQDN the gateway calls. Confirm it resolves to the private IP."
}

output "oci_genai_endpoint_private_ip" {
  value       = oci_generative_ai_generative_ai_private_endpoint.genai.private_endpoint_ip
  description = "Private IP in the VCN backing the endpoint."
}

# Two operational notes the console exposes but Terraform does not settle for you:
#
#   On-demand mode. A private endpoint can serve on-demand models as well as
#   models hosted on a dedicated AI cluster, but on-demand usage must be
#   allowed on the endpoint. Verify this on the created endpoint if the gateway
#   routes to on-demand models.
#
#   Tenancy limit. The default is five private endpoints per tenancy. A
#   multi-region or multi-environment rollout reaches that faster than expected;
#   raise the `private-endpoint-count` limit before it blocks a deployment.
