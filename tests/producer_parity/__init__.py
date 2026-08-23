"""Producer-unification parity harness (transient migration tooling).

This package grounds the producer-unification migration (P0-P7): the
instrumented consumer ledger, the two-check parity comparator (structural
Check A + attestation Check B), the planted-mutation proofs, the Tier-R
classification walker, and the capture probes. The comparator and the
attestation shims are deleted at the end of the migration; the ledger,
walker, and conformance tests are durable.
"""
