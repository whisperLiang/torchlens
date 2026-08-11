# Sparse runnable model

This document is the short normative model for TorchLens sparse runnable `.tlspec`
artifacts. It explains the ownership boundary and the invariants a maintainer needs in
order to change the subsystem safely. The exhaustive, frozen contract remains
[`runnable_tlspec_contract.md`](runnable_tlspec_contract.md). If this model and the
exhaustive contract disagree, the exhaustive contract and the behavior-free types in
`torchlens.runnable` are authoritative; the disagreement is a release-blocking defect.

## Scope

A sparse runnable artifact records one observed execution path without embedding a model
object or tensor-valued intermediates in its sparse core. It can be loaded for analysis and,
when readiness succeeds, replayed through the transactional `Trace.run()` API.

The subsystem has four ownership regions:

| Owner | Responsibility |
|---|---|
| `torchlens.runnable` | Frozen public schemas, enums, reports, and error vocabulary |
| Runnable internals | Descriptor semantics, validation, resolution, state preparation, execution, and attestation |
| `torchlens._io` | Manifest and blob transport, syntax-level encoding/decoding, integrity, and filesystem transactions |
| `Trace` | User-facing properties and provider selection through one private runnable-state seam |

The transport layer must not decide witness meaning, readiness, path faithfulness, or
attestation. Runnable code must not own filesystem transactions or manifest/blob integrity.
The narrow coordinator vocabulary is:

- `produce`: derive the semantic sparse descriptor and payload declarations from a live Trace;
- `decode`: parse transport data into typed runnable records without executing it;
- `prepare`: validate semantic consistency, resolve capabilities, bind payloads, and report
  non-executing readiness;
- `execute`: run the selected provider transactionally and settle one result.

`bundle.py` orchestrates transport through this boundary. It must not import or invoke private
witness-builder helpers directly. `torchlens.runnable` remains behavior-free and must not import
I/O, capture, state preparation, or execution modules.

No optional package extra is warranted. Sparse runnable execution requires no dependency beyond
TorchLens's existing torch requirement, so an extra would not describe a real capability boundary.

## Artifact families

The runnable artifact consists of one required sparse program and up to three external tensor
payload families. The sparse core itself is always tensor-value-free.

| Family | Presence | Purpose |
|---|---|---|
| `sparse_recorded_taken_path_v2` | Required | Taken-path call recipe, slots, contexts, state bindings, output contract, and honesty obligations |
| `state_dict_v1` | Optional | Full capture-time `state_dict`: named parameters plus persistent buffers |
| `runnable_nonpersistent_buffer_v1` | Required when used | Capture-time values of used non-persistent buffers in the declared state model |
| `selected_activation_v2` | Optional | Capture-selected `out`/`transformed_out` payloads for inspection and eligible byte attestation |

Weights and archived activations are independent opt-ins. Non-persistent-buffer payloads are not
an opt-in: when the taken path consumes one, its value is required state and the save discloses
that fact. Archived activations never seed execution. The complete payload rules are in
[contract section 5](runnable_tlspec_contract.md#5-producer-preflight-and-no-payload-invariant)
and the public option spelling is in
[section 12](runnable_tlspec_contract.md#12-optional-payload-api-spelling-and-docs-lockstep).

Every v2 call carries an explicit `CallExecutionContext`, and the descriptor carries an explicit
`AmbientExecutionContext`. Missing context records do not mean defaults; they identify a legacy
artifact that is analysis-only. The frozen call recipe is
`non_tensor_args_tensor_slots_context_and_obligations_v3`. These are internal schema version
strings, not a TorchLens major-version statement.

## Oracle and declared state

`verified` means faithful reproduction against oracle 1: a separate, fresh live-model execution
of the descriptor's program from the declared state, under the recorded execution context, on the
given inputs. It does not mean reproduction of a later call on the same mutated Python object.

The declared state is:

1. the capture-time `state_dict` universe of named parameters and persistent buffers;
2. the capture-time values of used non-persistent buffers; and
3. the recorded taken-path DAG and its explicit contexts and obligations.

Run preflight chooses state in this order: explicitly staged user state, embedded capture state,
then the frozen `torchlens_role_init_v2` random initializer. State staging is strict and atomic;
it never reconstructs a model object and never writes tensor payloads into the sparse descriptor.
The full boundary is specified in
[contract section 11](runnable_tlspec_contract.md#11-honesty-divergence-poison-and-exactness),
especially its declared-state subsection.

## Verdict lattice

Readiness is non-executing and uses exactly:

- `ready`;
- `unavailable`.

A completed path settles to exactly one `PathFaithfulness` value:

- `verified`: all applicable structural checks pass and no evidence ceiling prevents the claim;
- `diverged`: replay disagrees with the captured path or a required runtime fact;
- `unverifiable`: execution may complete, but evidence is insufficient to claim faithfulness.

Numeric attestation is downstream of path settlement:

- `attested`: eligible archived raw slots match exactly;
- `numeric_attestation_failed`: an eligible comparison mismatched and the transaction fails;
- `not_applicable`: the run is ineligible, including any path verdict other than `verified`;
- `not_present`: no selected-activation payload exists.

Attestation never upgrades path faithfulness. `attested` implies `verified` and an unpoisoned
result. A `diverged` result is returned only under the explicit `return_diverged` policy and is
monotonically poisoned. Incomplete witness coverage can settle only as `unverifiable`, never as
`verified`. The result and readiness shapes are frozen in
[contract section 8](runnable_tlspec_contract.md#8-readiness-and-result-shapes), and settlement
semantics are frozen in [section 11](runnable_tlspec_contract.md#11-honesty-divergence-poison-and-exactness).

## Witness obligations

The runnable proof is obligation-driven. Every replay-structure fact that can affect a verdict
creates a typed obligation on the record that owns it. Each obligation must be discharged by
exactly one matching witness or represented by a typed `WitnessCoverageGap`.

`WITNESS_FAMILY_REGISTRY` version `witness_family_registry_v2` is the closed inventory of every
verdict-steering witness family. It covers direct control witnesses, shape/structure fact
families, and claim-only families. The required-witness inventory is a redundant mirror of that
registry, not a separate authority.

`witness_completeness` is derived from the coverage-gap ledger. A persisted summary cannot
override that derivation. Deleting records or witnesses cannot improve a verdict: deletion either
leaves a still-complete exact discharge or creates a gap/refusal. The detailed owner fields,
anchors, XOR rule, and validation order are in
[contract section 4](runnable_tlspec_contract.md#4-authoritative-descriptor).

## Transaction boundary

`Trace.run(inputs=..., seed=...)` selects a live or loaded-sparse provider and returns
`RunResult(output, trace, report)`. The source Trace is unchanged for ordinary runs. State,
context, RNG, input binding, sparse calls, output reconstruction, honesty checks, and optional
numeric attestation settle inside one transaction.

The default divergence policy raises and rolls back. `return_diverged` is the sole opt-in that
returns a poisoned diagnostic Trace. Both providers use one settlement authority so they cannot
silently evolve different poison, exception, or report rules. Internal sparse calls run with
TorchLens logging paused.

`fast=True` is an explicit stateful static-loop mode, not a relaxation of the proof contract.
Loaded traces must first complete an ordinary verified run. Required input, path, output, and
control-witness guards remain active, and divergence always raises.

## Threat-model boundary

The proof covers faithful replay of the program described by the artifact. Coherent reauthoring is
out of scope: an author can construct a different, internally consistent descriptor for a weaker
program, and that descriptor may verify against that weaker program's own oracle 1. This is a
scope statement, not an undetected gap in witness coverage.

Within an artifact, record stripping must fail closed as a typed refusal, divergence, or
unverifiable verdict. The exact reauthoring boundary and residuals are frozen in
[the threat-model subsection of contract section 11](runnable_tlspec_contract.md#threat-model-scope-coherent-reauthoring-is-out-of-contract-scope-r71).

## Change discipline

The runnable vocabulary, witness meaning, verdict lattice, payload eligibility, and attestation
semantics are frozen. Changes require an explicit versioned contract amendment and synchronized
updates to typed schemas, serialization, validation, tests, and both runnable documents.

Refactors that do not amend the contract must preserve serialized bytes, error codes, readiness,
verdicts, poisoning, and attestation behavior. The exhaustive runnable, `.tlspec`, I/O, security,
and provider-parity suites are mandatory gates for changes at this boundary.
