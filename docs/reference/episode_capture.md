# Episode capture (`capture_kind=episode`)

An EPISODE is a multi-step generation run — a loop that calls one model N
times, feeding each step from the last — captured as ONE wrapped session
product. Declare it on the ordinary capture entry:

```python
import torch
import torchlens as tl
from torchlens.options import EpisodeSpec

class GreedyRunner(torch.nn.Module):
    def __init__(self, model, n_steps):
        super().__init__()
        self.model = model          # the stepped model
        self.n_steps = n_steps

    def forward(self, ids):
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)   # one integer tensor of emitted tokens

log = tl.trace(
    GreedyRunner(model, 20), prompt_ids,
    episode=EpisodeSpec(stepped_module=model, n_steps=20),
)
ledger_payload = log.annotations["episode"]     # header + per-step rows
```

Every spelling on this surface is DOCUMENTED-UNSTABLE (no deprecation shim
owed) pending the rolling naming session; the semantics below are the
ratified contract.

## Cost: a diagnostic-tier product, for tens of steps

The wrapped episode tier is the verification oracle / deep-dive product. Its
cost is SUPERLINEAR in step count — measured on gpt2-124M (CPU, 8-token
prompt): N=20 costs 79 s (103x native) with a 146 MB artifact and 1.9 GB peak
RSS; N=100 costs 657 s (323x native), 947 MB, 5.4 GB peak RSS. Do not plan
wrapped episode captures for hundreds of steps; the guarded-fast tier
(`trace.run(inputs=..., fast=True)`) is the default engine for episode-scale
re-runs, and it must reproduce the wrapped tier's tokens bit-exactly (a
pinned cross-tier identity test guards this).

## The declaration

`EpisodeSpec(stepped_module=...)` declares the STEPPED MODULE — the
`nn.Module` whose successive top-level calls define step boundaries. It must
be a PROPER submodule of the traced episode root (wrap the loop in a module;
a callable root is a separate, later contract). Step tallying is FLAT: each
top-level call of the stepped module is one step, regardless of loop nesting
inside the root's `forward`. Call 1 is the prefill (ledger row 0,
`role="prefill"`); later calls are decode steps.

Declared episode-carried state beyond the built-in scope (token prefix, KV
cache, RNG streams) is preflighted at DECLARATION time, unconditionally: any
item without snapshot/restore support refuses typed
(`episode_state_unsnapshotable`) before the forward runs.

Refused combinations (typed, `episode_declaration_invalid`): chunked
forwards, `cache=True`, value-free save policies (the token column derives
from the retained root output), a structure-only marker (structure-only
episodes are out of scope per the ratified marker-combination table), and
non-torch backends.

## The ledger

The per-step status ledger lands ON the product at
`trace.annotations["episode"]` after settlement: a header (`episode_id`,
stepped-module address, the managed-RNG `entry_seed`, token feed,
provenance tier, escalation disclosures) plus one row per step
(`episode_step`, `role`, `status`, coordinates, `cache_len`, emitted
`tokens`). Row status is a closed vocabulary:

- `complete` — step k's forward returned (row-scoped truth, never a claim
  about the product; product truth is the settled `CaptureOutcome`).
- `interrupted` — the settled outcome's frontier lies INSIDE step k (a halt
  or failure mid-step; the frontier disclosure rides the row).
- `absent` — the step never started.

Rows obey the MONOTONE PREFIX LAW (complete prefix, at most one interrupted
row, absent tail) and are write-once at settlement. The ledger is a
DISCLOSURE, never a settlement authority: the episode product settles through
the one existing `CaptureOutcome` authority with its vocabulary unchanged,
and every capability gate (N1–N5) applies verbatim. Truncation stays a
run/report term: an episode halted at a step boundary settles HALTED with a
clean prefix and the truncation disclosed on the ledger
(`steps_completed`, `truncated_at_step`) — nothing rounds up.

Loads validate fail-closed: an episode ledger on a capture without the
declaration refuses typed (`episode_ledger_without_declaration`); a
structure-only ledger carrying tokens refuses typed
(`episode_ledger_payload_in_structure_only`); any other geometry violation
QUARANTINES the payload with one warning (`episode_ledger_incoherent`) — the
rows stop being claims and the outcome derivation treats the ledger
fail-closed.

## Persistence

`annotations["episode"]` persists plainly as of the tlspec v8 coordinated
bump, as does the Bundle `member_relations` key (below). Loads validate
fail-closed: an undeclared ledger refuses `episode_ledger_without_declaration`
and geometry violations quarantine `episode_ledger_incoherent`. Pre-v8
artifacts never carry the key (the v7 scrub dropped it; the live trace kept
its session-time ledger).

## Teacher forcing (disclosed, non-verifying)

`EpisodeSpec(forced_tokens=...)` declares a teacher-forced feed: the driver
feeds the declared tokens instead of the model's emissions. The ledger header
records `token_feed="forced"` and `fidelity_basis="forced"` — an explicitly
NON-VERIFYING disclosed mode; token-fidelity obligations never verify a
forced episode. Recompute-and-compare remains the verifying default
everywhere else.

## The managed RNG recipe

Episode captures use the managed seeding discipline: the capture's effective
`random_seed` (drawn or passed, exactly as for any capture) is recorded as
the ledger `entry_seed`. Re-running the same declaration with the same seed
reproduces a sampled episode bit-identically; pass
`capture=CaptureOptions(random_seed=...)` explicitly for cross-run
reproduction.

## Escalation (session semantics)

A failed or suspect cheap-tier episode escalates by re-running the WHOLE
episode wrapped, disclosed — never a partial product. Build the declaration
with `torchlens.capture._episode_ledger.escalation_spec(producer,
stepped_module=..., reason=...)`: it derives `escalated_from` (a digest over
the producer's persisted outcome payload + ledger rows), the closed-vocabulary
`reason` (`step_failed` / `divergence` / `requested`), and the producer's
token column, so the escalated capture discharges the fidelity obligation at
write time: prefix-equal token columns record `fidelity_basis="tokens"`; a
mismatch records `"diverged"` — the escalated product is still a valid
capture of what it ran, it just is not an escalation of the original episode,
and says so. Fidelity is a disclosure, never a settlement input.

## The floor: per-step Bundles + the derived fold

The declared minimum episode product (invocable by the sprint orchestrator on
size evidence only) is N per-step captures composed in a `Bundle` with S6
`episode_member` relation rows, under OBSERVER DISCIPLINE: the driver's
native forward owns the live carried state; each capture observes an isolated
copy; a capture product is never the source of carried episode state.
`Bundle.derive_episode_status(episode_id, ledger=...)` folds member outcomes
into a DERIVED episode status (`episode_complete` /
`episode_halted_at_step` / `episode_aborted_at_step` /
`episode_failed_at_step` / `episode_unknown`) — a total, fail-closed
derivation, never a Bundle-level settlement. Fold arms that consume
ledger-only declaration facts (the declared step count; a driver-declared
boundary halt) degrade to `episode_unknown` without a re-supplied ledger —
disclosed truth loss until the schema bump persists the ledger. The episode's
provenance tier is the MINIMUM of its members' tiers, never the maximum.

Bundle relation rows never require alignment, never assert structural
comparability, and never dangle: mutators either cascade explicitly
(`cascade_relations=True`) or refuse typed. See
`docs/reference/error_refusal_contract.md` for the full episode/bundle
refusal code family.
