# Producer ledger maintenance

The four JSON files in `ledger/` inventory static read sites, observed runtime
reads, step-0 Trace reads, and amendment/mutation call sites. They are gate
artifacts, not model-output goldens. Ordinary test runs compare without writing.

Before refreshing, review static differences by file, field, and site count.
Check runtime differences separately and verify every new amendment caller uses
an existing registered family (or review a deliberate registry change first).
The closure test must still reject unexplained runtime reads, legacy
`replace_op_event` calls, and unsanctioned in-place event writes.

## Regeneration

The refresh runs the complete scenario battery and closure checks; record the
reviewed changes and regeneration provenance in `ledger/PROVENANCE`.

```bash
TORCHLENS_REFRESH_PRODUCER_LEDGER=1 \
TORCHLENS_GOLDEN_REASON='<reviewed changes and why>' \
python -m pytest tests/producer_parity/test_ledger.py::test_generate_and_close_ledger

# The generating run reports SKIP; only this separate run verifies the result.
python -m pytest tests/producer_parity/ -m "not rare and not slow" --tb=short
```

Do not set the refresh flag in CI or combine generation with an unrelated test
session. Keep the red-capability tests enabled: a mismatch, missing file, or
reasonless refresh must not silently overwrite its own evidence.
