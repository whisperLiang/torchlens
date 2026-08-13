"""Generate ``torchlens/ir/op_record_manifest.py`` from the source spec.

The manifest is the FROZEN ``CellSourceManifest`` v1: one row per op store
cell naming its step-0 ingest source class. Regenerate-and-diff is CI-gated
(``tests/producer_parity/test_p1_record_types.py``); editing the generated
file by hand is a build failure.

Run: ``python -m tools.generate_op_record_manifest``
"""

from __future__ import annotations

from pathlib import Path

HEADER = '''"""GENERATED FILE — do not edit. CellSourceManifest v{version}.

Regenerate with ``python -m tools.generate_op_record_manifest`` after editing
the source spec in ``torchlens/ir/op_record_scatter.py``. The joint-freeze
contract (producer DoR v4 section 5.2 / ppdag v3 section 9) versions this
table; post-freeze changes bump the version, never mutate v{version}.
"""

from __future__ import annotations

CELL_SOURCE_MANIFEST_VERSION = {version}

CELL_SOURCE_MANIFEST: dict[str, str] = {{
'''


def generate() -> str:
    from torchlens.ir.op_record_scatter import CELL_SOURCE_MANIFEST_VERSION, CELL_SOURCES

    lines = [HEADER.format(version=CELL_SOURCE_MANIFEST_VERSION)]
    for field_name in sorted(CELL_SOURCES):
        lines.append(f"    {field_name!r}: {CELL_SOURCES[field_name]!r},\n")
    lines.append("}\n")
    return "".join(lines)


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "torchlens" / "ir" / "op_record_manifest.py"
    target.write_text(generate())
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
