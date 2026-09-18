"""Native source positions preserve the disassembly oracle without its capture cost."""

import dis
from types import CodeType

import pytest

from torchlens.utils import _torch_compat, introspection

pytestmark = pytest.mark.skipif(
    not _torch_compat.HAS_CODE_POSITIONS, reason="Code positions require Python 3.11+"
)


@pytest.mark.parametrize(
    "source",
    [
        "def probe(x):\n    return x.sum() + x.mean()\n",
        "def probe(x):\n    return [item + 1 for item in x if item > 0]\n",
        "def probe(x):\n    return x.sin() if x.sum() > 0 else x.cos()\n",
        "def probe(数据):\n    return 数据.sum() + 数据.mean()\n",
        "def probe(x):\n    try:\n        yield x.sum()\n    finally:\n        x.clear()\n",
        "def probe(x):\n"
        + "".join(f"    value_{index} = x + {index}\n" for index in range(300))
        + "    return value_299\n",
    ],
    ids=["methods", "comprehension", "conditional", "unicode", "generator", "extended_args"],
)
def test_native_positions_match_every_disassembled_code_unit(source: str) -> None:
    """Compare all instructions and hidden cache slots, including nested code."""

    pending = [compile(source, "<position-table-parity>", "exec")]
    while pending:
        code = pending.pop()
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
        instructions = list(dis.get_instructions(code))
        expected = {}
        for index, instruction in enumerate(instructions):
            end = (
                instructions[index + 1].offset
                if index + 1 < len(instructions)
                else len(code.co_code)
            )
            column = None if instruction.positions is None else instruction.positions.col_offset
            expected.update(dict.fromkeys(range(instruction.offset, end, 2), column))
        actual = introspection._build_col_offset_map(code)
        assert actual == expected
        assert set(actual) == set(range(0, len(code.co_code), 2))


def test_position_lookup_does_not_disassemble(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold position-map build must not enter Python's full instruction decoder."""

    code = (lambda value: value.sum()).__code__
    expected = {index * 2: position[2] for index, position in enumerate(code.co_positions())}

    def reject_disassembly(*args: object, **kwargs: object) -> None:
        """Make any accidental full disassembly fail this regression."""

        raise AssertionError("source-column lookup must not disassemble instructions")

    monkeypatch.setattr(dis, "get_instructions", reject_disassembly)
    assert introspection._build_col_offset_map(code) == expected


def test_position_lookup_honors_missing_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-3.11 capability path still has no source-column evidence."""

    monkeypatch.setattr(_torch_compat, "HAS_CODE_POSITIONS", False)
    assert introspection._build_col_offset_map((lambda value: value).__code__) == {}


def test_absent_debug_table_never_invents_columns() -> None:
    """A code object without debug positions remains unknown at every offset."""

    code = (lambda value: value.sum()).__code__.replace(co_linetable=b"")
    positions = introspection._build_col_offset_map(code)
    assert all(positions.get(offset) is None for offset in range(0, len(code.co_code), 2))
