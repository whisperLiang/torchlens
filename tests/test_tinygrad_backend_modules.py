"""tinygrad object-module backend tests."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.tinygrad.backend import discover_tinygrad_module_tree
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.validation import check_metadata_invariants
from torchlens.validation.invariants import MetadataInvariantError

tinygrad = pytest.importorskip("tinygrad")
Tensor = pytest.importorskip("tinygrad").Tensor
nn = pytest.importorskip("tinygrad.nn")


pytestmark = pytest.mark.backend_tinygrad


class _NamedSetLayer:
    """Tinygrad layer whose display repr is unrelated to its stable name."""

    def __init__(self, name: str, display: str) -> None:
        """Store a semantic name and a deliberately volatile display value."""

        self.name = name
        self.display = display
        self.weight = Tensor([1.0])

    def __call__(self, x: Any) -> Any:
        """Apply the trainable leaf."""

        return x * self.weight

    def __repr__(self) -> str:
        """Return an ordering hint that must not determine module addresses."""

        return self.display


def test_tinygrad_set_addresses_ignore_object_repr_order() -> None:
    """A named member keeps its address across equivalent unordered containers."""

    class Model:
        """Hold two named layers in an unordered set."""

        def __init__(self, reverse_display: bool) -> None:
            """Construct the same semantic members with opposite repr ordering."""

            a = _NamedSetLayer("a", "z" if reverse_display else "a")
            b = _NamedSetLayer("b", "a" if reverse_display else "z")
            self.group = {a, b}

        def __call__(self, x: Any) -> Any:
            """Return an input; discovery does not need to execute it."""

            return x

    for reverse_display in (False, True):
        tree = discover_tinygrad_module_tree(Model(reverse_display))
        assert tree is not None
        assert tuple(
            tree.metadata[f"group.{index}"]["_module_object"].name for index in (0, 1)
        ) == ("a", "b")


def test_tinygrad_set_refuses_indistinguishable_module_addresses() -> None:
    """Two same-type members with the same stable name cannot receive ordinals."""

    class Model:
        """Hold members that have no stable relative order."""

        def __init__(self) -> None:
            """Create semantically ambiguous set members."""

            self.group = {_NamedSetLayer("same", "a"), _NamedSetLayer("same", "z")}

        def __call__(self, x: Any) -> Any:
            """Return an input; discovery does not need to execute it."""

            return x

    with pytest.raises(BackendUnsupportedError, match="stable address"):
        discover_tinygrad_module_tree(Model())


def test_tinygrad_ignores_unordered_metadata_without_module_state() -> None:
    """Unrelated set members cannot prevent module discovery or capture."""

    class Marker:
        """An identity-hashed model annotation without Tensor state."""

    class Model:
        """Mix one set-owned layer with unnamed metadata markers."""

        def __init__(self) -> None:
            """Create ordinary state beside unrelated unordered metadata."""

            self.weight = Tensor([2.0])
            self.markers = {Marker(), Marker()}
            self.blocks = {Marker(), _NamedSetLayer("scale", "display"), Marker()}
            self.scale = next(item for item in self.blocks if isinstance(item, _NamedSetLayer))

        def __call__(self, x: Any) -> Any:
            """Execute the discovered layer and direct parameter."""

            return self.scale(x) * self.weight

    model = Model()
    tree = discover_tinygrad_module_tree(model)
    assert tree is not None
    assert set(tree.metadata) == {"self", "blocks.0"}
    assert tree.metadata["blocks.0"]["all_addresses"] == ["blocks.0", "scale"]
    trace = tl.trace(model, Tensor([3.0]), backend="tinygrad")
    assert "weight" in trace.param_logs


def test_tinygrad_nonstring_mapping_keys_follow_insertion_order() -> None:
    """Custom-key dict paths keep insertion order even when repr order changes."""

    class Key:
        """Identity key with an unstable display repr."""

        def __init__(self, display: str) -> None:
            """Store the display value."""

            self.display = display

        def __repr__(self) -> str:
            """Expose a display value unrelated to insertion order."""

            return self.display

    class Model:
        """Hold a dict with two nonstring keys."""

        def __init__(self, reverse_display: bool) -> None:
            """Insert equivalent values in a fixed order."""

            first = Key("z" if reverse_display else "a")
            second = Key("a" if reverse_display else "z")
            self.parts = {
                first: _NamedSetLayer("a", "unused"),
                second: _NamedSetLayer("b", "unused"),
            }

        def __call__(self, x: Any) -> Any:
            """Return an input; discovery does not need to execute it."""

            return x

    for reverse_display in (False, True):
        tree = discover_tinygrad_module_tree(Model(reverse_display))
        assert tree is not None
        assert tuple(
            tree.metadata[f"parts.key_{index}"]["_module_object"].name for index in (0, 1)
        ) == ("a", "b")


def test_tinygrad_container_modules_and_parameters_are_discovered() -> None:
    """List, tuple, mapping, and cyclic containers retain module and tensor addresses."""

    class Affine:
        """One tinygrad object module with a trainable tensor."""

        def __init__(self) -> None:
            """Create an identity projection."""

            self.weight = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        def __call__(self, x: Any) -> Any:
            """Apply the projection."""

            return x @ self.weight

    class ContainerModel:
        """Compose nested modules and direct container parameters."""

        def __init__(self) -> None:
            """Build nested and aliased object references."""

            shared = Affine()
            self.layers = [shared, Affine()]
            self.blocks = {"a": Affine(), "b": shared}
            self.parts = (Affine(),)
            self.group = {Affine()}
            self.frozen = frozenset({Affine()})
            self.weights = [Tensor([1.0, 1.0, 1.0]), Tensor([0.0, 0.0, 0.0])]
            self.extra_weights = (self.weights[0],)
            self.loop: list[Any] = []
            self.loop.append(self.loop)

        def __call__(self, x: Any) -> Any:
            """Use both nested modules and container parameters."""

            return (
                self.blocks["a"](self.layers[1](self.layers[0](x))) * self.weights[0]
                + self.weights[1]
            )

    model = ContainerModel()
    tree = discover_tinygrad_module_tree(model)
    assert tree is not None
    assert set(tree.metadata) == {
        "self",
        "layers.0",
        "layers.1",
        "blocks.a",
        "parts.0",
        "group.0",
        "frozen.0",
    }
    assert tree.metadata["self"]["address_children"] == [
        "layers.0",
        "layers.1",
        "blocks.a",
        "parts.0",
        "group.0",
        "frozen.0",
    ]
    assert tree.metadata["layers.0"]["all_addresses"] == ["layers.0", "blocks.b"]
    assert {"weights.0", "weights.1"} <= set(tree.param_owner_by_address)
    assert tree.param_address_by_uop_id[id(model.weights[0].uop)] == "weights.0"
    trace = tl.trace(model, Tensor.ones(1, 3, device="PYTHON"), backend="tinygrad")
    assert {module.address for module in trace.modules} == set(tree.metadata)
    assert {"weights.0", "weights.1"} <= {param.address for param in trace.param_logs}
    assert "extra_weights.0" in trace.param_logs["weights.0"].all_addresses
    assert trace.modules["blocks.b"] is trace.modules["layers.0"]


def test_tinygrad_namedtuple_uses_field_names_for_model_state() -> None:
    """Named module and parameter addresses retain namedtuple field names."""

    class Model:
        """Keep two model layers in a namedtuple."""

        def __init__(self) -> None:
            """Build named child modules."""

            parts_type = namedtuple("Parts", "stem head")
            self.parts = parts_type(_NamedSetLayer("stem", "stem"), _NamedSetLayer("head", "head"))

        def __call__(self, x: Any) -> Any:
            """Run both named layers."""

            return self.parts.head(self.parts.stem(x))

    trace = tl.trace(Model(), Tensor([1.0], device="CPU").realize(), backend="tinygrad")
    assert {module.address for module in trace.modules} == {"self", "parts.stem", "parts.head"}
    assert {param.address for param in trace.param_logs} == {
        "parts.stem.weight",
        "parts.head.weight",
    }


class TinyLinearModel:
    """Simple tinygrad object model with one Linear child."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.fc = nn.Linear(3, 4)
        self.fc.weight = Tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.fc.bias = Tensor([0.0, 0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.fc(x).relu()


class TinyBlock:
    """Nested tinygrad object-model block."""

    def __init__(self) -> None:
        """Initialize the block."""

        self.proj = nn.Linear(3, 4)
        self.proj.weight = Tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.proj.bias = Tensor([0.0, 0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the block.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.proj(x).relu()


class TinyNestedModel:
    """Nested tinygrad model with a compound encoder and leaf head."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.encoder = TinyBlock()
        self.head = nn.Linear(4, 2)
        self.head.weight = Tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        self.head.bias = Tensor([0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.head(self.encoder(x))


class TinySharedModel:
    """tinygrad model reusing one Linear instance under two addresses."""

    def __init__(self) -> None:
        """Initialize the model."""

        shared = nn.Linear(3, 3)
        shared.weight = Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        shared.bias = Tensor([0.0, 0.0, 0.0])
        self.left = shared
        self.right = shared

    def __call__(self, x: Any) -> Any:
        """Run the shared module twice with distinct UOps.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return self.left(x) + self.right(x + 1.0)


class TinyReusedTensorModel:
    """tinygrad model that reuses an already-constructed UOp."""

    def __init__(self) -> None:
        """Initialize the model."""

        self.fc = nn.Linear(3, 3)
        self.fc.weight = Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        self.fc.bias = Tensor([0.0, 0.0, 0.0])

    def __call__(self, x: Any) -> Any:
        """Run the model.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        y = self.fc(x)
        return y + y


def test_tinygrad_simple_linear_uses_object_module_hierarchy() -> None:
    """Simple tinygrad Linear objects should produce object-module logs."""

    model = TinyLinearModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert [module.address for module in trace.modules] == ["self", "fc"]
    assert trace.modules["self"].address_children == ["fc"]
    assert trace.modules["fc"].address_parent == "self"
    assert trace.modules["fc"].params
    assert {param.module_address for param in trace.modules["fc"].params} == {"fc"}
    with pytest.warns(MultiMatchWarning, match="will fan out"):
        fc_labels = trace.resolve_sites(tl.in_module("fc"), max_fanout=16).labels()
    assert fc_labels
    assert all("fc:1" in trace[label].modules for label in fc_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_nested_modules_preserve_address_tree_and_selectors() -> None:
    """Nested tinygrad objects should preserve parent and child addresses."""

    model = TinyNestedModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert {module.address for module in trace.modules} == {
        "self",
        "encoder",
        "encoder.proj",
        "head",
    }
    assert set(trace.modules["self"].address_children) == {"encoder", "head"}
    assert trace.modules["encoder"].address_parent == "self"
    assert trace.modules["encoder"].address_children == ["encoder.proj"]
    assert trace.modules["encoder.proj"].address_parent == "encoder"
    with pytest.warns(MultiMatchWarning, match="will fan out"):
        proj_labels = trace.resolve_sites(tl.in_module("encoder.proj"), max_fanout=16).labels()
    with pytest.warns(MultiMatchWarning, match="will fan out"):
        encoder_labels = trace.resolve_sites(tl.in_module("encoder"), max_fanout=32).labels()
    assert set(proj_labels) < set(encoder_labels)
    assert all("encoder.proj:1" in trace[label].modules for label in proj_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_shared_submodule_aliases_and_multicall() -> None:
    """Shared tinygrad objects should mirror primary-address alias semantics."""

    model = TinySharedModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    shared = trace.modules["left"]
    assert trace.modules["right"] is shared
    assert shared.address == "left"
    assert shared.all_addresses == ["left", "right"]
    assert shared.num_calls == 2
    assert set(shared.ops.keys()) == {1, 2}
    assert shared.call_labels == ["left:1", "left:2"]
    assert trace.modules["self"].address_children == ["left"]
    assert {param.module_address for param in shared.params} == {"left"}
    assert {tuple(param.all_module_addresses) for param in shared.params} == {("left", "right")}
    with pytest.warns(MultiMatchWarning, match="will fan out"):
        left_labels = trace.resolve_sites(tl.in_module("left"), max_fanout=32).labels()
    assert any("left:1" in trace[label].modules for label in left_labels)
    assert any("left:2" in trace[label].modules for label in left_labels)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_reused_uop_keeps_first_construction_attribution() -> None:
    """Reused tinygrad UOps should not invent a second module-call attribution."""

    model = TinyReusedTensorModel()
    trace = tl.trace(model, Tensor.ones(1, 3), backend="tinygrad")

    with pytest.warns(MultiMatchWarning, match="will fan out"):
        fc_labels = trace.resolve_sites(tl.in_module("fc"), max_fanout=16).labels()
    assert fc_labels
    assert trace.modules["fc"].num_calls == 1
    assert all("fc:1" in trace[label].modules for label in fc_labels)
    root_only = [
        op.label
        for op in trace.layer_list
        if op.module == "self:1" and op.layer_type == "add" and not op.is_input
    ]
    assert root_only
    assert check_metadata_invariants(trace) is True


def test_tinygrad_raw_function_traces_remain_function_root() -> None:
    """Raw tinygrad callables should keep the existing function-root module mode."""

    def raw_fn(x: Any) -> Any:
        """Return a raw tinygrad function output.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return (x + 1.0).relu()

    trace = tl.trace(raw_fn, Tensor.ones(3), backend="tinygrad")

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_tinygrad_object_model_can_force_function_root() -> None:
    """Explicit function_root should preserve the old root-only behavior."""

    trace = tl.trace(
        TinyLinearModel(),
        Tensor.ones(1, 3),
        backend="tinygrad",
        module_identity_mode="function_root",
    )

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    assert trace.param_source == "none"
    assert check_metadata_invariants(trace) is True


def test_tinygrad_object_module_requires_discoverable_object() -> None:
    """Explicit object_module should reject raw tinygrad functions."""

    def raw_fn(x: Any) -> Any:
        """Return a raw tinygrad function output.

        Parameters
        ----------
        x
            tinygrad input tensor.

        Returns
        -------
        Any
            tinygrad output tensor.
        """

        return x + 1.0

    with pytest.raises(BackendUnsupportedError, match="object_module.*callable object"):
        tl.trace(
            raw_fn,
            Tensor.ones(3),
            backend="tinygrad",
            module_identity_mode="object_module",
        )


def test_tinygrad_object_module_public_surface_matrix(tmp_path: Path) -> None:
    """Assert executable public surfaces on a real tinygrad object-module trace."""

    trace = tl.trace(TinyLinearModel(), Tensor.ones(1, 3), backend="tinygrad")

    assert trace.module_identity_mode == "object_module"
    assert trace.modules["fc"].params
    assert trace.modules["fc"].forward_args is not None
    assert trace.module_calls["fc:1"].forward_kwargs == {}
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert len(trace.modules.to_pandas()) == len(trace.modules)
    assert len(trace.module_calls["fc:1"].to_pandas()) == 1
    dot = trace.draw(
        vis_outpath=str(tmp_path / "tinygrad_object_graph"),
        vis_save_only=True,
        vis_fileformat="dot",
    )
    assert isinstance(dot, str)
    audit_path = tmp_path / "tinygrad_object_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "tinygrad"
    assert loaded.module_identity_mode == "object_module"
    assert all(op.out is None for op in loaded.layer_list)

    portable_path = tmp_path / "tinygrad_object_portable.tlspec"
    source_out = trace[trace.output_layers[0]].out
    expected = source_out.numpy()
    trace.save(portable_path)
    loaded_portable = tl.load(portable_path)
    loaded_out = loaded_portable[loaded_portable.output_layers[0]].out
    assert loaded_portable.backend == "tinygrad"
    assert loaded_portable.module_identity_mode == "object_module"
    assert getattr(loaded_portable, "payload_load_status") == "loaded_device_best_effort"
    assert isinstance(loaded_out, Tensor)
    assert loaded_out.shape == expected.shape
    assert str(loaded_out.dtype) == str(source_out.dtype)
    np.testing.assert_allclose(loaded_out.numpy(), expected)
    loaded_status = loaded_portable.validate_forward_pass([])
    assert loaded_status is loaded_portable.validation_replay_status
    assert loaded_status.state == "unavailable"
    assert loaded_status.reason == "loaded_trace_runtime_capture_stripped"


def test_tinygrad_object_module_corruption_invariant_fails() -> None:
    """Dropping object-module op attribution should fail metadata invariants."""

    trace = tl.trace(TinyLinearModel(), Tensor.ones(1, 3), backend="tinygrad")
    victim = next(layer for layer in trace.layer_list if layer.module and not layer.is_input)
    victim.module = None
    victim.modules = []
    victim.module_call_stack = []
    victim.output_of_modules = []
    victim.output_of_module_calls = []
    victim.atomic_module_call = None

    with pytest.raises(MetadataInvariantError, match="module_attribution"):
        check_metadata_invariants(trace)
