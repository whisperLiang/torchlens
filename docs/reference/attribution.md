# `tl.attribution` reference

`tl.attribution` provides input and intermediate-layer attribution methods directly on
PyTorch modules. Each call returns `AttributionResult(method, values, target_repr, extra)`.
`target` is an output-class index or a callable that maps model output to one scalar tensor;
the layer methods name a module from `dict(model.named_modules())`.

The stage-4b compute-kit spellings (Integrated Gradients completeness fields,
`occlusion`, and Grad-CAM overlay fields) are **DOCUMENTED-UNSTABLE**: they are
provisional pending the naming session and may change without a deprecation shim. The exact
inventory is locked in the [glossary](glossary.md#documented-unstable-attribution-kit-index).

The examples use a deterministic small classifier. Input methods return a tensor shaped like
the attributed input; layer methods return the named layer's activation shape. For convolutional
maps, [`grad_cam`](#grad_cam) returns an input-resolution `N, 1, H, W` map.

Passing the same tensor object to several input slots preserves that identity inside the
attribution forward (`a is b` control flow runs exactly as in your own call), and every
occurrence slot reports the shared tensor's full accumulated gradient. Path methods require
repeated references to carry identical baselines. A module fired several times contributes
through every firing: `layer_attribution`, `layer_integrated_gradients`, and
`layer_conductance` total the per-firing terms, while `grad_cam` requires the target layer to
fire exactly once and raises `AttributionError` otherwise.

## `saliency`

`tl.attribution.saliency(model, inputs, input_kwargs=None, *, target=...)` returns absolute
input gradients for the selected scalar target.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.saliency(model, torch.tensor([[1.0, 2.0]]), target=0))
```

Output:

```text
AttributionResult(method='saliency', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=[])
```

## `input_x_grad`

`tl.attribution.input_x_grad(model, inputs, input_kwargs=None, *, target=...)` returns the
gradient times each attributed input value. Compare it with [`saliency`](#saliency) when the
input magnitude should be part of the score.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.input_x_grad(model, torch.tensor([[1.0, 2.0]]), target=0))
```

Output:

```text
AttributionResult(method='input_x_grad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=[])
```

## `integrated_gradients`

`tl.attribution.integrated_gradients(model, inputs, input_kwargs=None, *, target=...,
n_steps=50, baseline=None)` integrates input gradients along a straight path from `baseline`
(zeros by default). Baseline choice changes the interpretation. Every result computes the
endpoint `target_delta = f(input) - f(baseline)` and reports `attribution_sum` plus the signed
`completeness_residual = attribution_sum - target_delta`; callers can therefore inspect the IG
axiom directly instead of trusting an attribution array without its convergence evidence.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.integrated_gradients(model, torch.tensor([[1.0, 2.0]]), target=0, n_steps=4))
```

Output:

```text
AttributionResult(method='integrated_gradients', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['attribution_sum', 'baseline', 'completeness_residual', 'n_steps', 'target_delta'])
```

## `occlusion`

`tl.attribution.occlusion(trace, selection, *, score=..., baseline="zeros",
blur_kernel_size=3)` resolves an ACT Selection, applies the occlusion through a temporary
`Trace.fork().do(...)`, and returns `score(original) - score(occluded)`. The scorer receives the
Trace, so the output projection is explicit (for example,
`score=lambda run: run.output_ops[0].out[..., 0].sum()`).

The baseline is always named and disclosed. `"zeros"` uses `tl.zero_ablate()` and is the
documented default; `"mean"` uses `tl.mean_ablate()`; `"blur"` uses a same-shaped spatial mean
blur before the Selection engine scatters the chosen region. These counterfactuals are not
interchangeable, and the result records the endpoint scores, Selection digest, and chosen policy.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(
    nn.ReLU(),
    torch.tensor([[-1.0, 2.0]]),
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
region = tl.units(trace.input_ops[0].label, [(0, 1)])
result = tl.attribution.occlusion(
    trace,
    region,
    score=lambda run: run.output_ops[0].out.sum(),
    baseline="zeros",
)
print(result)
trace.cleanup()
```

## `smoothgrad`

`tl.attribution.smoothgrad(model, inputs, input_kwargs=None, *, target=..., n_samples=25,
noise_level=0.1, seed=None)` averages saliency across Gaussian-noised inputs. Supplying `seed`
makes the noise deterministic without changing the global torch RNG state.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.smoothgrad(model, torch.tensor([[1.0, 2.0]]), target=0, n_samples=3, seed=0))
```

Output:

```text
AttributionResult(method='smoothgrad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['n_samples', 'noise_level', 'seed'])
```

## `grad_cam`

`tl.attribution.grad_cam(model, inputs, input_kwargs=None, *, target=..., layer=..., relu=True,
overlay=True, image=None, alpha=0.6, cmap="magma")`
forms a Grad-CAM map from a 4D `N, C, H, W` convolution-style layer and upsamples it to the
spatial size of the input that actually feeds the target layer (proven through the autograd
graph, not taken from argument order). Use a layer name from `model.named_modules()`. If no
spatial input feeds the layer, or several feed it with different grids, the call raises
`AttributionError` instead of guessing a coordinate system.

The values remain the input-resolution interpolated tensor for compatibility, but
`native_map_resolution` and `rendered_map_resolution` are separate metadata. The default rendered
overlay uses the same heatmap blending path as `receptive_field.show()` and carries a visible
footer such as `CAM native map: 7x7; display interpolated`. Interpolation is presentation, not new
measured precision.

```python
import torch
from torch import nn
import torchlens as tl

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(1, 2, 1); self.head = nn.Linear(8, 2)
    def forward(self, x): return self.head(torch.relu(self.conv(x)).flatten(1))

print(tl.attribution.grad_cam(ImageModel(), torch.ones(1, 1, 2, 2), target=0, layer="conv"))
```

Output:

```text
AttributionResult(method='grad_cam', values=Tensor(shape=(1, 1, 2, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'native_map_resolution', 'overlay', 'relu', 'rendered_map_resolution', 'upsampling'])
```

## `layer_integrated_gradients`

`tl.attribution.layer_integrated_gradients(model, inputs, input_kwargs=None, *, target=...,
layer=..., baseline=None, n_steps=50)` applies the integrated-gradients path rule to a named
intermediate activation. See [`integrated_gradients`](#integrated_gradients) for input-level IG.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_integrated_gradients(model, torch.ones(1, 2), target=0, layer="0", n_steps=4))
```

Output:

```text
AttributionResult(method='layer_integrated_gradients', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'n_steps'])
```

## `layer_conductance`

`tl.attribution.layer_conductance(model, inputs, input_kwargs=None, *, target=..., layer=...,
baseline=None, n_steps=50)` decomposes input integrated gradients onto the selected hidden
units along the input path. It uses the same baseline and step controls as
[`layer_integrated_gradients`](#layer_integrated_gradients).

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_conductance(model, torch.ones(1, 2), target=0, layer="0", n_steps=4))
```

Output:

```text
AttributionResult(method='layer_conductance', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'n_steps'])
```

## `layer_attribution`

`tl.attribution.layer_attribution(model, inputs, input_kwargs=None, *, target=..., layer=...,
method="activation_x_grad")` returns either activation-times-gradient or absolute gradient for a
named layer. It is the one-step layer counterpart to the path methods above.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_attribution(model, torch.ones(1, 2), target=0, layer="0"))
```

Output:

```text
AttributionResult(method='layer_activation_x_grad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer'])
```
