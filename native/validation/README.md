# Video Depth Anything Small native validation

The correctness-first native target is the Apache-2.0 licensed
`video_depth_anything_vits` checkpoint and the fixed 32-frame relative-depth
inference graph.

## Canonical model boundary

- Canonical file: `video_depth_anything_vits.pth`
- Canonical SHA-256:
  `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609`
- Canonical bytes: 116,440,756
- Derived format: `VDA1` version 1
- Converter: `vda-export-pytorch-weights-v1`
- Expected inference tensors: 351 FP32 tensors

The canonical `.pth` remains the shared downloaded artifact. A hidden,
content-addressed `.vda` is the deployment-safe representation. The converter
uses PyTorch's restricted `weights_only=True` loader during development and
rejects non-tensors, non-FP32 values, unsupported ranks, long names, and
invalid lengths. The eventual native DLL will only map `.vda`; it will not
contain Python or a pickle implementation.

## Temporal transformer gate

The dependency-free scalar C++ oracle now implements the four learned
temporal modules: GroupNorm, temporal positional encoding, two eight-head
self-attention residual blocks, LayerNorm, exact-GELU GEGLU feed-forward,
input/output projections, and the outer residual.

All four modules were compared at their representative 32-frame decoder
shapes against PyTorch CPU:

| Module | C x T x H x W | Relative L1 | Maximum absolute |
|---:|---:|---:|---:|
| 0 | 192 x 32 x 2 x 2 | 0.0000259% | 0.00000191 |
| 1 | 384 x 32 x 1 x 1 | 0.0000743% | 0.0000119 |
| 2 | 64 x 32 x 2 x 2 | 0.0000194% | 0.00000858 |
| 3 | 64 x 32 x 4 x 4 | 0.0000269% | 0.00000739 |

The raw fixtures are generated locally and excluded from source control.

## Spatial encoder gate

The scalar oracle also implements the complete ViT-S/14 DINOv2 encoder:
patch projection, offset-aware bicubic position interpolation, twelve
pre-normalized transformer blocks, six-head attention, LayerScale, exact GELU
MLP, and normalized captures at blocks 2, 5, 8, and 11.

For a deterministic 28x28 tensor, the prepared tokens match PyTorch CPU at
`5.04e-7` relative L1. The worst of the four captured features is
`3.13e-6` relative L1 (`0.000313%`) with maximum absolute error
`0.0000725`.

## Full native CPU graph

The dependency-free DLL now connects the encoder and all four temporal
modules through the complete DPT projection, transpose-convolution,
refinement, align-corners bilinear, and depth head. Its public tensor API
accepts the official 32-frame normalized RGB TCHW input and returns THW
relative depth for spatial dimensions that are multiples of 14.

Deterministic complete-DLL comparisons against PyTorch CPU:

| Input | Relative L1 | Maximum absolute | Scalar native time |
|---:|---:|---:|---:|
| 32 x 28 x 28 | 0.0000457% | 0.00000548 | 4.07 s |
| 32 x 56 x 56 | 0.0000346% | 0.00000596 | 16.24 s |

These correctness-first results are far inside the 1% requirement and prove
that different valid spatial sizes execute accurately. The scalar executor is
an oracle, not the performance backend: official 518-class inference must be
translated to Vulkan before it is practical. No GPU capability is advertised.
