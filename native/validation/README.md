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
that different valid spatial sizes execute accurately. The scalar executor
remains the numerical oracle.

## Full native Vulkan graph

ABI version 2 adds `vda_create_vulkan`. It selects a zero-based Vulkan
physical-device index and fails if a real Vulkan context cannot be created;
there is no CPU fallback behind this entry point. The complete graph stays on
the selected GPU between the input upload and final depth download:

- batched DINOv2 patch embedding and all 12 spatial transformer blocks;
- bounded encoder frame chunks that cap attention scratch memory and avoid
  oversized Windows watchdog submissions;
- all four temporal modules, including GroupNorm, two eight-head attention
  residuals, exact-GELU GEGLU feed-forward, and layout transforms;
- all DPT projections, resize layers, refinement blocks, and output head.

Full-graph comparisons against the same PyTorch CPU fixtures:

| GPU | Input | Relative L1 | Maximum absolute |
|---|---:|---:|---:|
| Radeon RX 9070 | 32 x 28 x 28 | 0.141551% | 0.034287 |
| GeForce GTX 1080 | 32 x 28 x 28 | 0.160712% | 0.0374296 |
| Radeon RX 6700 XT | 32 x 28 x 28 | 0.152964% | 0.0375371 |
| Radeon RX 9070 | 32 x 56 x 56 | 0.0353831% | 0.00282288 |

Twenty consecutive 32 x 28 x 28 jobs on one persistent RX 9070 context pass
with a 58.4 ms median. A 32 x 518 x 518 full-graph smoke completes without
GPU watchdog or allocation failure on all three adapters: 2.61 s on the
RX 9070, 8.29 s on the GTX 1080, and 3.26 s on the RX 6700 XT.

This first coherent Vulkan API is a host-tensor compatibility path: normalized
input is uploaded once and final depth is downloaded once. It does not yet
advertise external texture import or GPU-resident output leasing.

## InferBridge stateful streaming gate

ABI 3 adds `vda_infer_stream_bgra8_f32` and `vda_stream_reset` while
preserving the independent 32-frame tensor entry point. The stream API matches
InferBridge's worker rather than substituting clip inference:

- BGRA's first three BGR bytes retain their current channel ordering;
- frames use the worker's nearest square resize and ImageNet normalization;
- the first frame performs the worker's seed pass and cached query pass;
- all eight temporal attention-block inputs remain GPU-resident;
- each new frame uses the exact first-two plus latest-29 cache selection;
- cache deletion begins at the same frame ID and reset restores first-frame
  behavior;
- depth is align-corners bilinear resized and min/max normalized at the source
  dimensions.

A 15-frame 37x29 sequence at network size 28 crosses the cache-deletion
threshold on the RX 9070. Maximum normalized-depth deviation from Python CPU
was `0.001187` and reset reproduced frame zero with `0.000405` maximum error.
Three-frame plus reset canaries passed on the GTX 1080 and RX 6700 XT with
maximum errors of `0.000809` and `0.000815`, respectively. The original
32-frame fixtures also retain their prior results on all adapters.

The cache implementation exposed and fixed a Vulkan command-ordering issue:
buffer copies inside a compute batch now record transfer barriers and the copy
in that same command buffer instead of submitting ahead of their producers.
`native/tools/validate_streaming.py` reproduces the stateful comparison.

## Embedded InferBridge harness

The model DLL exports `ibrh_get_api` for InferBridge harness ABI 1.0. The
single Video Depth Anything Small catalog entry maps its canonical `.pth` to
the hidden, content-addressed `.vda` representation described above. The
harness accepts the existing `Encoder=vits` and multiple-of-14 `Size`
parameters; it rejects unimplemented encoder variants rather than silently
selecting the wrong graph.

Each host-memory BGRA8 submission advances the model's real temporal cache and
returns a leased, source-size normalized FP32 depth image with preserved frame
and timestamp correlation. Submit calls are serialized because stream order
is semantically significant. `Reset=YES` on a submission resets the cache
immediately before that frame; model unload/reload also restores initial
state. An acquired output lease owns its result independently of later stream
advances and remains valid after job release.

Capability probing reports only host input/output and one synchronous
in-flight job. The spatial graph, temporal cache, and DPT graph execute on the
selected Vulkan device, but capture upload and final depth readback remain
host boundaries. External GPU resources, asynchronous completion, and
cancellation are not advertised.

The Windows Release harness gate validates model loading, three correlated
frames, temporal-state progression, explicit reset, normalized output, and
lease lifetime on the RX 9070. Reset reproduces the first-frame result within
`1e-6`. The longer Python CPU stream and three-GPU numerical gates remain as
reported above.
