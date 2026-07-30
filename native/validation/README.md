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

Graph inference and numerical validation are still pending. No native
inference or GPU capability is advertised by this foundation.
