"""Generate a deterministic development-only full graph fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video_depth_anything.dinov2 import DINOv2
from video_depth_anything.dpt_temporal import DPTHeadTemporal


class Reference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = DINOv2("vits")
        self.head = DPTHeadTemporal(
            384, 64, False, [48, 96, 192, 384],
            False, 32, "ape")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, _, height, width = value.shape
        features = self.pretrained.get_intermediate_layers(
            value.flatten(0, 1), [2, 5, 8, 11],
            return_class_token=True)
        depth = self.head(
            features, height // 14, width // 14, frames)[0]
        depth = torch.nn.functional.interpolate(
            depth, size=(height, width),
            mode="bilinear", align_corners=True)
        return torch.relu(depth).squeeze(1).unflatten(
            0, (batch, frames))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--frames", type=int, default=32)
    args = parser.parse_args()

    model = Reference()
    model.load_state_dict(torch.load(
        args.checkpoint, map_location="cpu", weights_only=True),
        strict=True)
    model.eval()
    generator = torch.Generator().manual_seed(20260730)
    value = torch.randn(
        1, args.frames, 3, args.size, args.size,
        generator=generator)
    with torch.inference_mode():
        depth = model(value)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    depth.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".output.bin"))
    print(json.dumps({
        "input_shape": list(value.shape),
        "output_shape": list(depth.shape),
        "minimum": float(depth.min()),
        "maximum": float(depth.max()),
        "mean": float(depth.mean()),
        "sum": float(depth.double().sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
