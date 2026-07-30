"""Generate deterministic development-only temporal operator fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video_depth_anything.dpt_temporal import DPTHeadTemporal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--module", type=int, default=3, choices=range(4))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    args = parser.parse_args()

    channels = [192, 384, 64, 64][args.module]
    head = DPTHeadTemporal(
        384, 64, False, [48, 96, 192, 384],
        False, 32, "ape")
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    head_state = {
        name.removeprefix("head."): value
        for name, value in state.items()
        if name.startswith("head.")
    }
    head.load_state_dict(head_state, strict=True)
    head.eval()
    generator = torch.Generator().manual_seed(20260730 + args.module)
    value = torch.randn(
        1, channels, args.frames, args.height, args.width,
        generator=generator)
    with torch.inference_mode():
        result, _ = head.motion_modules[args.module](
            value, None, None, None)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    result.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".output.bin"))
    print(json.dumps({
        "module": args.module,
        "shape": list(value.shape),
        "input_sum": float(value.double().sum()),
        "output_sum": float(result.double().sum()),
        "input": str(prefix.with_suffix(".input.bin").resolve()),
        "output": str(prefix.with_suffix(".output.bin").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
