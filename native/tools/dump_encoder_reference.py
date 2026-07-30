"""Generate deterministic development-only DINOv2 encoder fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video_depth_anything.dinov2 import DINOv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--size", type=int, default=28)
    args = parser.parse_args()

    model = DINOv2("vits")
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    encoder_state = {
        name.removeprefix("pretrained."): value
        for name, value in state.items()
        if name.startswith("pretrained.")
    }
    model.load_state_dict(encoder_state, strict=True)
    model.eval()
    generator = torch.Generator().manual_seed(20260730)
    value = torch.randn(
        1, 3, args.size, args.size, generator=generator)
    with torch.inference_mode():
        patch = model.patch_embed(value)
        prepared = model.prepare_tokens_with_masks(value)
        features = model.get_intermediate_layers(
            value, [2, 5, 8, 11], return_class_token=False)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    patch.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".patch.bin"))
    prepared.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".prepared.bin"))
    for index, feature in enumerate(features):
        feature.numpy().astype(np.float32).tofile(
            prefix.with_suffix(f".feature{index}.bin"))
    print(json.dumps({
        "input_shape": list(value.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_sums": [
            float(feature.double().sum()) for feature in features],
    }, indent=2))


if __name__ == "__main__":
    main()
