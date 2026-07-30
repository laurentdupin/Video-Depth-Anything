"""Validate the stateful InferBridge stream against Python CPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def python_frame(image: np.ndarray, model, size: int) -> np.ndarray:
    model.id += 1
    height, width = image.shape[:2]
    mean = torch.tensor([0.485, 0.456, 0.406])
    deviation = torch.tensor([0.229, 0.224, 0.225])
    value = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    value = functional.interpolate(value, (size, size))
    value = value.permute(0, 2, 3, 1)[0].float() / 255.0
    value = ((value - mean) / deviation).permute(2, 0, 1)
    value = value.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        feature = model.forward_features(value)
    shape = value.shape
    current = (
        model.frame_cache_list[0:2] +
        model.frame_cache_list[-29:])
    if not current:
        with torch.inference_mode():
            _, seed = model.forward_depth(feature, shape)
        model.frame_cache_list = [seed] * 32
        model.frame_id_list.extend([0] * 31)
        current = (
            model.frame_cache_list[0:2] +
            model.frame_cache_list[-29:])
    cache = [
        torch.cat([entry[index] for entry in current], dim=1)
        for index in range(len(current[0]))]
    with torch.inference_mode():
        depth, new_cache = model.forward_depth(
            feature, shape, cached_hidden_state_list=cache)
    depth = functional.interpolate(
        depth.flatten(0, 1).unsqueeze(1),
        size=(height, width), mode="bilinear",
        align_corners=True)
    result = depth[-1, 0].numpy()
    result = (result - result.min()) / (result.max() - result.min())
    model.frame_cache_list.append(new_cache)
    model.frame_id_list.append(model.id)
    if model.id + 32 > model.gap + 1:
        del model.frame_id_list[1]
        del model.frame_cache_list[1]
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--width", type=int, default=37)
    parser.add_argument("--height", type=int, default=29)
    args = parser.parse_args()

    torchvision_library = torch.library.Library("torchvision", "DEF")
    torchvision_library.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    try:
        import easydict  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("easydict")

        class EasyDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        module.EasyDict = EasyDict
        sys.modules["easydict"] = module
    sys.path.insert(0, str(args.repo.resolve()))
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    model = VideoDepthAnything(
        encoder="vits", features=64,
        out_channels=[48, 96, 192, 384])
    archive = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(archive)
    model.eval()

    rng = np.random.default_rng(20260730)
    frames = rng.integers(
        0, 256,
        (args.frames, args.height, args.width, 4),
        dtype=np.uint8)
    frames[:, :, :, 3] = 255
    references = [
        python_frame(frame[:, :, :3], model, args.size)
        for frame in frames]

    library = ctypes.CDLL(str(args.dll.resolve()))
    library.vda_create_vulkan.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p)]
    library.vda_create_vulkan.restype = ctypes.c_int
    library.vda_infer_stream_bgra8_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64]
    library.vda_infer_stream_bgra8_f32.restype = ctypes.c_int
    library.vda_stream_reset.argtypes = [ctypes.c_void_p]
    library.vda_stream_reset.restype = ctypes.c_int
    library.vda_last_error.restype = ctypes.c_char_p
    library.vda_destroy.argtypes = [ctypes.c_void_p]

    context = ctypes.c_void_p()
    status = library.vda_create_vulkan(
        str(args.native_model.resolve()).encode(), 0, args.device,
        ctypes.byref(context))
    if status:
        raise RuntimeError(library.vda_last_error().decode())
    reports = []
    try:
        for index, frame in enumerate(frames):
            actual = np.empty(
                (args.height, args.width), dtype=np.float32)
            status = library.vda_infer_stream_bgra8_f32(
                context,
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                frame.strides[0], args.width, args.height, args.size,
                actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                actual.size)
            if status:
                raise RuntimeError(library.vda_last_error().decode())
            difference = np.abs(actual - references[index])
            reports.append({
                "frame": index,
                "maximum_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
            })
        if library.vda_stream_reset(context):
            raise RuntimeError(library.vda_last_error().decode())
        reset_actual = np.empty(
            (args.height, args.width), dtype=np.float32)
        status = library.vda_infer_stream_bgra8_f32(
            context,
            frames[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            frames[0].strides[0], args.width, args.height, args.size,
            reset_actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            reset_actual.size)
        if status:
            raise RuntimeError(library.vda_last_error().decode())
        reset_difference = np.abs(reset_actual - references[0])
        reports.append({
            "frame": "reset-0",
            "maximum_absolute_error": float(reset_difference.max()),
            "mean_absolute_error": float(reset_difference.mean()),
        })
    finally:
        library.vda_destroy(context)
    print(json.dumps(reports, indent=2))
    if max(item["maximum_absolute_error"] for item in reports) > 0.01:
        raise SystemExit("VDA streaming accuracy gate failed")


if __name__ == "__main__":
    main()
