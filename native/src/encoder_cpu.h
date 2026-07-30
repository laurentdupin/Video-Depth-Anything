#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace vda_native {

struct EncoderCpuOutput {
    std::uint32_t frames = 0;
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::uint32_t embedding = 384;
    std::vector<float> patch_tokens;
    std::vector<float> prepared_tokens;
    std::vector<std::vector<float>> features;
};

EncoderCpuOutput encoder_cpu(
    const ModelFile& model,
    const float* normalized_rgb_tchw,
    std::uint32_t frames,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace vda_native
