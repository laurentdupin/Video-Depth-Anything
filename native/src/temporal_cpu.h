#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace vda_native {

void temporal_module_cpu(
    const ModelFile& model,
    std::uint32_t module_index,
    const float* input_cfhw,
    std::uint32_t channels,
    std::uint32_t frames,
    std::uint32_t height,
    std::uint32_t width,
    std::vector<float>& output_cfhw);

}  // namespace vda_native
