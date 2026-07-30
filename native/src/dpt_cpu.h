#pragma once

#include "encoder_cpu.h"
#include "model.h"

#include <vector>

namespace vda_native {

std::vector<float> dpt_cpu(
    const ModelFile& model,
    EncoderCpuOutput&& encoded);

}  // namespace vda_native
