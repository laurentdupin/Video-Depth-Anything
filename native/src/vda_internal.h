#pragma once

#include "external_gpu.h"

#include <memory>

struct vda_context;

namespace vda_native {

std::shared_ptr<ExternalGpu> create_metal_external_gpu(vda_context* context);

}  // namespace vda_native
