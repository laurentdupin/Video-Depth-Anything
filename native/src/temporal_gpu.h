#pragma once

#include "dpt_gpu.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <string>

namespace vda_native {

class TemporalGpu {
public:
    TemporalGpu(
        VulkanContext& context,
        GpuModel& weights,
        VulkanOperators& operators);

    FeatureMap forward(
        std::uint32_t module_index,
        FeatureMap&& input);

private:
    const VulkanBuffer& weight(const std::string& name) const;
    std::string prefix(std::uint32_t module_index) const;

    VulkanContext& context_;
    GpuModel& weights_;
    VulkanOperators& operators_;
    VulkanBuffer zero_bias_;
    VulkanPipeline group_norm_;
    VulkanPipeline transpose_;
    VulkanPipeline position_;
    VulkanPipeline attention_;
    VulkanPipeline geglu_;
    VulkanPipeline output_;
};

}  // namespace vda_native
