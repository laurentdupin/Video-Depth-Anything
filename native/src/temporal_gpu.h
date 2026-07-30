#pragma once

#include "dpt_gpu.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"

#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace vda_native {

struct TemporalFrameCache {
    std::array<VulkanBuffer, 2> attention;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
};

class TemporalGpu {
public:
    TemporalGpu(
        VulkanContext& context,
        GpuModel& weights,
        VulkanOperators& operators);

    FeatureMap forward(
        std::uint32_t module_index,
        FeatureMap&& input);
    FeatureMap forward_stream(
        std::uint32_t module_index,
        FeatureMap&& input,
        const std::vector<const TemporalFrameCache*>& history,
        TemporalFrameCache& output_cache);

private:
    FeatureMap forward_impl(
        std::uint32_t module_index,
        FeatureMap&& input,
        const std::vector<const TemporalFrameCache*>* history,
        TemporalFrameCache* output_cache);
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
    VulkanPipeline attention_stream_;
    VulkanPipeline geglu_;
    VulkanPipeline output_;
};

}  // namespace vda_native
