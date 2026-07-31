#pragma once

#include "vulkan.h"

#include <cstdint>

namespace vda_native {

class GpuIo {
public:
    explicit GpuIo(VulkanContext& context);
    void preprocess(
        VulkanBuffer& destination, const VulkanImage& source,
        std::uint32_t size);
    void resize_normalize(
        VulkanImage& destination, const VulkanBuffer& source,
        std::uint32_t source_width, std::uint32_t source_height);

private:
    VulkanContext& context_;
    VulkanPipeline preprocess_;
    VulkanPipeline resize_;
    VulkanPipeline reduce_;
    VulkanPipeline normalize_;
};

}  // namespace vda_native
