#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "resize_depth_spv.h"
#include "reduce_minmax_spv.h"
#include "normalize_depth_image_spv.h"
#include "copy_depth_image_spv.h"

#include <stdexcept>

namespace vda_native {

GpuIo::GpuIo(VulkanContext& context)
    : context_(context),
      preprocess_(context.create_pipeline(
          vda_preprocess_texture_spv, vda_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}, 12)),
      resize_(context.create_pipeline(
          vda_resize_depth_spv, vda_resize_depth_spv_size, 2, 16)),
      reduce_(context.create_pipeline(
          vda_reduce_minmax_spv, vda_reduce_minmax_spv_size, 2, 4)),
      normalize_(context.create_pipeline(
          vda_normalize_depth_image_spv,
          vda_normalize_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_ACCESS_SHADER_READ_BIT}, 8)),
      copy_(context.create_pipeline(
          vda_copy_depth_image_spv,
          vda_copy_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}, 8)) {
    preprocess_.set_debug_name("stream_preprocess_texture");
    resize_.set_debug_name("stream_resize_depth");
    reduce_.set_debug_name("stream_reduce_minmax");
    normalize_.set_debug_name("stream_normalize_depth_image");
    copy_.set_debug_name("stream_copy_depth_image");
}

void GpuIo::preprocess(
    VulkanBuffer& destination, const VulkanImage& source,
    std::uint32_t size) {
    if (size == 0u || destination.size() <
        static_cast<std::uint64_t>(size) * size * 3u * sizeof(float))
        throw std::invalid_argument("invalid VDA GPU preprocess dimensions");
    struct Parameters { std::uint32_t width, height, size; } parameters{
        source.width(), source.height(), size};
    context_.dispatch_image_to_buffer(
        preprocess_, source, destination, &parameters, sizeof(parameters),
        (size + 7u) / 8u, (size + 7u) / 8u);
}

void GpuIo::resize_depth(
    VulkanImage& destination, const VulkanBuffer& source,
    std::uint32_t source_width, std::uint32_t source_height,
    bool normalize) {
    const std::uint32_t width = destination.width();
    const std::uint32_t height = destination.height();
    if (source_width == 0u || source_height == 0u ||
        width == 0u || height == 0u ||
        destination.format() != VK_FORMAT_R32_SFLOAT)
        throw std::invalid_argument("invalid VDA GPU depth output dimensions");
    VulkanBuffer resized = context_.create_device_buffer(
        static_cast<std::uint64_t>(width) * height * sizeof(float));
    struct ResizeParameters {
        std::uint32_t source_width, source_height, width, height;
    } resize_parameters{source_width, source_height, width, height};
    context_.dispatch(
        resize_, {&source, &resized}, &resize_parameters,
        sizeof(resize_parameters), (width + 7u) / 8u, (height + 7u) / 8u);
    struct OutputParameters { std::uint32_t width, height; } output_parameters{
        width, height};
    if (normalize) {
        VulkanBuffer range = context_.create_device_buffer(2u * sizeof(float));
        const std::uint32_t count = width * height;
        context_.dispatch(
            reduce_, {&resized, &range}, &count, sizeof(count), 1u);
        context_.dispatch_buffers_to_image(
            normalize_, {&resized, &range}, destination,
            &output_parameters, sizeof(output_parameters),
            (width + 7u) / 8u, (height + 7u) / 8u);
    } else {
        context_.dispatch_buffers_to_image(
            copy_, {&resized}, destination,
            &output_parameters, sizeof(output_parameters),
            (width + 7u) / 8u, (height + 7u) / 8u);
    }
}

}  // namespace vda_native
