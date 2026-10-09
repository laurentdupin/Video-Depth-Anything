#include "vulkan.h"
#include "inferbridge/native_harness_vulkan_queue_priority.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <linux_capture_preprocess_spv.h>
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vda_native {
namespace {

std::atomic<std::uint64_t> g_tensor_upload_bytes{0u};
std::atomic<std::uint64_t> g_tensor_download_bytes{0u};

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

#if defined(_WIN32)
bool has_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}
#endif

}  // namespace

struct VulkanSubmission::Resources {
    std::vector<VulkanBatchedDescriptor> descriptor_sets;
    std::vector<VulkanDeferredBuffer> deferred_buffers;
    std::vector<VkCommandBuffer> commands;
    VulkanSemaphore wait;
    VulkanSemaphore signal;
    VulkanSemaphore timeline;
};

#include <inferbridge/vulkan/context/vulkan_context_check.h>

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_semaphore_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_semaphore_destroy_VulkanSemaphore.h>

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_submission_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_submission_destroy_VulkanSubmission.h>

#include <inferbridge/vulkan/context/vulkan_submission_ready.h>

#include <inferbridge/vulkan/context/vulkan_submission_wait.h>

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_buffer_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_buffer_destroy_VulkanBuffer.h>

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_image_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_image_destroy_VulkanImage.h>

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_pipeline_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_destroy_VulkanPipeline.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_set_debug_name.h>

VulkanContext::VulkanContext(
    std::uint32_t device_index,
    bool track_resource_hazards)
    : track_resource_hazards_(track_resource_hazards) {
    try {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Depth Anything V2 C",
        1,
        "VDA",
        1,
        VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        0,
        nullptr,
    };
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
        "vkEnumeratePhysicalDevices");
    if (device_count == 0 || device_index >= device_count) {
        throw std::runtime_error("requested Vulkan device does not exist");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices");
    physical_device_ = devices[device_index];

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    device_name_ = properties.deviceName;
#if defined(_WIN32)
    VkPhysicalDeviceIDProperties identity{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &identity,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    if (identity.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == sizeof(adapter_luid_));
        std::memcpy(
            &adapter_luid_, identity.deviceLUID, VK_LUID_SIZE);
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);

    std::uint32_t extension_count = 0;
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<const char*> enabled_extensions;
#if defined(__linux__) && !defined(__ANDROID__)
    linux_dma_buf_enabled_ = inferbridge::linux_capture::enable_extensions(
        extensions, enabled_extensions);
#endif
#if defined(_WIN32)
    const bool has_external_memory_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    const bool has_external_semaphore_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    if (has_external_memory_win32) {
        const VkPhysicalDeviceExternalBufferInfo external_buffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            nullptr,
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        };
        VkExternalBufferProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(
            physical_device_,
            &external_buffer,
            &external_properties);
        external_capabilities_.d3d12_resource_import =
            (external_properties.externalMemoryProperties
                 .externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const auto image_importable =
            [&](VkFormat format, VkImageUsageFlags usage) {
                const VkPhysicalDeviceExternalImageFormatInfo external{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                    nullptr,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                };
                const VkPhysicalDeviceImageFormatInfo2 image_info{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                    &external,
                    format,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    0,
                };
                VkExternalImageFormatProperties external_properties{
                    VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                };
                VkImageFormatProperties2 image_properties{
                    VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                    &external_properties,
                };
                const VkResult queried =
                    vkGetPhysicalDeviceImageFormatProperties2(
                        physical_device_,
                        &image_info,
                        &image_properties);
                return queried == VK_SUCCESS &&
                    (external_properties.externalMemoryProperties
                         .externalMemoryFeatures &
                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
            };
        external_capabilities_
            .d3d12_bgra8_sampled_image_import =
            image_importable(
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_rgba8_sampled_image_import =
            image_importable(
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_r32_storage_image_import =
            image_importable(
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    if (has_external_semaphore_win32) {
        const VkPhysicalDeviceExternalSemaphoreInfo external_semaphore{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            nullptr,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        };
        VkExternalSemaphoreProperties semaphore_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalSemaphoreProperties(
            physical_device_,
            &external_semaphore,
            &semaphore_properties);
        external_capabilities_.d3d12_fence_import =
            (semaphore_properties.externalSemaphoreFeatures &
             VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#else
    (void)extensions;
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, families.data());
#if defined(_WIN32)
    constexpr bool prefer_dedicated = true;
#else
    constexpr bool prefer_dedicated = false;
#endif
    auto family = inferbridge::native_harness::select_inference_queue_family(families, prefer_dedicated);
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));
    const char* profile_environment =
        std::getenv("VDA_VULKAN_PROFILE");
    profile_dispatches_ =
        profile_environment != nullptr &&
        profile_environment[0] != '\0' &&
        profile_environment[0] != '0' &&
        family->timestampValidBits != 0;
    timestamp_period_ns_ = properties.limits.timestampPeriod;

    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    timeline_features.pNext = &integer_dot_features;
    VkPhysicalDeviceFeatures2 features2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &timeline_features,
    };
    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
    external_capabilities_.timeline_semaphore =
        timeline_features.timelineSemaphore == VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductProperties integer_dot_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 integer_dot_properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &integer_dot_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &integer_dot_properties2);
    packed_int8_dot_supported_ =
        integer_dot_features.shaderIntegerDotProduct == VK_TRUE &&
        integer_dot_properties
            .integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
    integer_dot_features.shaderIntegerDotProduct =
        packed_int8_dot_supported_ ? VK_TRUE : VK_FALSE;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        &timeline_features,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.data(),
        nullptr,
    };
    check(
        inferbridge::native_harness::create_inference_vulkan_device(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
#if defined(_WIN32)
    get_memory_win32_handle_properties_ =
        reinterpret_cast<
            PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_win32_handle_ =
        reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkImportSemaphoreWin32HandleKHR"));
    external_capabilities_.d3d12_resource_import =
        external_capabilities_.d3d12_resource_import &&
        get_memory_win32_handle_properties_ != nullptr;
    external_capabilities_.d3d12_fence_import =
        external_capabilities_.d3d12_fence_import &&
        import_semaphore_win32_handle_ != nullptr;
#endif
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_,
    };
    check(
        vkCreateCommandPool(
            device_, &command_pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16384},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        4096,
        2,
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    if (profile_dispatches_) {
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0,
        };
        check(
            vkCreateQueryPool(
                device_,
                &query_pool_info,
                nullptr,
                &profile_query_pool_),
            "vkCreateQueryPool");
    }
    } catch (...) {
        release();
        throw;
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_VulkanContext.h>

#include <inferbridge/vulkan/context/vulkan_context_release_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_record_profile.h>

#include <inferbridge/vulkan/context/vulkan_context_print_profile_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_find_memory_type.h>

#include <inferbridge/vulkan/context/vulkan_context_create_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_create_device_buffer.h>

#if defined(_WIN32)
#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_image.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_fence.h>

#endif

#include <inferbridge/vulkan/context/vulkan_context_create_host_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_write_host.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_end_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_submit_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_batch.h>

#include <inferbridge/vulkan/context/vulkan_context_release_batch_resources.h>

#include <inferbridge/vulkan/context/vulkan_context_end_batch_async_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_end_batch.h>

#include <inferbridge/vulkan/context/vulkan_context_cancel_batch_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_end_segmented_batch_async.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_segmented_batch.h>

VkCommandBuffer VulkanContext::active_batch_command() {
    if (batch_command_ == VK_NULL_HANDLE && segmented_batch_) {
        batch_command_ = begin_commands();
        segment_has_commands_ = false;
        if (!batch_commands_.empty()) {
            const VkMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            vkCmdPipelineBarrier(
                batch_command_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
            segment_has_commands_ = true;
        }
    }
    return batch_command_;
}

void VulkanContext::finish_segment() {
    if (!segmented_batch_ || batch_command_ == VK_NULL_HANDLE ||
        !segment_has_commands_) return;
    check(vkEndCommandBuffer(batch_command_), "vkEndCommandBuffer(segment)");
    batch_commands_.push_back(batch_command_);
    batch_command_ = VK_NULL_HANDLE;
    segment_has_commands_ = false;
    batch_has_dispatch_ = false;
    batch_buffer_access_.clear();
    batch_image_access_.clear();
    batch_image_layout_.clear();
}

VulkanSemaphore VulkanContext::create_timeline_semaphore(
    std::uint64_t initial_value) {
    if (!external_capabilities_.timeline_semaphore)
        throw std::runtime_error(
            "Vulkan device has no timeline semaphore support");
    const VkSemaphoreTypeCreateInfo timeline_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr,
        VK_SEMAPHORE_TYPE_TIMELINE, initial_value};
    const VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline_info, 0};
    VulkanSemaphore result;
    result.owner_ = this;
    check(vkCreateSemaphore(
        device_, &semaphore_info, nullptr, &result.semaphore_),
        "vkCreateSemaphore(timeline)");
    result.value_ = initial_value;
    return result;
}

void VulkanContext::copy_buffer_raw(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize source_offset,
    VkDeviceSize destination_offset,
    VkDeviceSize bytes) {
    if (!segmented_batch_ && batch_command_ != VK_NULL_HANDLE) {
        const VkMemoryBarrier before{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(
            batch_command_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &before, 0, nullptr, 0, nullptr);
        const VkBufferCopy region{
            source_offset, destination_offset, bytes};
        vkCmdCopyBuffer(batch_command_, source, destination, 1, &region);
        const VkMemoryBarrier after{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(
            batch_command_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &after, 0, nullptr, 0, nullptr);
        batch_has_dispatch_ = true;
        batch_buffer_access_.clear();
        return;
    }
    VkCommandBuffer command = segmented_batch_
        ? active_batch_command()
        : begin_commands();
    const VkBufferCopy region{
        source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    if (segmented_batch_) {
        const VkMemoryBarrier release{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &release, 0, nullptr, 0, nullptr);
        segment_has_commands_ = true;
        finish_segment();
    } else {
        end_commands(command);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_copy.h>

#include <inferbridge/vulkan/context/vulkan_context_upload_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_download_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_transfer_counters.h>

#include <inferbridge/vulkan/context/vulkan_context_acquire_external_buffer_variant_2.h>

void global_transfer_counters(
    std::uint64_t& upload_bytes,
    std::uint64_t& download_bytes) {
    upload_bytes = g_tensor_upload_bytes.load(std::memory_order_relaxed);
    download_bytes = g_tensor_download_bytes.load(std::memory_order_relaxed);
}

#include <inferbridge/vulkan/context/vulkan_context_release_external_buffer_variant_1.h>

#include <inferbridge/linux_capture_vulkan_context.inl>


#include <inferbridge/vulkan/context/vulkan_context_acquire_external_image_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_image_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_3.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_image_to_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_buffer_to_image.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_buffers_to_image.h>

void VulkanContext::dispatch_resources(
    const VulkanPipeline& pipeline,
    const std::vector<VulkanDispatchResource>& resources,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    if (pipeline.owner_ != this ||
        resources.size() != pipeline.descriptor_types_.size() ||
        push_constant_bytes != pipeline.push_constant_bytes_ ||
        (push_constant_bytes && push_constants == nullptr) ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        throw std::invalid_argument("invalid Vulkan dispatch");
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (!pipeline.cached_descriptor_sets_.empty()) {
        descriptor_set = pipeline.cached_descriptor_sets_.back();
        pipeline.cached_descriptor_sets_.pop_back();
    } else {
        const VkDescriptorSetAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool_,
            1,
            &pipeline.descriptor_layout_,
        };
        check(
            vkAllocateDescriptorSets(
                device_, &allocate_info, &descriptor_set),
            "vkAllocateDescriptorSets");
    }
    std::vector<VkDescriptorBufferInfo> buffer_info(resources.size());
    std::vector<VkDescriptorImageInfo> image_info(resources.size());
    std::vector<VkWriteDescriptorSet> writes(resources.size());
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const VkDescriptorType type =
            pipeline.descriptor_types_[index];
        const bool image_descriptor =
            type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (image_descriptor) {
            const VulkanImage* image = resources[index].image;
            if (image == nullptr || image->owner_ != this ||
                image->view_ == VK_NULL_HANDLE ||
                (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                 image->sampler_ == VK_NULL_HANDLE)) {
                pipeline.cached_descriptor_sets_.push_back(
                    descriptor_set);
                throw std::invalid_argument(
                    "foreign or invalid Vulkan image");
            }
            image_info[index] = {
                type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    ? image->sampler_
                    : VK_NULL_HANDLE,
                image->view_,
                type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            writes[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                static_cast<std::uint32_t>(index),
                0,
                1,
                type,
                &image_info[index],
                nullptr,
                nullptr,
            };
            continue;
        }
        const VulkanBuffer* buffer = resources[index].buffer;
        if (buffer == nullptr || buffer->owner_ != this) {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
            throw std::invalid_argument("foreign Vulkan buffer");
        }
        buffer_info[index] = {
            buffer->buffer_,
            0,
            buffer->size_,
        };
        writes[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(index),
            0,
            1,
            type,
            nullptr,
            &buffer_info[index],
            nullptr,
        };
    }
    vkUpdateDescriptorSets(
        device_,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    const bool batched =
        batch_command_ != VK_NULL_HANDLE || segmented_batch_;
    if (batched && wait != nullptr) {
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
        throw std::invalid_argument(
            "external wait is not supported inside a Vulkan batch");
    }
    VkCommandBuffer command =
        batched ? active_batch_command() : begin_commands();
    const bool profile =
        !batched && profile_query_pool_ != VK_NULL_HANDLE;
    if (profile) {
        vkCmdResetQueryPool(
            command, profile_query_pool_, 0, 2);
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            profile_query_pool_,
            0);
    }
    if (batched && batch_has_dispatch_ &&
        !track_resource_hazards_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr);
    } else if (batched) {
        std::unordered_map<VkBuffer, VkAccessFlags>
            current_buffers;
        std::unordered_map<VkImage, VkAccessFlags>
            current_images;
        std::unordered_map<VkImage, VkImageLayout> image_layouts;
        for (std::size_t index = 0;
             index < resources.size();
             ++index) {
            const VkAccessFlags access =
                pipeline.descriptor_access_[index];
            if (resources[index].buffer != nullptr) {
                current_buffers[
                    resources[index].buffer->buffer_] |= access;
            } else {
                const VkImage image =
                    resources[index].image->image_;
                current_images[image] |= access;
                image_layouts[image] =
                    pipeline.descriptor_types_[index] ==
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        std::vector<VkBufferMemoryBarrier> buffer_barriers;
        std::vector<VkImageMemoryBarrier> image_barriers;
        for (const auto& [buffer, access] : current_buffers) {
            const auto previous =
                batch_buffer_access_.find(buffer);
            if (previous != batch_buffer_access_.end() &&
                ((previous->second | access) &
                 VK_ACCESS_SHADER_WRITE_BIT) != 0) {
                buffer_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        buffer,
                        0,
                        VK_WHOLE_SIZE,
                    });
            }
            batch_buffer_access_[buffer] = access;
        }
        for (const auto& [image, access] : current_images) {
            const auto previous =
                batch_image_access_.find(image);
            const VkImageLayout layout = image_layouts[image];
            if (previous != batch_image_access_.end()) {
                const VkImageLayout previous_layout =
                    batch_image_layout_.at(image);
                if (((previous->second | access) &
                     VK_ACCESS_SHADER_WRITE_BIT) == 0 &&
                    previous_layout == layout) {
                    batch_image_access_[image] = access;
                    continue;
                }
                image_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        previous_layout,
                        layout,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        image,
                        {
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            0,
                            1,
                            0,
                            1,
                        },
                    });
            }
            batch_image_access_[image] = access;
            batch_image_layout_[image] = layout;
        }
        if (!buffer_barriers.empty() ||
            !image_barriers.empty()) {
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<std::uint32_t>(
                    buffer_barriers.size()),
                buffer_barriers.data(),
                static_cast<std::uint32_t>(
                    image_barriers.size()),
                image_barriers.data());
        }
    }
    vkCmdBindPipeline(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
    vkCmdBindDescriptorSets(
        command,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout_,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    if (push_constant_bytes) {
        vkCmdPushConstants(
            command,
            pipeline.layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            push_constant_bytes,
            push_constants);
    }
    vkCmdDispatch(command, group_x, group_y, group_z);
    if (segmented_batch_) {
        const VkMemoryBarrier release{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(
            command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &release, 0, nullptr, 0, nullptr);
        segment_has_commands_ = true;
    }
    if (profile) {
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_query_pool_,
            1);
    }
    if (batched) {
        batch_has_dispatch_ = true;
        batch_descriptor_sets_.push_back(
            {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
        if (segmented_batch_) finish_segment();
    } else {
        end_commands(command, wait);
        if (profile) {
            std::uint64_t timestamps[2]{};
            check(
                vkGetQueryPoolResults(
                    device_,
                    profile_query_pool_,
                    0,
                    2,
                    sizeof(timestamps),
                    timestamps,
                    sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT |
                        VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");
            record_profile(
                pipeline, timestamps[1] - timestamps[0]);
        }
        pipeline.cached_descriptor_sets_.push_back(descriptor_set);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_6.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_5.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_2.h>

void VulkanContext::release_submission_resources(
    VulkanSubmission& submission) noexcept {
    if (submission.resources_ == nullptr) return;
    if (!submission.resources_->commands.empty()) {
        vkFreeCommandBuffers(
            device_, command_pool_,
            static_cast<std::uint32_t>(
                submission.resources_->commands.size()),
            submission.resources_->commands.data());
    }
    for (const VulkanBatchedDescriptor& descriptor :
         submission.resources_->descriptor_sets) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    for (const VulkanDeferredBuffer& buffer :
         submission.resources_->deferred_buffers) {
        recycle_or_destroy(buffer);
    }
    delete submission.resources_;
    submission.resources_ = nullptr;
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_recycle_or_destroy.h>

#include <inferbridge/vulkan/context/vulkan_context_discard.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_3.h>

}  // namespace vda_native
