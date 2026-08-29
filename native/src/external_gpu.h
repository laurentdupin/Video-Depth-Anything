#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "video_depth_anything_native.h"

namespace vda_native {

struct ExternalGpuCapabilities {
    bool available = false;
    std::uint64_t adapter_luid = 0;
    std::uint32_t maximum_in_flight_jobs = 0;
};
struct ExternalTextureRequest {
    std::uintptr_t shared_texture_handle = 0;
    std::uint64_t shared_texture_identity = 0;
    std::uint32_t width = 0, height = 0, process_resolution = 0;
    std::uintptr_t wait_fence_handle = 0;
    std::uint64_t wait_fence_value = 0;
    std::uintptr_t output_texture_handle = 0;
    std::uint64_t output_texture_identity = 0;
    std::uint32_t output_width = 0, output_height = 0;
    std::uintptr_t signal_fence_handle = 0;
    std::uint64_t signal_fence_value = 0;
    std::uint64_t source_frame_id = 0, timestamp_ns = 0;
    bool reset = false;
};
enum class ExternalJobState { running, complete, cancelled };
class ExternalJob {
public:
    virtual ~ExternalJob() = default;
    virtual ExternalJobState state() const = 0;
    virtual void cancel() = 0;
};
class ExternalGpu : public std::enable_shared_from_this<ExternalGpu> {
public:
    virtual ~ExternalGpu() = default;
    virtual ExternalGpuCapabilities capabilities() const = 0;
    virtual std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) = 0;
    virtual void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const = 0;
};
std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& model_path, std::uint32_t device_index,
    vda_model_kind model_kind);
ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index);

}  // namespace vda_native
