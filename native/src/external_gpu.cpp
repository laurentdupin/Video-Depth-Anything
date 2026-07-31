#include "external_gpu.h"

#include "video_depth_anything_native.h"
#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_io.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "temporal_gpu.h"
#include "vulkan.h"

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace vda_native {
namespace {

struct StreamEntry {
    std::array<TemporalFrameCache, 4> modules;
};

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kGpuSlotCount = 3u;
void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) throw std::runtime_error(
        std::string(operation) + " failed with HRESULT " +
        std::to_string(static_cast<long>(result)));
}
ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    if (luid == 0u) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
                  "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(D3D12CreateDevice(
            adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        return device;
    }
    return {};
}
struct SharedOutput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE resource_handle = nullptr, fence_handle = nullptr;
    SharedOutput() = default;
    SharedOutput(const SharedOutput&) = delete;
    SharedOutput& operator=(const SharedOutput&) = delete;
    SharedOutput(SharedOutput&& other) noexcept
        : resource(std::move(other.resource)), fence(std::move(other.fence)),
          resource_handle(std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)) {}
    SharedOutput& operator=(SharedOutput&& other) noexcept {
        if (this != &other) {
            if (resource_handle) CloseHandle(resource_handle);
            if (fence_handle) CloseHandle(fence_handle);
            resource = std::move(other.resource); fence = std::move(other.fence);
            resource_handle = std::exchange(other.resource_handle, nullptr);
            fence_handle = std::exchange(other.fence_handle, nullptr);
        }
        return *this;
    }
    ~SharedOutput() {
        if (resource_handle) CloseHandle(resource_handle);
        if (fence_handle) CloseHandle(fence_handle);
    }
};
struct GpuSlot {
    std::atomic<bool> occupied{false};
    SharedOutput shared;
    std::uint32_t width = 0, height = 0;
    std::uint64_t fence_value = 0;
};
SharedOutput create_shared_output(
    ID3D12Device* device, std::uint32_t width, std::uint32_t height) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    SharedOutput output;
    check_hresult(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &description,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&output.resource)), "CreateCommittedResource(VDA output)");
    check_hresult(device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&output.fence)),
        "CreateFence(VDA output)");
    check_hresult(device->CreateSharedHandle(
        output.resource.Get(), nullptr, GENERIC_ALL, nullptr,
        &output.resource_handle), "CreateSharedHandle(VDA output)");
    check_hresult(device->CreateSharedHandle(
        output.fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &output.fence_handle), "CreateSharedHandle(VDA fence)");
    return output;
}
void validate_input(
    ID3D12Device* device, std::uintptr_t handle,
    std::uint32_t width, std::uint32_t height) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&resource)),
        "OpenSharedHandle(VDA input)");
    const auto d = resource->GetDesc();
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        d.Width != width || d.Height != height || d.DepthOrArraySize != 1u ||
        d.MipLevels != 1u || d.SampleDesc.Count != 1u ||
        d.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
        throw std::invalid_argument("shared VDA input is not declared BGRA8");
}
std::shared_ptr<GpuSlot> acquire_slot(
    const std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount>& slots,
    std::atomic<std::uint32_t>& next) {
    const std::uint32_t first = next.fetch_add(1u) % kGpuSlotCount;
    for (std::uint32_t offset = 0; offset < kGpuSlotCount; ++offset) {
        auto slot = slots[(first + offset) % kGpuSlotCount];
        bool expected = false;
        if (slot->occupied.compare_exchange_strong(expected, true)) return slot;
    }
    throw GpuSlotsExhausted();
}
VulkanImage prepare_output(
    GpuSlot& slot, ID3D12Device* device, VulkanContext& context,
    std::uint32_t width, std::uint32_t height) {
    if (!slot.shared.resource || slot.width != width || slot.height != height) {
        slot.shared = SharedOutput{};
        slot.width = slot.height = 0; slot.fence_value = 0;
        slot.shared = create_shared_output(device, width, height);
        slot.width = width; slot.height = height;
    }
    return context.import_d3d12_image(
        slot.shared.resource_handle, width, height, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}
#endif

class ExternalGpuImpl;
#if defined(_WIN32)
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(
        std::shared_ptr<ExternalGpu> owner, std::shared_ptr<GpuSlot> slot,
        VulkanImage input, VulkanImage output, VulkanSubmission submission,
        std::vector<std::shared_ptr<StreamEntry>> retained,
        const ExternalTextureRequest& request, std::uint64_t fence_value)
        : owner_(std::move(owner)), slot_(std::move(slot)),
          input_(std::move(input)), output_(std::move(output)),
          submission_(std::move(submission)), retained_(std::move(retained)),
          request_(request), fence_value_(fence_value) {}
    ~ExternalJobImpl() override {
        try { submission_.wait(); } catch (...) {}
        submission_ = VulkanSubmission{}; output_ = VulkanImage{};
        input_ = VulkanImage{}; retained_.clear();
        slot_->occupied.store(false);
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        return submission_.ready() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true); }
    ExternalTextureOutput output() const override {
        if (cancelled_.load()) throw std::runtime_error("VDA job cancelled");
        return {reinterpret_cast<std::uintptr_t>(slot_->shared.resource_handle),
            request_.width, request_.height,
            reinterpret_cast<std::uintptr_t>(slot_->shared.fence_handle),
            fence_value_, request_.source_frame_id, request_.timestamp_ns};
    }
private:
    std::shared_ptr<ExternalGpu> owner_;
    std::shared_ptr<GpuSlot> slot_;
    VulkanImage input_, output_;
    VulkanSubmission submission_;
    std::vector<std::shared_ptr<StreamEntry>> retained_;
    ExternalTextureRequest request_{};
    std::uint64_t fence_value_ = 0;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(const std::string& path, std::uint32_t index)
        : model_(path, VDA_MODEL_VITS_RELATIVE_32_FRAMES), context_(index),
          gpu_model_(model_, context_), operators_(context_),
          encoder_(context_, gpu_model_, operators_),
          dpt_(context_, gpu_model_, operators_), io_(context_)
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid())),
          slots_{std::make_shared<GpuSlot>(), std::make_shared<GpuSlot>(),
                 std::make_shared<GpuSlot>()}
#endif
          {}
    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& c = context_.external_capabilities();
        const bool available = d3d12_ && c.timeline_semaphore &&
            c.d3d12_resource_import && c.d3d12_fence_import &&
            c.d3d12_bgra8_sampled_image_import &&
            c.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0,
                available ? kGpuSlotCount : 0};
#else
        return {};
#endif
    }
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request; throw std::runtime_error("VDA D3D12 unavailable");
#else
        if (!capabilities().available) throw std::runtime_error(
            "complete VDA D3D12/Vulkan interop is unavailable");
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.width || !request.height || !request.process_resolution ||
            request.process_resolution % 14u != 0u)
            throw std::invalid_argument("invalid VDA GPU texture request");
        validate_input(d3d12_.Get(), request.shared_texture_handle,
                       request.width, request.height);
        auto slot = acquire_slot(slots_, next_slot_);
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            if (request.reset) reset_stream();
            if (!cache_.empty() && (width_ != request.width ||
                height_ != request.height || size_ != request.process_resolution))
                throw std::invalid_argument(
                    "stream dimensions changed without Reset YES");
            VulkanImage output = prepare_output(
                *slot, d3d12_.Get(), context_, request.width, request.height);
            const std::uint64_t signal_value = ++slot->fence_value;
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height, VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
            std::vector<std::shared_ptr<StreamEntry>> retained;
            VulkanSubmission submission = context_.segmented_batch_async(
                std::move(wait), std::move(signal), [&] {
                    const std::uint32_t size = request.process_resolution;
                    VulkanBuffer image = context_.create_device_buffer(
                        static_cast<std::uint64_t>(size) * size * 3u * sizeof(float));
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    io_.preprocess(image, input, size);
                    auto run = [&](const std::vector<std::shared_ptr<StreamEntry>>& selected,
                                   std::shared_ptr<StreamEntry>& current) {
                        std::vector<const TemporalFrameCache*> history[4];
                        for (const auto& entry : selected)
                            for (std::uint32_t m=0;m<4;++m)
                                history[m].push_back(&entry->modules[m]);
                        current = std::make_shared<StreamEntry>();
                        TemporalFrameCache* outputs[4] = {
                            &current->modules[0], &current->modules[1],
                            &current->modules[2], &current->modules[3]};
                        return dpt_.forward_stream(
                            encoder_.forward(image, 1, size, size), history, outputs);
                    };
                    if (cache_.empty()) {
                        std::shared_ptr<StreamEntry> seed;
                        std::vector<std::shared_ptr<StreamEntry>> empty;
                        (void)run(empty, seed);
                        cache_.assign(32, seed);
                        width_=request.width; height_=request.height; size_=size;
                    }
                    std::vector<std::shared_ptr<StreamEntry>> selected;
                    selected.reserve(31); selected.push_back(cache_[0]);
                    selected.push_back(cache_[1]);
                    const std::size_t tail=cache_.size()-29;
                    selected.insert(selected.end(),cache_.begin()+tail,cache_.end());
                    std::shared_ptr<StreamEntry> current;
                    FeatureMap depth=run(selected,current);
                    retained=selected;
                    retained.push_back(current);
                    cache_.push_back(current); ++stream_id_;
                    if(stream_id_+32>42) cache_.erase(cache_.begin()+1);
                    io_.resize_normalize(
                        output, depth.buffer, depth.width, depth.height);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), slot, std::move(input), std::move(output),
                std::move(submission), std::move(retained), request, signal_value);
        } catch (...) { slot->occupied.store(false); throw; }
#endif
    }
    void transfer_counters(std::uint64_t& up, std::uint64_t& down) const override {
        context_.transfer_counters(up, down);
    }
private:
    void reset_stream() {
        cache_.clear(); stream_id_=-1; width_=height_=size_=0;
    }
    ModelFile model_; VulkanContext context_; GpuModel gpu_model_;
    VulkanOperators operators_; VdaGpuEncoder encoder_; VdaGpuDpt dpt_; GpuIo io_;
    std::vector<std::shared_ptr<StreamEntry>> cache_;
    std::int64_t stream_id_=-1;
    std::uint32_t width_=0,height_=0,size_=0;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::array<std::shared_ptr<GpuSlot>,kGpuSlotCount> slots_;
    std::atomic<std::uint32_t> next_slot_{0};
    std::mutex record_mutex_;
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path, std::uint32_t index) {
    return std::make_shared<ExternalGpuImpl>(path,index);
}
ExternalGpuCapabilities probe_external_gpu(std::uint32_t index) {
#if defined(_WIN32)
    VulkanContext context(index);
    auto device=matching_d3d12_device(context.adapter_luid());
    const auto& c=context.external_capabilities();
    const bool available=device && c.timeline_semaphore &&
        c.d3d12_resource_import && c.d3d12_fence_import &&
        c.d3d12_bgra8_sampled_image_import && c.d3d12_r32_storage_image_import;
    return {available,available?context.adapter_luid():0,
            available?kGpuSlotCount:0};
#else
    (void)index; return {};
#endif
}

}  // namespace vda_native
