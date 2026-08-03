
#include "external_gpu.h"

#include "video_depth_anything_native.h"
#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_io.h"
#include "gpu_model.h"
#include "model.h"
#include "model_config.h"
#include "operators.h"
#include "temporal_gpu.h"
#include "vulkan.h"
#include "inferbridge/native_harness_resource_lifetime.h"

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
constexpr std::uint32_t kMaxInFlightJobs = 3u;
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
void validate_texture(ID3D12Device* device, std::uintptr_t handle,
    std::uint32_t width, std::uint32_t height, DXGI_FORMAT format,
    const char* operation) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&resource)), operation);
    const auto d=resource->GetDesc();
    if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.Width!=width||
       d.Height!=height||d.DepthOrArraySize!=1u||d.MipLevels!=1u||
       d.SampleDesc.Count!=1u||d.Format!=format)
        throw std::invalid_argument("VDA shared texture descriptor mismatch");
}
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(std::shared_ptr<ExternalGpu> owner,VulkanImage input,
        VulkanImage output,VulkanSubmission submission,
        std::vector<std::shared_ptr<StreamEntry>> retained,
        inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime)
        :owner_(std::move(owner)),input_(std::move(input)),output_(std::move(output)),
         submission_(std::move(submission)),retained_(std::move(retained)),
          lifetime_(std::move(lifetime)){}
    ~ExternalJobImpl()override{
        inferbridge::native_harness::wait_then_retire(
            lifetime_, submission_, [this] {
                output_={};input_={};retained_.clear();
            });}
    ExternalJobState state()const override{if(cancelled_.load())return ExternalJobState::cancelled;
        return submission_.ready()?ExternalJobState::complete:ExternalJobState::running;}
    void cancel()override{cancelled_.store(true);}
private:
    std::shared_ptr<ExternalGpu> owner_;VulkanImage input_,output_;
    VulkanSubmission submission_;std::vector<std::shared_ptr<StreamEntry>> retained_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(
        const std::string& path, std::uint32_t index, vda_model_kind kind)
        : config_(model_config(kind)), model_(path, kind), context_(index),
          gpu_model_(model_, context_), operators_(context_),
          encoder_(context_, gpu_model_, operators_, config_),
          dpt_(context_, gpu_model_, operators_, config_), io_(context_)
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid()))
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
                available ? kMaxInFlightJobs : 0};
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
            !request.output_texture_handle || !request.signal_fence_handle ||
            !request.width || !request.height || !request.process_resolution ||
            request.process_resolution % 14u != 0u ||
            request.output_width != request.width ||
            request.output_height != request.height)
            throw std::invalid_argument("invalid VDA GPU texture request");
        validate_texture(d3d12_.Get(),request.shared_texture_handle,
            request.width,request.height,DXGI_FORMAT_B8G8R8A8_UNORM,
            "OpenSharedHandle(VDA input)");
        validate_texture(d3d12_.Get(),request.output_texture_handle,
            request.output_width,request.output_height,DXGI_FORMAT_R32_FLOAT,
            "OpenSharedHandle(VDA output)");
        try {
            auto lifetime_guard = lifetime_->acquire();
            if (request.reset) reset_stream();
            if (!cache_.empty() && (width_ != request.width ||
                height_ != request.height || size_ != request.process_resolution))
                throw std::invalid_argument(
                    "stream dimensions changed without Reset YES");
            VulkanImage output=context_.import_d3d12_image(
                reinterpret_cast<void*>(request.output_texture_handle),
                request.output_width,request.output_height,VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height, VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal=context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.signal_fence_handle),
                request.signal_fence_value);
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
                    io_.resize_depth(
                        output, depth.buffer, depth.width, depth.height,
                        !config_.metric);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(),std::move(input),std::move(output),
                std::move(submission),std::move(retained),lifetime_);
        } catch (...) { throw; }
#endif
    }
    void transfer_counters(std::uint64_t& up, std::uint64_t& down) const override {
        context_.transfer_counters(up, down);
    }
private:
    void reset_stream() {
        cache_.clear(); stream_id_=-1; width_=height_=size_=0;
    }
    const ModelConfig& config_;
    ModelFile model_; VulkanContext context_; GpuModel gpu_model_;
    VulkanOperators operators_; VdaGpuEncoder encoder_; VdaGpuDpt dpt_; GpuIo io_;
    std::vector<std::shared_ptr<StreamEntry>> cache_;
    std::int64_t stream_id_=-1;
    std::uint32_t width_=0,height_=0,size_=0;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_ =
        inferbridge::native_harness::make_resource_lifetime_domain();
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path, std::uint32_t index, vda_model_kind kind) {
    return std::make_shared<ExternalGpuImpl>(path,index,kind);
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
            available?kMaxInFlightJobs:0};
#else
    (void)index; return {};
#endif
}
}  // namespace vda_native
