#include "video_depth_anything_native.h"
#include "inferbridge_harness.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT value, const char* operation) {
    if (FAILED(value)) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<long>(value)));
}
void check(ibrh_result value, const char* operation) {
    if (value != IBRH_OK) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<unsigned>(value)));
}
void check(vda_status value, const char* operation) {
    if (value != VDA_STATUS_OK) throw std::runtime_error(
        std::string(operation) + " failed: " + vda_last_error());
}

struct Capture {
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE texture_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t value = 1u;
};

std::vector<std::uint8_t> pixels(
    std::uint32_t width, std::uint32_t height, std::uint32_t frame) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * 4u;
            result[offset] = static_cast<std::uint8_t>((x * 11u + y + frame) & 255u);
            result[offset + 1u] = static_cast<std::uint8_t>((x + y * 7u + frame * 3u) & 255u);
            result[offset + 2u] = static_cast<std::uint8_t>((x * 3u + y * 5u + frame * 13u) & 255u);
            result[offset + 3u] = 255u;
        }
    }
    return result;
}

Capture upload_texture(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& source,
    std::uint32_t width, std::uint32_t height, bool signal = true) {
    Capture result;
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC texture_desc{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_B8G8R8A8_UNORM, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    check(device->CreateCommittedResource(
        &default_heap, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&result.texture)), "CreateCommittedResource(input)");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0u;
    UINT64 row_bytes = 0u;
    UINT64 upload_bytes = 0u;
    device->GetCopyableFootprints(
        &texture_desc, 0, 1, 0, &footprint, &rows,
        &row_bytes, &upload_bytes);
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, upload_bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    check(device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&result.upload)), "CreateCommittedResource(upload)");
    std::uint8_t* mapped = nullptr;
    const D3D12_RANGE no_read{0, 0};
    check(result.upload->Map(
        0, &no_read, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    for (std::uint32_t y = 0; y < height; ++y)
        std::memcpy(
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            source.data() + static_cast<std::size_t>(y) * width * 4u,
            static_cast<std::size_t>(width) * 4u);
    result.upload->Unmap(0, nullptr);

    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, result.allocator.Get(), nullptr,
        IID_PPV_ARGS(&result.commands)), "CreateCommandList");
    const D3D12_TEXTURE_COPY_LOCATION destination{
        result.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION upload_location{};
    upload_location.pResource = result.upload.Get();
    upload_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    upload_location.PlacedFootprint = footprint;
    result.commands->CopyTextureRegion(
        &destination, 0, 0, 0, &upload_location, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = result.texture.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(upload list)");
    ID3D12CommandList* lists[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    check(device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result.fence)),
        "CreateFence(input)");
    if (signal) check(queue->Signal(result.fence.Get(), result.value), "Signal(input)");
    check(device->CreateSharedHandle(
        result.texture.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.texture_handle), "CreateSharedHandle(input)");
    check(device->CreateSharedHandle(
        result.fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.fence_handle), "CreateSharedHandle(input fence)");
    return result;
}

void close_capture(Capture& value) {
    if (value.texture_handle) CloseHandle(value.texture_handle);
    if (value.fence_handle) CloseHandle(value.fence_handle);
    value.texture_handle = nullptr;
    value.fence_handle = nullptr;
}

void wait_fence(ID3D12Device* device, const ibrh_synchronization& ready) {
    ComPtr<ID3D12Fence> fence;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(ready.native_handle),
        IID_PPV_ARGS(&fence)), "OpenSharedHandle(output fence)");
    if (fence->GetCompletedValue() >= ready.value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
    check(fence->SetEventOnCompletion(ready.value, event), "SetEventOnCompletion");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
}

std::vector<float> read_output(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const ibrh_output_descriptor& output) {
    wait_fence(device, output.ready);
    ComPtr<ID3D12Resource> texture;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(output.resource.native_handle),
        IID_PPV_ARGS(&texture)), "OpenSharedHandle(output texture)");
    const D3D12_RESOURCE_DESC description = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    UINT64 bytes = 0;
    device->GetCopyableFootprints(
        &description, 0, 1, 0, &footprint, &rows, &row_bytes, &bytes);
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> readback;
    check(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback)), "CreateCommittedResource(readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(readback)");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&list)), "CreateCommandList(readback)");
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = texture.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    check(list->Close(), "Close(readback list)");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> done;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)),
          "CreateFence(readback)");
    check(queue->Signal(done.Get(), 1), "Signal(readback)");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(done->SetEventOnCompletion(1, event), "SetEventOnCompletion(readback)");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    const std::uint8_t* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    check(readback->Map(0, &range, reinterpret_cast<void**>(
        const_cast<std::uint8_t**>(&mapped))), "Map(readback)");
    std::vector<float> result(
        static_cast<std::size_t>(output.resource.width) * output.resource.height);
    for (std::uint32_t y = 0; y < output.resource.height; ++y)
        std::memcpy(
            result.data() + static_cast<std::size_t>(y) * output.resource.width,
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            static_cast<std::size_t>(output.resource.width) * sizeof(float));
    readback->Unmap(0, nullptr);
    return result;
}

struct SelectedDevice {
    ComPtr<ID3D12Device> device;
    std::string luid_json;
    std::string name;
};

SelectedDevice select_device(const ibrh_api& api) {
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            &description.AdapterLuid);
        char json[40]{};
        std::snprintf(json, sizeof(json),
            "{\"luid\":\"%02x%02x%02x%02x%02x%02x%02x%02x\"}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]);
        ibrh_runtime_create_request request{};
        request.struct_size = sizeof(request);
        request.api_version = IBRH_CURRENT_API_VERSION;
        request.backend = {"native", 6u};
        request.requested_device_json = {json, std::strlen(json)};
        ibrh_runtime* runtime = nullptr;
        if (api.runtime_create(sizeof(request), &request, &runtime) != IBRH_OK)
            continue;
        api.runtime_destroy(runtime);
        SelectedDevice result;
        check(D3D12CreateDevice(
            adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&result.device)), "D3D12CreateDevice");
        result.luid_json = json;
        char narrow[128]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
                            narrow, sizeof(narrow), nullptr, nullptr);
        result.name = narrow;
        return result;
    }
    throw std::runtime_error("no D3D12 adapter accepted by VDA Vulkan");
}

struct Submitted {
    ibrh_job* job = nullptr;
    ibrh_output_lease* lease = nullptr;
    ibrh_output_descriptor output{};
};

Submitted submit(
    const ibrh_api& api, ibrh_model* model,
    Capture& capture, std::uint32_t width, std::uint32_t height,
    std::uint64_t frame) {
    ibrh_resource resource{};
    resource.struct_size = sizeof(resource);
    resource.api_version = IBRH_CURRENT_API_VERSION;
    resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    resource.access = IBRH_RESOURCE_ACCESS_READ;
    resource.pixel_format = IBRH_PIXEL_BGRA8;
    resource.width = width;
    resource.height = height;
    resource.depth = 1u;
    resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    resource.native_handle = reinterpret_cast<std::uintptr_t>(capture.texture_handle);
    ibrh_synchronization wait{};
    wait.struct_size = sizeof(wait);
    wait.api_version = IBRH_CURRENT_API_VERSION;
    wait.kind = IBRH_SYNC_D3D12_FENCE;
    wait.operation = IBRH_SYNC_WAIT;
    wait.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    wait.native_handle = reinterpret_cast<std::uintptr_t>(capture.fence_handle);
    wait.value = capture.value;
    const std::string parameters = "{\"Size\":\"140\"}";
    ibrh_submit_request request{};
    request.struct_size = sizeof(request);
    request.api_version = IBRH_CURRENT_API_VERSION;
    request.inputs = &resource;
    request.input_count = 1u;
    request.synchronizations = &wait;
    request.synchronization_count = 1u;
    request.source_frame_id = frame;
    request.timestamp_ns = 900000u + frame;
    request.parameters_json = {parameters.data(), parameters.size()};
    Submitted result;
    check(api.submit(model, sizeof(request), &request, &result.job), "submit");
    close_capture(capture);
    check(api.output_acquire(
        result.job, 0, sizeof(result.output), &result.output, &result.lease),
        "output_acquire");
    if (result.output.source_frame_id != frame ||
        result.output.timestamp_ns != request.timestamp_ns ||
        result.output.resource.domain != IBRH_RESOURCE_DOMAIN_D3D12 ||
        result.output.resource.pixel_format != IBRH_PIXEL_DEPTH_FLOAT32 ||
        result.output.resource.width != width ||
        result.output.resource.height != height ||
        result.output.ready.kind != IBRH_SYNC_D3D12_FENCE)
        throw std::runtime_error("VDA output descriptor correlation failed");
    return result;
}

void normalize(std::vector<float>& values) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second) ||
        !(*bounds.second > *bounds.first))
        throw std::runtime_error("VDA output is not finite and varying");
    const float low = *bounds.first;
    const float span = *bounds.second - low;
    for (float& value : values) value = (value - low) / span;
}

std::filesystem::path model_path() {
    if (const char* value = std::getenv("VDA_MODEL")) return value;
    return {};
}
}  // namespace

int main() try {
    const auto model_file = model_path();
    if (model_file.empty() || !std::filesystem::exists(model_file)) return 77;
    ibrh_api api{};
    check(ibrh_get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api), "ibrh_get_api");
    ibrh_capabilities capabilities{};
    check(api.query_capabilities(sizeof(capabilities), &capabilities), "capabilities");
    const std::uint64_t required = IBRH_CAP_GPU_RESOURCES |
        IBRH_CAP_EXTERNAL_SYNCHRONIZATION | IBRH_CAP_GPU_RESIDENT_OUTPUT;
    if ((capabilities.flags & required) != required ||
        capabilities.maximum_in_flight_jobs != 3u)
        throw std::runtime_error("VDA GPU capability contract is incomplete");
    SelectedDevice selected = select_device(api);
    std::cout << "device=" << selected.name << '\n';
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(selected.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)),
          "CreateCommandQueue");
    ibrh_runtime_create_request runtime_request{};
    runtime_request.struct_size = sizeof(runtime_request);
    runtime_request.api_version = IBRH_CURRENT_API_VERSION;
    runtime_request.backend = {"native", 6u};
    runtime_request.requested_device_json = {
        selected.luid_json.data(), selected.luid_json.size()};
    ibrh_runtime* runtime = nullptr;
    check(api.runtime_create(sizeof(runtime_request), &runtime_request, &runtime),
          "runtime_create");
    const std::string model_text = model_file.string();
    const std::string parameters = "{\"Size\":\"140\"}";
    ibrh_model_load_request load{};
    load.struct_size = sizeof(load);
    load.api_version = IBRH_CURRENT_API_VERSION;
    load.model_path = {model_text.data(), model_text.size()};
    load.parameters_json = {parameters.data(), parameters.size()};
    ibrh_model* model = nullptr;
    check(api.model_load(runtime, sizeof(load), &load, &model), "model_load");

    vda_transfer_counters before{sizeof(before), VDA_ABI_VERSION, 0u, 0u};
    check(vda_get_transfer_counters(&before), "transfer counters before");
    constexpr std::uint32_t width = 37u;
    constexpr std::uint32_t height = 23u;
    std::array<Submitted, 3> retained{};
    for (std::uint32_t index = 0; index < retained.size(); ++index) {
        auto source = pixels(width, height, index);
        Capture capture = upload_texture(
            selected.device.Get(), queue.Get(), source, width, height);
        retained[index] = submit(api, model, capture, width, height, 1000u + index);
        wait_fence(selected.device.Get(), retained[index].output.ready);
        ibrh_job_status status{};
        for (std::uint32_t attempt = 0u; attempt < 1000u; ++attempt) {
            check(api.job_poll(
                retained[index].job, sizeof(status), &status), "job_poll");
            if (status.state != IBRH_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != IBRH_JOB_COMPLETE ||
            status.source_frame_id != 1000u + index)
            throw std::runtime_error("VDA job completion correlation failed");
        api.job_release(retained[index].job);
        retained[index].job = nullptr;
    }
    if (retained[0].output.resource.native_handle ==
            retained[1].output.resource.native_handle ||
        retained[0].output.resource.native_handle ==
            retained[2].output.resource.native_handle)
        throw std::runtime_error("retained VDA leases alias output slots");

    const auto reused_pixels = pixels(width, height, 4);
    Capture dropped = upload_texture(
        selected.device.Get(), queue.Get(), reused_pixels, width, height);
    ibrh_resource dropped_resource{};
    dropped_resource.struct_size = sizeof(dropped_resource);
    dropped_resource.api_version = IBRH_CURRENT_API_VERSION;
    dropped_resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    dropped_resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    dropped_resource.pixel_format = IBRH_PIXEL_BGRA8;
    dropped_resource.width = width;
    dropped_resource.height = height;
    dropped_resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    dropped_resource.native_handle = reinterpret_cast<std::uintptr_t>(dropped.texture_handle);
    ibrh_synchronization dropped_wait{};
    dropped_wait.struct_size = sizeof(dropped_wait);
    dropped_wait.api_version = IBRH_CURRENT_API_VERSION;
    dropped_wait.kind = IBRH_SYNC_D3D12_FENCE;
    dropped_wait.operation = IBRH_SYNC_WAIT;
    dropped_wait.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    dropped_wait.native_handle = reinterpret_cast<std::uintptr_t>(dropped.fence_handle);
    dropped_wait.value = dropped.value;
    ibrh_submit_request dropped_request{};
    dropped_request.struct_size = sizeof(dropped_request);
    dropped_request.api_version = IBRH_CURRENT_API_VERSION;
    dropped_request.inputs = &dropped_resource;
    dropped_request.input_count = 1u;
    dropped_request.synchronizations = &dropped_wait;
    dropped_request.synchronization_count = 1u;
    dropped_request.source_frame_id = 1003u;
    dropped_request.parameters_json = {parameters.data(), parameters.size()};
    ibrh_job* dropped_job = nullptr;
    if (api.submit(model, sizeof(dropped_request), &dropped_request, &dropped_job) !=
            IBRH_ERROR_INVALID_STATE || dropped_job != nullptr)
        throw std::runtime_error("VDA fourth live lease was not rejected");
    const auto reusable_handle = retained[0].output.resource.native_handle;
    api.output_release(retained[0].lease);
    retained[0].lease = nullptr;
    Submitted reused;
    const std::string reset_parameters =
        "{\"Size\":\"140\",\"Reset\":\"YES\"}";
    dropped_request.parameters_json = {
        reset_parameters.data(), reset_parameters.size()};
    check(api.submit(model, sizeof(dropped_request), &dropped_request, &reused.job),
          "submit(reuse)");
    close_capture(dropped);
    check(api.output_acquire(
        reused.job, 0, sizeof(reused.output), &reused.output, &reused.lease),
        "output_acquire(reuse)");
    if (reused.output.resource.native_handle != reusable_handle)
        throw std::runtime_error("VDA output slot handle was not reused");
    wait_fence(selected.device.Get(), reused.output.ready);
    std::vector<float> gpu = read_output(selected.device.Get(), queue.Get(), reused.output);
    normalize(gpu);
    vda_transfer_counters gpu_after{
        sizeof(gpu_after), VDA_ABI_VERSION, 0u, 0u};
    check(vda_get_transfer_counters(&gpu_after), "GPU transfer counters after");
    if (gpu_after.tensor_upload_bytes != before.tensor_upload_bytes ||
        gpu_after.tensor_download_bytes != before.tensor_download_bytes)
        throw std::runtime_error("VDA GPU path performed host tensor staging");
    vda_context* cpu = nullptr;
    check(vda_create_vulkan(
        model_text.c_str(), VDA_MODEL_VITS_RELATIVE_32_FRAMES, 0u, &cpu),
        "vda_create_vulkan(reference)");
    std::vector<float> processed(
        static_cast<std::size_t>(width) * height);
    check(vda_infer_stream_bgra8_f32(
        cpu, reused_pixels.data(), width * 4u, width, height, 140,
        processed.data(), processed.size()),
        "vda_infer_stream_bgra8_f32(reference)");
    vda_destroy(cpu);
    std::vector<float> reference = std::move(processed);
    normalize(reference);
    float maximum_difference = 0.0f;
    for (std::size_t index = 0; index < gpu.size(); ++index)
        maximum_difference = std::max(
            maximum_difference, std::abs(gpu[index] - reference[index]));
    std::cout << "CPU correlation max/range=" << maximum_difference << '\n';
    if (maximum_difference >= 0.01f)
        throw std::runtime_error("VDA GPU output exceeds the 1% CPU gate");

    Capture blocked = upload_texture(
        selected.device.Get(), queue.Get(), pixels(width, height, 9), width, height, false);
    api.output_release(retained[1].lease);
    retained[1].lease = nullptr;
    Submitted cancelled = submit(api, model, blocked, width, height, 2000u);
    check(api.job_cancel(cancelled.job), "job_cancel");
    ibrh_job_status cancelled_status{};
    check(api.job_poll(cancelled.job, sizeof(cancelled_status), &cancelled_status),
          "job_poll(cancelled)");
    if (cancelled_status.state != IBRH_JOB_CANCELLED)
        throw std::runtime_error("VDA cancellation state failed");
    check(queue->Signal(blocked.fence.Get(), blocked.value),
          "Signal(cancelled input)");
    api.output_release(cancelled.lease);
    api.job_release(cancelled.job);

    api.output_release(retained[2].lease);
    api.output_release(reused.lease);
    api.job_release(reused.job);
    // One final lease must survive complete public object shutdown.
    Capture final_capture = upload_texture(
        selected.device.Get(), queue.Get(), pixels(width, height, 12), width, height);
    Submitted final_job = submit(api, model, final_capture, width, height, 3000u);
    api.job_release(final_job.job);
    api.model_unload(model);
    api.runtime_destroy(runtime);
    std::vector<float> final_depth = read_output(
        selected.device.Get(), queue.Get(), final_job.output);
    normalize(final_depth);
    api.output_release(final_job.lease);
    std::cout << "VDA common D3D12/Vulkan full graph passed; zero transfers; "
                 "three leases; reuse; cancellation; shutdown lease\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
