#include "inferbridge_harness.h"

#include "video_depth_anything_native.h"
#if defined(VDA_WITH_VULKAN)
#include "external_gpu.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    vda_context* context = nullptr;
    std::string model_path;
#if defined(VDA_WITH_VULKAN)
    std::shared_ptr<vda_native::ExternalGpu> external_gpu;
#endif
    uint32_t input_size = 280u;
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
#if defined(VDA_WITH_VULKAN)
    std::shared_ptr<vda_native::ExternalJob> gpu_job;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
};

struct ibrh_output_lease {
    ibrh_job* job = nullptr;
#if defined(VDA_WITH_VULKAN)
    std::shared_ptr<vda_native::ExternalJob> gpu_job;
#endif
};

namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.video-depth-anything.native";
constexpr char kHarnessVersion[] = "1.1.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}

std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_uint(
    const std::string& json, const std::string& key, uint32_t& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    if (json[position] == '"') ++position;
    size_t end = position;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
    if (end == position) return false;
    uint64_t parsed = 0u;
    for (size_t index = position; index < end; ++index) {
        parsed = parsed * 10u + static_cast<uint32_t>(json[index] - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t i = 0; i < 8; ++i) {
        const int high = nibble(value[i * 2]);
        const int low = nibble(value[i * 2 + 1]);
        if (high < 0 || low < 0) return false;
        bytes[i] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
#if defined(VDA_WITH_VULKAN) && defined(_WIN32)
    for (int32_t index = 0; index < 32; ++index) {
        try {
            const auto capabilities = vda_native::probe_external_gpu(index);
            if (capabilities.available && capabilities.adapter_luid == luid) {
                device_index = index;
                return true;
            }
        } catch (...) {
            if (index == 0) return false;
            break;
        }
    }
#else
    (void)luid; (void)device_index;
#endif
    return false;
}

bool input_size(
    const std::string& json, uint32_t fallback, uint32_t& value) {
    value = fallback;
    uint32_t parsed = 0u;
    if (!json_uint(json, "Size", parsed)) return true;
    if (parsed == 0u || parsed > 4096u || parsed % 14u != 0u)
        return false;
    value = parsed;
    return true;
}

ibrh_result status_result(vda_status status) {
    switch (status) {
        case VDA_STATUS_OK: return IBRH_OK;
        case VDA_STATUS_INVALID_ARGUMENT:
        case VDA_STATUS_BUFFER_TOO_SMALL:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case VDA_STATUS_VULKAN_UNAVAILABLE:
        case VDA_STATUS_UNSUPPORTED:
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) delete job;
}

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 1u;
#if defined(VDA_WITH_VULKAN) && defined(_WIN32)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    capabilities->output_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_D3D12;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_D3D12_FENCE;
    capabilities->maximum_in_flight_jobs = 3u;
#endif
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    const std::string device = copy_string(request->requested_device_json);
    uint32_t index = 0u;
    if (json_uint(device, "index", index)) {
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "VDA requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        "VDA could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "VDA model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    std::string encoder;
    if (json_string(parameters, "Encoder", encoder) &&
        encoder != "vits") {
        return fail(
            runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "VDA native harness currently supports the catalog vits model");
    }
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    if (!input_size(parameters, model->input_size, model->input_size)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "VDA Size must be a multiple of 14 up to 4096");
    }
#if defined(VDA_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u) {
        try {
            model->external_gpu = vda_native::create_external_gpu(
                path, static_cast<uint32_t>(runtime->vulkan_device_index));
            const auto capabilities = model->external_gpu->capabilities();
            if (!capabilities.available ||
                capabilities.adapter_luid != runtime->adapter_luid)
                throw std::runtime_error(
                    "VDA loaded on a GPU other than the requested LUID");
        } catch (const std::exception& error) {
            delete model;
            return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        error.what());
        }
    } else
#endif
    {
        const vda_status status = vda_create_vulkan(
            path.c_str(), VDA_MODEL_VITS_RELATIVE_32_FRAMES,
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != VDA_STATUS_OK) {
            const std::string message =
                std::string("VDA model load failed: ") + vda_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(VDA_WITH_VULKAN)
    model->external_gpu.reset();
#endif
    vda_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL submit(
    ibrh_model* model, size_t request_size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if (model == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || request->inputs == nullptr)
        return fail(
            model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "VDA requires exactly one BGRA8 input");
    const ibrh_resource& input = request->inputs[0];
    if (input.struct_size < sizeof(input))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    uint32_t size = model->input_size;
    const std::string parameters = copy_string(request->parameters_json);
    if (!input_size(parameters, size, size))
        return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                    "VDA Size must be a multiple of 14 up to 4096");
    if (input.domain == IBRH_RESOURCE_DOMAIN_D3D12 &&
        input.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        input.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED) {
#if !defined(VDA_WITH_VULKAN) || !defined(_WIN32)
        return fail(model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                    "VDA D3D12 texture input is unavailable in this build");
#else
        if (input.pixel_format != IBRH_PIXEL_BGRA8 ||
            input.native_handle == 0u || input.width == 0u ||
            input.height == 0u)
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                        "VDA D3D12 texture descriptor is invalid");
        if (request->synchronization_count != 0u &&
            request->synchronizations == nullptr)
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                        "VDA synchronization array is missing");
        const ibrh_synchronization* wait = nullptr;
        for (uint32_t index = 0; index < request->synchronization_count; ++index) {
            const auto& candidate = request->synchronizations[index];
            if (candidate.struct_size < sizeof(candidate))
                return IBRH_ERROR_STRUCT_TOO_SMALL;
            if (candidate.kind == IBRH_SYNC_D3D12_FENCE &&
                candidate.operation == IBRH_SYNC_WAIT &&
                candidate.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED) {
                if (wait != nullptr)
                    return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                                "VDA received multiple D3D12 wait fences");
                wait = &candidate;
            }
        }
        if (wait == nullptr || wait->native_handle == 0u)
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                        "VDA D3D12 input requires a wait fence");
        auto* job = new (std::nothrow) ibrh_job();
        if (job == nullptr) return IBRH_ERROR_INTERNAL;
        try {
            std::lock_guard<std::mutex> lock(model->submit_mutex);
            if (!model->external_gpu) {
                vda_destroy(model->context);
                model->context = nullptr;
                model->external_gpu = vda_native::create_external_gpu(
                    model->model_path,
                    static_cast<uint32_t>(model->runtime->vulkan_device_index));
                const auto capabilities = model->external_gpu->capabilities();
                if (!capabilities.available ||
                    (model->runtime->adapter_luid != 0u &&
                     capabilities.adapter_luid != model->runtime->adapter_luid))
                    throw std::runtime_error(
                        "VDA GPU does not match the requested LUID");
            }
            std::string reset;
            const bool reset_stream =
                json_string(parameters, "Reset", reset) && reset == "YES";
            job->gpu_job = model->external_gpu->submit_texture(
                {input.native_handle, input.width, input.height, size,
                 wait->native_handle, wait->value,
                 request->source_frame_id, request->timestamp_ns,
                 reset_stream});
        } catch (const vda_native::GpuSlotsExhausted& error) {
            delete job;
            return fail(model->runtime, IBRH_ERROR_INVALID_STATE, error.what());
        } catch (const std::invalid_argument& error) {
            delete job;
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
        } catch (const std::exception& error) {
            delete job;
            return fail(model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        error.what());
        }
        job->source_frame_id = request->source_frame_id;
        job->timestamp_ns = request->timestamp_ns;
        job->width = input.width; job->height = input.height;
        *output = job;
        return IBRH_OK;
#endif
    }
    if (request->synchronization_count != 0u)
        return fail(
            model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "VDA host harness does not accept external synchronization");
    if (input.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        input.kind != IBRH_RESOURCE_KIND_IMAGE_2D ||
        input.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        input.pixel_format != IBRH_PIXEL_BGRA8 ||
        input.native_handle == 0u || input.width == 0u ||
        input.height == 0u || input.width > UINT32_MAX / 4u ||
        input.row_stride_bytes < input.width * 4u ||
        input.byte_offset > input.byte_size ||
        input.byte_size - input.byte_offset <
            static_cast<uint64_t>(input.row_stride_bytes) * input.height) {
        return fail(
            model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "VDA harness requires a valid host BGRA8 image");
    }
#if defined(VDA_WITH_VULKAN)
    if (model->context == nullptr)
        return fail(model->runtime, IBRH_ERROR_INVALID_STATE,
                    "VDA model is already active in GPU-resource mode");
#endif
    auto* job = new (std::nothrow) ibrh_job();
    if (job == nullptr) return IBRH_ERROR_INTERNAL;
    job->source_frame_id = request->source_frame_id;
    job->timestamp_ns = request->timestamp_ns;
    job->width = input.width;
    job->height = input.height;
    try {
        job->depth.resize(
            static_cast<size_t>(job->width) * job->height);
    } catch (...) {
        delete job;
        return IBRH_ERROR_INTERNAL;
    }
    const auto* bgra = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(input.native_handle)) + input.byte_offset;
    {
        std::lock_guard<std::mutex> lock(model->submit_mutex);
        std::string reset;
        if (json_string(
                copy_string(request->parameters_json), "Reset", reset) &&
            reset == "YES") {
            const vda_status reset_status =
                vda_stream_reset(model->context);
            if (reset_status != VDA_STATUS_OK) {
                delete job;
                return fail(
                    model->runtime, status_result(reset_status),
                    vda_last_error());
            }
        }
        const vda_status status = vda_infer_stream_bgra8_f32(
            model->context,
            bgra,
            input.row_stride_bytes,
            static_cast<int32_t>(input.width),
            static_cast<int32_t>(input.height),
            static_cast<int32_t>(size),
            job->depth.data(), job->depth.size());
        if (status != VDA_STATUS_OK) {
            const std::string message =
                std::string("VDA inference failed: ") + vda_last_error();
            delete job;
            return fail(model->runtime, status_result(status), message);
        }
    }
    *output = job;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
#if defined(VDA_WITH_VULKAN)
    if (job->gpu_job) {
        switch (job->gpu_job->state()) {
            case vda_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING; break;
            case vda_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE; break;
            case vda_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED; break;
        }
    } else
#endif
    status->state = IBRH_JOB_COMPLETE;
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(VDA_WITH_VULKAN)
    if (job->gpu_job) { job->gpu_job->cancel(); return IBRH_OK; }
#endif
    return IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL output_acquire(
    ibrh_job* job, uint32_t output_index, size_t descriptor_size,
    ibrh_output_descriptor* descriptor, ibrh_output_lease** output) {
    if (job == nullptr || descriptor == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (descriptor_size < sizeof(*descriptor))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (output_index != 0u) return IBRH_ERROR_NOT_FOUND;
    auto* lease = new (std::nothrow) ibrh_output_lease();
    if (lease == nullptr) return IBRH_ERROR_INTERNAL;
#if defined(VDA_WITH_VULKAN)
    if (job->gpu_job) {
        vda_native::ExternalTextureOutput native{};
        try { native = job->gpu_job->output(); }
        catch (...) { delete lease; return IBRH_ERROR_CANCELLED; }
        lease->gpu_job = job->gpu_job;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->api_version = IBRH_CURRENT_API_VERSION;
        descriptor->output_index = output_index;
        descriptor->payload_type = IBRH_PIXEL_DEPTH_FLOAT32;
        descriptor->source_frame_id = native.source_frame_id;
        descriptor->timestamp_ns = native.timestamp_ns;
        descriptor->resource.struct_size = sizeof(descriptor->resource);
        descriptor->resource.api_version = IBRH_CURRENT_API_VERSION;
        descriptor->resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
        descriptor->resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
        descriptor->resource.access = IBRH_RESOURCE_ACCESS_READ;
        descriptor->resource.pixel_format = IBRH_PIXEL_DEPTH_FLOAT32;
        descriptor->resource.width = native.width;
        descriptor->resource.height = native.height;
        descriptor->resource.depth = 1u;
        descriptor->resource.row_stride_bytes = native.width * sizeof(float);
        descriptor->resource.byte_size =
            static_cast<uint64_t>(native.width) * native.height * sizeof(float);
        descriptor->resource.native_handle_type =
            IBRH_NATIVE_HANDLE_WIN32_SHARED;
        descriptor->resource.native_handle = native.shared_texture_handle;
        descriptor->ready.struct_size = sizeof(descriptor->ready);
        descriptor->ready.api_version = IBRH_CURRENT_API_VERSION;
        descriptor->ready.kind = IBRH_SYNC_D3D12_FENCE;
        descriptor->ready.operation = IBRH_SYNC_WAIT;
        descriptor->ready.native_handle_type =
            IBRH_NATIVE_HANDLE_WIN32_SHARED;
        descriptor->ready.native_handle = native.ready_fence_handle;
        descriptor->ready.value = native.ready_fence_value;
        *output = lease;
        return IBRH_OK;
    }
#endif
    retain_job(job);
    lease->job = job;
    *descriptor = {};
    descriptor->struct_size = sizeof(*descriptor);
    descriptor->api_version = IBRH_CURRENT_API_VERSION;
    descriptor->output_index = 0u;
    descriptor->payload_type = IBRH_PIXEL_DEPTH_FLOAT32;
    descriptor->source_frame_id = job->source_frame_id;
    descriptor->timestamp_ns = job->timestamp_ns;
    descriptor->resource.struct_size = sizeof(descriptor->resource);
    descriptor->resource.api_version = IBRH_CURRENT_API_VERSION;
    descriptor->resource.domain = IBRH_RESOURCE_DOMAIN_HOST;
    descriptor->resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    descriptor->resource.access = IBRH_RESOURCE_ACCESS_READ;
    descriptor->resource.pixel_format = IBRH_PIXEL_DEPTH_FLOAT32;
    descriptor->resource.width = job->width;
    descriptor->resource.height = job->height;
    descriptor->resource.depth = 1u;
    descriptor->resource.row_stride_bytes = job->width * sizeof(float);
    descriptor->resource.byte_size = job->depth.size() * sizeof(float);
    descriptor->resource.native_handle_type =
        IBRH_NATIVE_HANDLE_HOST_POINTER;
    descriptor->resource.native_handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(job->depth.data()));
    *output = lease;
    return IBRH_OK;
}

void IBRH_CALL output_release(ibrh_output_lease* lease) {
    if (lease == nullptr) return;
#if defined(VDA_WITH_VULKAN)
    if (lease->gpu_job) {
        lease->gpu_job.reset();
        delete lease;
        return;
    }
#endif
    release_job(lease->job);
    delete lease;
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->output_acquire = output_acquire;
    api->output_release = output_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
