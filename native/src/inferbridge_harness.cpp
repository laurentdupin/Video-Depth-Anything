
#include "inferbridge_harness.h"

#include "video_depth_anything_native.h"
#include "inferbridge/native_harness_precision.h"
#include "model.h"
#include "model_config.h"
#if defined(VDA_WITH_VULKAN) || defined(VDA_WITH_METAL)
#include "external_gpu.h"
#endif
#if defined(VDA_WITH_METAL)
#include "vda_internal.h"
#endif

#if (defined(VDA_WITH_VULKAN) && defined(_WIN32)) || \
    (defined(VDA_WITH_METAL) && defined(__APPLE__))
#define VDA_WITH_EXTERNAL_GPU 1
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
    bool force_host_transfers = false;
};

struct ibrh_job;
struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    vda_context* context = nullptr;
    std::string model_path;
    vda_model_kind model_kind = VDA_MODEL_VITS_RELATIVE_32_FRAMES;
    bool metric = false;
#if defined(VDA_WITH_EXTERNAL_GPU)
    std::shared_ptr<vda_native::ExternalGpu> external_gpu;
#endif
    uint32_t input_size = 280u;
    std::mutex submit_mutex;
    std::shared_ptr<std::atomic<uint32_t>> occupied_slots =
        std::make_shared<std::atomic<uint32_t>>(0u);
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::deque<ibrh_job*> queue;
    bool stopping = false;
    std::thread worker;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
    std::atomic<uint32_t> state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel_requested{false};
    std::shared_ptr<std::atomic<uint32_t>> occupied_slots;
#if defined(VDA_WITH_EXTERNAL_GPU)
    std::shared_ptr<vda_native::ExternalJob> gpu_job;
    vda_native::ExternalTextureRequest request{};
    std::mutex gpu_mutex;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
};


namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.video-depth-anything.native";
constexpr char kHarnessVersion[] = "1.2.0";

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
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) {
        if (job->occupied_slots) job->occupied_slots->fetch_sub(1u);
        delete job;
    }
}

#if defined(VDA_WITH_EXTERNAL_GPU)
void worker_loop(ibrh_model* model) {
    for (;;) {
        ibrh_job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(model->queue_mutex);
            model->queue_condition.wait(lock, [&] {
                return model->stopping || !model->queue.empty();
            });
            if (model->stopping && model->queue.empty()) return;
            job = model->queue.front();
            model->queue.pop_front();
        }
        if (job->cancel_requested.load()) {
            job->state.store(IBRH_JOB_CANCELLED);
            release_job(job);
            continue;
        }
        try {
            auto gpu = model->external_gpu->submit_texture(job->request);
            {
                std::lock_guard<std::mutex> lock(job->gpu_mutex);
                job->gpu_job = std::move(gpu);
            }
            job->state.store(job->cancel_requested.load() ?
                IBRH_JOB_CANCELLED : IBRH_JOB_RUNNING);
            if (job->cancel_requested.load()) {
                std::lock_guard<std::mutex> lock(job->gpu_mutex);
                job->gpu_job->cancel();
            }
        } catch (...) {
            job->state.store(IBRH_JOB_FAILED);
        }
        release_job(job);
    }
}
#endif

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
#if defined(VDA_WITH_METAL) && defined(__APPLE__)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->output_domain_mask |= 1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_METAL_SHARED_EVENT;
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
    std::string transfer_mode;
    runtime->force_host_transfers =
        json_string(device, "transfer_mode", transfer_mode) &&
        transfer_mode == "host";
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
    inferbridge::native::Precision precision;
    try {
        precision = inferbridge::native::precision_from_parameters_json(parameters);
    } catch (const std::exception& error) {
        return fail(runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
    }
    const inferbridge::native::ScopedPrecisionRequest precision_scope(precision);
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    try {
        vda_native::ModelFile artifact(
            path, VDA_MODEL_VITS_RELATIVE_32_FRAMES, false);
        model->model_kind = artifact.model_kind();
        const auto& config = vda_native::model_config(model->model_kind);
        model->metric = config.metric;
        std::string encoder;
        if (json_string(parameters, "Encoder", encoder)) {
            const std::string selector = config.metric ?
                std::string("metric_") + config.encoder : config.encoder;
            if (encoder != config.encoder && encoder != selector) {
                delete model;
                return fail(
                    runtime, IBRH_ERROR_INVALID_ARGUMENT,
                    "VDA Encoder does not match the derived artifact");
            }
        }
    } catch (const std::exception& error) {
        delete model;
        return fail(runtime, IBRH_ERROR_INVALID_ARGUMENT, error.what());
    }
    if (!input_size(parameters, model->input_size, model->input_size)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "VDA Size must be a multiple of 14 up to 4096");
    }
#if defined(VDA_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u && !runtime->force_host_transfers) {
        try {
            model->external_gpu = vda_native::create_external_gpu(
                path, static_cast<uint32_t>(runtime->vulkan_device_index),
                model->model_kind);
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
            path.c_str(), model->model_kind,
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != VDA_STATUS_OK) {
            const std::string message =
                std::string("VDA model load failed: ") + vda_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
#if defined(VDA_WITH_METAL) && defined(__APPLE__)
        if (!runtime->force_host_transfers) {
            try {
                model->external_gpu =
                    vda_native::create_metal_external_gpu(model->context);
            } catch (const std::exception& error) {
                vda_destroy(model->context);
                delete model;
                return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                    error.what());
            }
        }
#endif
    }
#if defined(VDA_WITH_EXTERNAL_GPU)
    if (model->external_gpu) {
        try {
            model->worker = std::thread(worker_loop, model);
        } catch (...) {
            delete model;
            return fail(runtime, IBRH_ERROR_INTERNAL,
                "VDA could not start its inference worker");
        }
    }
#endif
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(VDA_WITH_EXTERNAL_GPU)
    {
        std::lock_guard<std::mutex> lock(model->queue_mutex);
        model->stopping = true;
        for (ibrh_job* job : model->queue) {
            job->cancel_requested.store(true);
            job->state.store(IBRH_JOB_CANCELLED);
            release_job(job);
        }
        model->queue.clear();
    }
    model->queue_condition.notify_all();
    if (model->worker.joinable()) model->worker.join();
    model->external_gpu.reset();
#endif
    vda_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(const ibrh_model* m,size_t n,ibrh_model_io_descriptor* o){if(!m||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*o))return IBRH_ERROR_STRUCT_TOO_SMALL;*o={};o->struct_size=sizeof(*o);o->api_version=IBRH_CURRENT_API_VERSION;o->input_count=o->output_count=1;return IBRH_OK;}
ibrh_result IBRH_CALL model_get_port(const ibrh_model* m,uint32_t d,uint32_t i,size_t n,ibrh_port_descriptor* o){if(!m||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*o))return IBRH_ERROR_STRUCT_TOO_SMALL;if(i||(d!=IBRH_PORT_INPUT&&d!=IBRH_PORT_OUTPUT))return IBRH_ERROR_NOT_FOUND;*o={};o->struct_size=sizeof(*o);o->api_version=IBRH_CURRENT_API_VERSION;o->direction=d;o->semantic=d==IBRH_PORT_INPUT?IBRH_SEMANTIC_IMAGE:IBRH_SEMANTIC_DEPTH;o->payload_type=d==IBRH_PORT_INPUT?IBRH_PIXEL_BGRA8:(m->metric?IBRH_PIXEL_DEPTH_METRIC_FLOAT32:IBRH_PIXEL_DEPTH_FLOAT32);o->pixel_format=o->payload_type;o->accepted_pixel_format_mask=1ull<<o->pixel_format;o->resource_kind=IBRH_RESOURCE_KIND_IMAGE_2D;o->depth=1;o->flags=IBRH_DESCRIPTOR_DYNAMIC_WIDTH|IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;return IBRH_OK;}
ibrh_result IBRH_CALL model_plan_outputs(const ibrh_model* m,size_t n,const ibrh_output_plan_request* r,uint32_t c,ibrh_port_descriptor* o){if(!m||!r||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*r)||r->struct_size<sizeof(*r)||c<1)return IBRH_ERROR_STRUCT_TOO_SMALL;if(r->input_count!=1||!r->inputs)return IBRH_ERROR_INVALID_ARGUMENT;auto x=model_get_port(m,IBRH_PORT_OUTPUT,0,sizeof(o[0]),&o[0]);if(x!=IBRH_OK)return x;o[0].width=r->inputs[0].width;o[0].height=r->inputs[0].height;o[0].flags=0;return IBRH_OK;}

ibrh_result IBRH_CALL submit(ibrh_model* model,size_t n,const ibrh_submit_request* r,ibrh_job** out){
 if(!model||!r||!out)return IBRH_ERROR_INVALID_ARGUMENT;*out=nullptr;if(n<sizeof(*r)||r->struct_size<sizeof(*r))return IBRH_ERROR_STRUCT_TOO_SMALL;
 if(r->input_count!=1||!r->inputs||r->output_count!=1||!r->outputs)return IBRH_ERROR_INVALID_ARGUMENT;
 const auto&s=r->inputs[0];const auto&t=r->outputs[0];const auto&i=s.resource;const auto&o=t.resource;
 uint32_t size=model->input_size;const std::string p=copy_string(r->parameters_json);if(!input_size(p,size,size))return IBRH_ERROR_INVALID_ARGUMENT;
 const uint32_t output_format=model->metric?IBRH_PIXEL_DEPTH_METRIC_FLOAT32:IBRH_PIXEL_DEPTH_FLOAT32;
 if(!i.width||!i.height||o.width!=i.width||o.height!=i.height||o.pixel_format!=output_format)return IBRH_ERROR_INVALID_ARGUMENT;
 std::string reset;bool reset_stream=json_string(p,"Reset",reset)&&reset=="YES";
#if defined(VDA_WITH_VULKAN) && defined(_WIN32)
 if(i.domain==IBRH_RESOURCE_DOMAIN_D3D12){if(!model->external_gpu||o.domain!=IBRH_RESOURCE_DOMAIN_D3D12||i.pixel_format!=IBRH_PIXEL_BGRA8||i.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||o.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||s.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||s.synchronization.operation!=IBRH_SYNC_WAIT||t.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||t.synchronization.operation!=IBRH_SYNC_SIGNAL)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
 uint32_t occupied=model->occupied_slots->load();
 while(occupied<3u&&!model->occupied_slots->compare_exchange_weak(occupied,occupied+1u)){}
 if(occupied>=3u)return IBRH_ERROR_INVALID_STATE;
 auto*j=new(std::nothrow)ibrh_job();if(!j){model->occupied_slots->fetch_sub(1u);return IBRH_ERROR_INTERNAL;}
 j->occupied_slots=model->occupied_slots;j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;
 j->request={static_cast<uintptr_t>(i.native_handle),i.auxiliary_handle,i.width,i.height,size,static_cast<uintptr_t>(s.synchronization.native_handle),s.synchronization.value,static_cast<uintptr_t>(o.native_handle),o.auxiliary_handle,o.width,o.height,static_cast<uintptr_t>(t.synchronization.native_handle),t.synchronization.value,r->source_frame_id,r->timestamp_ns,reset_stream};
 {std::lock_guard<std::mutex>l(model->queue_mutex);if(model->stopping){release_job(j);return IBRH_ERROR_INVALID_STATE;}retain_job(j);model->queue.push_back(j);}
 model->queue_condition.notify_one();*out=j;return IBRH_OK;}
#endif
#if defined(VDA_WITH_METAL) && defined(__APPLE__)
 if(i.domain==IBRH_RESOURCE_DOMAIN_METAL){const auto&wait=s.synchronization;const auto&signal=t.synchronization;const bool no_wait=wait.kind==IBRH_SYNC_NONE;const bool event_wait=wait.kind==IBRH_SYNC_METAL_SHARED_EVENT&&wait.operation==IBRH_SYNC_WAIT&&wait.native_handle_type==IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT&&wait.native_handle!=0u;if(!model->external_gpu||o.domain!=IBRH_RESOURCE_DOMAIN_METAL||i.pixel_format!=IBRH_PIXEL_BGRA8||i.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!i.native_handle||o.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!o.native_handle||(!no_wait&&!event_wait)||signal.kind!=IBRH_SYNC_METAL_SHARED_EVENT||signal.operation!=IBRH_SYNC_SIGNAL||signal.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT||!signal.native_handle||!signal.value)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
 uint32_t occupied=model->occupied_slots->load();while(occupied<3u&&!model->occupied_slots->compare_exchange_weak(occupied,occupied+1u)){}if(occupied>=3u)return IBRH_ERROR_INVALID_STATE;auto*j=new(std::nothrow)ibrh_job();if(!j){model->occupied_slots->fetch_sub(1u);return IBRH_ERROR_INTERNAL;}j->occupied_slots=model->occupied_slots;j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;j->request={static_cast<uintptr_t>(i.native_handle),i.auxiliary_handle,i.width,i.height,size,event_wait?static_cast<uintptr_t>(wait.native_handle):0u,event_wait?wait.value:0u,static_cast<uintptr_t>(o.native_handle),o.auxiliary_handle,o.width,o.height,static_cast<uintptr_t>(signal.native_handle),signal.value,r->source_frame_id,r->timestamp_ns,reset_stream};{std::lock_guard<std::mutex>l(model->queue_mutex);if(model->stopping){release_job(j);return IBRH_ERROR_INVALID_STATE;}retain_job(j);model->queue.push_back(j);}model->queue_condition.notify_one();*out=j;return IBRH_OK;}
#endif
 if(i.domain!=IBRH_RESOURCE_DOMAIN_HOST||o.domain!=IBRH_RESOURCE_DOMAIN_HOST||i.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||o.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||i.pixel_format!=IBRH_PIXEL_BGRA8||s.synchronization.kind!=IBRH_SYNC_NONE||t.synchronization.kind!=IBRH_SYNC_NONE)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
 const auto*bgra=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(i.native_handle))+i.byte_offset;auto*depth=reinterpret_cast<float*>(static_cast<uintptr_t>(o.native_handle)+o.byte_offset);
 {std::lock_guard<std::mutex>l(model->submit_mutex);if(reset_stream){auto q=vda_stream_reset(model->context);if(q!=VDA_STATUS_OK)return fail(model->runtime,status_result(q),vda_last_error());}auto q=vda_infer_stream_bgra8_f32(model->context,bgra,i.row_stride_bytes,i.width,i.height,size,depth,static_cast<size_t>(i.width)*i.height);if(q!=VDA_STATUS_OK)return fail(model->runtime,status_result(q),vda_last_error());}
 auto*j=new(std::nothrow)ibrh_job();if(!j)return IBRH_ERROR_INTERNAL;j->state.store(IBRH_JOB_COMPLETE);j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;*out=j;return IBRH_OK;}
ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
#if defined(VDA_WITH_EXTERNAL_GPU)
    std::shared_ptr<vda_native::ExternalJob> gpu_job;
    {
        std::lock_guard<std::mutex> lock(
            const_cast<ibrh_job*>(job)->gpu_mutex);
        gpu_job = job->gpu_job;
    }
    if (gpu_job) {
        switch (gpu_job->state()) {
            case vda_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING; break;
            case vda_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE; break;
            case vda_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED; break;
        }
    } else
#endif
    status->state = job->state.load();
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(VDA_WITH_EXTERNAL_GPU)
    job->cancel_requested.store(true);
    std::shared_ptr<vda_native::ExternalJob> gpu_job;
    {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        gpu_job = job->gpu_job;
    }
    if (gpu_job) gpu_job->cancel();
    const uint32_t state=job->state.load();
    if(state==IBRH_JOB_QUEUED||state==IBRH_JOB_RUNNING)return IBRH_OK;
#endif
    return IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
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

#if defined(__linux__) && !defined(__ANDROID__) && defined(VDA_WITH_VULKAN)
#include "linux_capture.h"
#include <inferbridge/linux_capture_harness.h>
namespace {
struct LinuxCaptureHooks {
    static ibr_linux_capture_capabilities capabilities(ibrh_model* model) {
        std::lock_guard<std::mutex> lock(model->submit_mutex);
        return vda_linux_capture_capabilities(model->context);
    }
    static void infer(ibrh_model* model,const inferbridge::linux_capture::LinuxDmaBufImage& source,
        const std::string& parameters,float* output,uint64_t,uint64_t) {
        std::lock_guard<std::mutex> lock(model->submit_mutex);
        uint32_t size=model->input_size;
        if(!input_size(parameters,size,size)) throw std::invalid_argument("invalid capture input size");
        std::string reset;
        if(json_string(parameters,"Reset",reset) && reset=="YES" && vda_stream_reset(model->context)!=VDA_STATUS_OK)
            throw std::runtime_error(vda_last_error());
        vda_infer_linux_capture(model->context,source,size,output);
    }
};
}
#include <inferbridge/linux_capture_export.inl>
#endif

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
    api->model_unload = model_unload;api->model_describe_io=model_describe_io;api->model_get_port=model_get_port;api->model_plan_outputs=model_plan_outputs;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;

#if defined(__linux__) && !defined(__ANDROID__) && defined(VDA_WITH_VULKAN)
    inferbridge::linux_capture::HarnessAdapter<LinuxCaptureHooks>::install(api);
#endif
    return IBRH_OK;
}
