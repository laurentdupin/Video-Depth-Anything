#include "video_depth_anything_native.h"

#include "dpt_cpu.h"
#include "encoder_cpu.h"
#include "model.h"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct vda_context {
    std::unique_ptr<vda_native::ModelFile> model;
};

namespace {
thread_local std::string last_error;

vda_status fail(vda_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

template <typename Function>
vda_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return VDA_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(VDA_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(VDA_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(VDA_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(VDA_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}
}

extern "C" {

uint32_t VDA_CALL vda_abi_version(void) {
    return VDA_ABI_VERSION;
}

const char* VDA_CALL vda_version_string(void) {
    return "0.2.0-cpu-graph";
}

const char* VDA_CALL vda_status_string(vda_status status) {
    switch (status) {
        case VDA_STATUS_OK: return "ok";
        case VDA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case VDA_STATUS_MODEL_IO: return "model I/O error";
        case VDA_STATUS_MODEL_FORMAT: return "invalid model format";
        case VDA_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case VDA_STATUS_OUT_OF_MEMORY: return "out of memory";
        case VDA_STATUS_INFERENCE_FAILED: return "inference failed";
        case VDA_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case VDA_STATUS_UNSUPPORTED: return "unsupported";
        case VDA_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

const char* VDA_CALL vda_last_error(void) {
    return last_error.c_str();
}

vda_status VDA_CALL vda_create(
    const char* model_path_utf8,
    vda_model_kind model,
    vda_context** context) {
    if (!context) {
        return fail(VDA_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (!model_path_utf8 || model_path_utf8[0] == '\0' ||
        model != VDA_MODEL_VITS_RELATIVE_32_FRAMES) {
        return fail(VDA_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
    return protect([&] {
        auto result = std::make_unique<vda_context>();
        result->model = std::make_unique<vda_native::ModelFile>(
            model_path_utf8, model);
        *context = result.release();
    });
}

void VDA_CALL vda_destroy(vda_context* context) {
    delete context;
}

vda_status VDA_CALL vda_infer_tensor_f32(
    vda_context* context,
    const float* normalized_rgb_tchw,
    int32_t frames,
    int32_t width,
    int32_t height,
    float* depth_thw,
    uint64_t depth_elements) {
    if (!context || !context->model ||
        !normalized_rgb_tchw || !depth_thw ||
        frames != 32 ||
        width <= 0 || height <= 0 ||
        width % 14 != 0 || height % 14 != 0 ||
        depth_elements <
            std::uint64_t(frames) * width * height) {
        return fail(
            VDA_STATUS_INVALID_ARGUMENT,
            "invalid 32-frame tensor inference input");
    }
    return protect([&] {
        std::vector<float> result = vda_native::dpt_cpu(
            *context->model,
            vda_native::encoder_cpu(
                *context->model,
                normalized_rgb_tchw,
                static_cast<std::uint32_t>(frames),
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height)));
        std::copy(
            result.begin(), result.end(), depth_thw);
    });
}

}
