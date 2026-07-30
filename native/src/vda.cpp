#include "video_depth_anything_native.h"

#include "dpt_cpu.h"
#include "encoder_cpu.h"
#include "model.h"
#if defined(VDA_WITH_VULKAN)
#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "operators.h"
#include "vulkan.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct vda_context {
    std::unique_ptr<vda_native::ModelFile> model;
#if defined(VDA_WITH_VULKAN)
    std::unique_ptr<vda_native::VulkanContext> vulkan;
    std::unique_ptr<vda_native::GpuModel> gpu_model;
    std::unique_ptr<vda_native::VulkanOperators> operators;
    std::unique_ptr<vda_native::VdaGpuEncoder> encoder;
    std::unique_ptr<vda_native::VdaGpuDpt> dpt;
#endif
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
    return "0.3.0-cpu-vulkan-full-graph";
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

vda_status VDA_CALL vda_create_vulkan(
    const char* model_path_utf8,
    vda_model_kind model,
    uint32_t device_index,
    vda_context** context) {
#if !defined(VDA_WITH_VULKAN)
    (void)model_path_utf8;
    (void)model;
    (void)device_index;
    if (context) {
        *context = nullptr;
    }
    return fail(
        VDA_STATUS_VULKAN_UNAVAILABLE,
        "this DLL was built without Vulkan");
#else
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
        result->vulkan =
            std::make_unique<vda_native::VulkanContext>(device_index);
        result->gpu_model =
            std::make_unique<vda_native::GpuModel>(
                *result->model, *result->vulkan);
        result->operators =
            std::make_unique<vda_native::VulkanOperators>(
                *result->vulkan);
        result->encoder =
            std::make_unique<vda_native::VdaGpuEncoder>(
                *result->vulkan,
                *result->gpu_model,
                *result->operators);
        result->dpt =
            std::make_unique<vda_native::VdaGpuDpt>(
                *result->vulkan,
                *result->gpu_model,
                *result->operators);
        *context = result.release();
    });
#endif
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
#if defined(VDA_WITH_VULKAN)
        if (context->dpt) {
            const std::uint64_t input_elements =
                std::uint64_t(frames) * 3 * width * height;
            vda_native::VulkanBuffer image =
                context->vulkan->create_device_buffer(
                    input_elements * sizeof(float));
            context->vulkan->upload(
                image, normalized_rgb_tchw,
                input_elements * sizeof(float));
            vda_native::FeatureMap result =
                context->dpt->forward(
                    context->encoder->forward(
                        image,
                        static_cast<std::uint32_t>(frames),
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height)));
            context->vulkan->download(
                result.buffer,
                depth_thw,
                std::uint64_t(frames) * width * height *
                    sizeof(float));
            return;
        }
#endif
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
