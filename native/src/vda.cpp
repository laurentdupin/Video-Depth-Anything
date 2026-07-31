#include "video_depth_anything_native.h"

#include "dpt_cpu.h"
#include "encoder_cpu.h"
#include "model.h"
#if defined(VDA_WITH_VULKAN)
#include "dpt_gpu.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "operators.h"
#include "temporal_gpu.h"
#include "vulkan.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(VDA_WITH_VULKAN)
struct vda_stream_entry {
    std::array<vda_native::TemporalFrameCache, 4> modules;
};
#endif

struct vda_context {
    std::unique_ptr<vda_native::ModelFile> model;
#if defined(VDA_WITH_VULKAN)
    std::unique_ptr<vda_native::VulkanContext> vulkan;
    std::unique_ptr<vda_native::GpuModel> gpu_model;
    std::unique_ptr<vda_native::VulkanOperators> operators;
    std::unique_ptr<vda_native::VdaGpuEncoder> encoder;
    std::unique_ptr<vda_native::VdaGpuDpt> dpt;
    std::vector<std::shared_ptr<vda_stream_entry>> stream_cache;
    std::int64_t stream_id = -1;
    std::uint32_t stream_width = 0;
    std::uint32_t stream_height = 0;
    std::uint32_t stream_input_size = 0;
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

std::vector<float> preprocess_stream_bgra(
    const std::uint8_t* bgra,
    std::uint64_t stride,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t size) {
    const std::uint64_t plane = std::uint64_t(size) * size;
    std::vector<float> output(
        static_cast<std::size_t>(3 * plane));
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float deviation[3] = {0.229f, 0.224f, 0.225f};
    for (std::uint32_t y = 0; y < size; ++y) {
        const std::uint32_t sy = std::min(
            height - 1,
            static_cast<std::uint32_t>(
                std::uint64_t(y) * height / size));
        const std::uint8_t* row = bgra + std::uint64_t(sy) * stride;
        for (std::uint32_t x = 0; x < size; ++x) {
            const std::uint32_t sx = std::min(
                width - 1,
                static_cast<std::uint32_t>(
                    std::uint64_t(x) * width / size));
            const std::uint64_t pixel =
                std::uint64_t(y) * size + x;
            for (std::uint32_t channel = 0;
                 channel < 3; ++channel) {
                output[static_cast<std::size_t>(
                    std::uint64_t(channel) * plane + pixel)] =
                    (row[std::uint64_t(sx) * 4 + channel] /
                        255.0f - mean[channel]) /
                    deviation[channel];
            }
        }
    }
    return output;
}

void resize_normalize_stream_depth(
    const std::vector<float>& source,
    std::uint32_t source_size,
    float* destination,
    std::uint32_t width,
    std::uint32_t height) {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t y = 0; y < height; ++y) {
        const float source_y = height > 1
            ? static_cast<float>(y) * (source_size - 1) /
                static_cast<float>(height - 1)
            : 0.0f;
        const std::uint32_t y0 =
            static_cast<std::uint32_t>(source_y);
        const std::uint32_t y1 =
            std::min(y0 + 1, source_size - 1);
        const float fy = source_y - y0;
        for (std::uint32_t x = 0; x < width; ++x) {
            const float source_x = width > 1
                ? static_cast<float>(x) * (source_size - 1) /
                    static_cast<float>(width - 1)
                : 0.0f;
            const std::uint32_t x0 =
                static_cast<std::uint32_t>(source_x);
            const std::uint32_t x1 =
                std::min(x0 + 1, source_size - 1);
            const float fx = source_x - x0;
            const float top =
                source[std::uint64_t(y0) * source_size + x0] *
                    (1.0f - fx) +
                source[std::uint64_t(y0) * source_size + x1] * fx;
            const float bottom =
                source[std::uint64_t(y1) * source_size + x0] *
                    (1.0f - fx) +
                source[std::uint64_t(y1) * source_size + x1] * fx;
            const float value =
                top * (1.0f - fy) + bottom * fy;
            destination[std::uint64_t(y) * width + x] = value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    const float range = maximum - minimum;
    for (std::uint64_t index = 0;
         index < std::uint64_t(width) * height; ++index) {
        destination[index] = range > 0.0f
            ? (destination[index] - minimum) / range
            : 0.0f;
    }
}
}

extern "C" {

uint32_t VDA_CALL vda_abi_version(void) {
    return VDA_ABI_VERSION;
}

const char* VDA_CALL vda_version_string(void) {
    return "0.4.0-streaming-cpu-vulkan-full-graph";
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

vda_status VDA_CALL vda_stream_reset(
    vda_context* context) {
    if (!context || !context->model) {
        return fail(
            VDA_STATUS_INVALID_ARGUMENT,
            "invalid stream reset context");
    }
#if !defined(VDA_WITH_VULKAN)
    return fail(
        VDA_STATUS_UNSUPPORTED,
        "streaming requires the Vulkan executor");
#else
    if (!context->vulkan) {
        return fail(
            VDA_STATUS_UNSUPPORTED,
            "streaming requires a Vulkan context");
    }
    context->stream_cache.clear();
    context->stream_id = -1;
    context->stream_width = 0;
    context->stream_height = 0;
    context->stream_input_size = 0;
    last_error.clear();
    return VDA_STATUS_OK;
#endif
}

vda_status VDA_CALL vda_get_transfer_counters(
    vda_transfer_counters* counters) {
    if (counters == nullptr || counters->struct_size < sizeof(*counters))
        return fail(VDA_STATUS_INVALID_ARGUMENT,
                    "invalid VDA transfer counter descriptor");
    *counters = {};
    counters->struct_size = sizeof(*counters);
    counters->abi_version = VDA_ABI_VERSION;
#if defined(VDA_WITH_VULKAN)
    vda_native::global_transfer_counters(
        counters->tensor_upload_bytes, counters->tensor_download_bytes);
#endif
    last_error.clear();
    return VDA_STATUS_OK;
}

vda_status VDA_CALL vda_infer_stream_bgra8_f32(
    vda_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t width,
    int32_t height,
    int32_t input_size,
    float* depth,
    uint64_t depth_elements) {
    if (!context || !context->model || !bgra || !depth ||
        width <= 0 || height <= 0 || input_size <= 0 ||
        input_size % 14 != 0 ||
        bgra_stride_bytes <
            static_cast<std::uint64_t>(width) * 4 ||
        depth_elements <
            static_cast<std::uint64_t>(width) * height) {
        return fail(
            VDA_STATUS_INVALID_ARGUMENT,
            "invalid streaming BGRA inference input");
    }
#if !defined(VDA_WITH_VULKAN)
    return fail(
        VDA_STATUS_UNSUPPORTED,
        "streaming requires the Vulkan executor");
#else
    if (!context->vulkan || !context->encoder || !context->dpt) {
        return fail(
            VDA_STATUS_UNSUPPORTED,
            "streaming requires a Vulkan context");
    }
    if (!context->stream_cache.empty() &&
        (context->stream_width != static_cast<std::uint32_t>(width) ||
         context->stream_height != static_cast<std::uint32_t>(height) ||
         context->stream_input_size !=
            static_cast<std::uint32_t>(input_size))) {
        return fail(
            VDA_STATUS_INVALID_ARGUMENT,
            "stream dimensions changed without vda_stream_reset");
    }
    return protect([&] {
        const std::uint32_t network_size =
            static_cast<std::uint32_t>(input_size);
        std::vector<float> prepared = preprocess_stream_bgra(
            bgra, bgra_stride_bytes,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            network_size);
        vda_native::VulkanBuffer image =
            context->vulkan->create_device_buffer(
                prepared.size() * sizeof(float));
        context->vulkan->upload(
            image, prepared.data(),
            prepared.size() * sizeof(float));

        const auto run = [&](
            const std::vector<std::shared_ptr<vda_stream_entry>>& selected,
            std::shared_ptr<vda_stream_entry>& output) {
            std::vector<const vda_native::TemporalFrameCache*>
                history[4];
            for (const auto& entry : selected) {
                for (std::uint32_t module = 0;
                     module < 4; ++module) {
                    history[module].push_back(
                        &entry->modules[module]);
                }
            }
            output = std::make_shared<vda_stream_entry>();
            vda_native::TemporalFrameCache* output_modules[4] = {
                &output->modules[0], &output->modules[1],
                &output->modules[2], &output->modules[3]};
            return context->dpt->forward_stream(
                context->encoder->forward(
                    image, 1, network_size, network_size),
                history, output_modules);
        };

        if (context->stream_cache.empty()) {
            std::shared_ptr<vda_stream_entry> seed;
            std::vector<std::shared_ptr<vda_stream_entry>> empty;
            (void)run(empty, seed);
            context->stream_cache.assign(32, seed);
            context->stream_width =
                static_cast<std::uint32_t>(width);
            context->stream_height =
                static_cast<std::uint32_t>(height);
            context->stream_input_size = network_size;
        }

        std::vector<std::shared_ptr<vda_stream_entry>> selected;
        selected.reserve(31);
        selected.push_back(context->stream_cache[0]);
        selected.push_back(context->stream_cache[1]);
        const std::size_t tail =
            context->stream_cache.size() - 29;
        selected.insert(
            selected.end(),
            context->stream_cache.begin() +
                static_cast<std::ptrdiff_t>(tail),
            context->stream_cache.end());
        std::shared_ptr<vda_stream_entry> current;
        vda_native::FeatureMap result = run(selected, current);
        context->stream_cache.push_back(current);
        ++context->stream_id;
        if (context->stream_id + 32 > 42) {
            context->stream_cache.erase(
                context->stream_cache.begin() + 1);
        }

        std::vector<float> network_depth(
            static_cast<std::size_t>(
                std::uint64_t(network_size) * network_size));
        context->vulkan->download(
            result.buffer, network_depth.data(),
            network_depth.size() * sizeof(float));
        resize_normalize_stream_depth(
            network_depth, network_size, depth,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height));
    });
#endif
}

}
