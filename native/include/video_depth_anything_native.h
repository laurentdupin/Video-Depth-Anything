#ifndef VIDEO_DEPTH_ANYTHING_NATIVE_H
#define VIDEO_DEPTH_ANYTHING_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(VDA_BUILD_DLL)
#    define VDA_API __declspec(dllexport)
#  else
#    define VDA_API __declspec(dllimport)
#  endif
#  define VDA_CALL __cdecl
#else
#  define VDA_API __attribute__((visibility("default")))
#  define VDA_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VDA_ABI_VERSION 4u

typedef struct vda_context vda_context;

typedef enum vda_status {
    VDA_STATUS_OK = 0,
    VDA_STATUS_INVALID_ARGUMENT = 1,
    VDA_STATUS_MODEL_IO = 2,
    VDA_STATUS_MODEL_FORMAT = 3,
    VDA_STATUS_VULKAN_UNAVAILABLE = 4,
    VDA_STATUS_OUT_OF_MEMORY = 5,
    VDA_STATUS_INFERENCE_FAILED = 6,
    VDA_STATUS_BUFFER_TOO_SMALL = 7,
    VDA_STATUS_UNSUPPORTED = 8,
    VDA_STATUS_INTERNAL_ERROR = 9
} vda_status;

typedef enum vda_model_kind {
    VDA_MODEL_VITS_RELATIVE_32_FRAMES = 0
} vda_model_kind;

typedef struct vda_transfer_counters {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} vda_transfer_counters;

VDA_API uint32_t VDA_CALL vda_abi_version(void);
VDA_API const char* VDA_CALL vda_version_string(void);
VDA_API const char* VDA_CALL vda_status_string(vda_status status);
VDA_API const char* VDA_CALL vda_last_error(void);
VDA_API vda_status VDA_CALL vda_create(
    const char* model_path_utf8,
    vda_model_kind model,
    vda_context** context);
/*
 * Creates a Vulkan tensor context on the zero-based Vulkan physical-device
 * index. This is a real full-graph GPU backend; creation fails instead of
 * silently falling back to CPU.
 */
VDA_API vda_status VDA_CALL vda_create_vulkan(
    const char* model_path_utf8,
    vda_model_kind model,
    uint32_t device_index,
    vda_context** context);
VDA_API void VDA_CALL vda_destroy(vda_context* context);
/*
 * Executes the official 32-frame graph. Input is normalized RGB FP32 in
 * contiguous TCHW order. Output is contiguous THW relative depth.
 * Width and height must be positive multiples of 14.
 */
VDA_API vda_status VDA_CALL vda_infer_tensor_f32(
    vda_context* context,
    const float* normalized_rgb_tchw,
    int32_t frames,
    int32_t width,
    int32_t height,
    float* depth_thw,
    uint64_t depth_elements);

/*
 * Stateful single-frame contract used by InferBridge's Python worker.
 * Input is BGRA8 capture memory. The runtime preserves BGR channel ordering,
 * performs nearest square resize and ImageNet normalization, advances the
 * official 32-frame hidden-state cache, then returns min/max-normalized depth
 * at the source dimensions. This entry point requires a Vulkan context.
 */
VDA_API vda_status VDA_CALL vda_infer_stream_bgra8_f32(
    vda_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t width,
    int32_t height,
    int32_t input_size,
    float* depth_hw,
    uint64_t depth_elements);

VDA_API vda_status VDA_CALL vda_stream_reset(
    vda_context* context);
VDA_API vda_status VDA_CALL vda_get_transfer_counters(
    vda_transfer_counters* counters);

#ifdef __cplusplus
}
#endif

#endif
