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

#define VDA_ABI_VERSION 1u

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

VDA_API uint32_t VDA_CALL vda_abi_version(void);
VDA_API const char* VDA_CALL vda_version_string(void);
VDA_API const char* VDA_CALL vda_status_string(vda_status status);
VDA_API const char* VDA_CALL vda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
