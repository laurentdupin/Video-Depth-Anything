#include "video_depth_anything_native.h"

#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" {

uint32_t VDA_CALL vda_abi_version(void) {
    return VDA_ABI_VERSION;
}

const char* VDA_CALL vda_version_string(void) {
    return "0.1.0-model-foundation";
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

}
