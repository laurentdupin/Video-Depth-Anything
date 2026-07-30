#include "video_depth_anything_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    vda_context* context = 0;
    assert(vda_abi_version() == VDA_ABI_VERSION);
    assert(strcmp(vda_status_string(VDA_STATUS_OK), "ok") == 0);
    assert(vda_version_string() != 0);
    assert(vda_last_error() != 0);
    assert(
        vda_create(
            0, VDA_MODEL_VITS_RELATIVE_32_FRAMES, &context) ==
        VDA_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    assert(
        vda_infer_stream_bgra8_f32(
            0, 0, 0, 0, 0, 0, 0, 0) ==
        VDA_STATUS_INVALID_ARGUMENT);
    assert(
        vda_stream_reset(0) == VDA_STATUS_INVALID_ARGUMENT);
    vda_destroy(0);
    return 0;
}
