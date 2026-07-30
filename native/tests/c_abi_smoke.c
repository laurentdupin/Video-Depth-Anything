#include "video_depth_anything_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(vda_abi_version() == VDA_ABI_VERSION);
    assert(strcmp(vda_status_string(VDA_STATUS_OK), "ok") == 0);
    assert(vda_version_string() != 0);
    assert(vda_last_error() != 0);
    return 0;
}
