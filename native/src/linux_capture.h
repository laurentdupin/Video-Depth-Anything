#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct vda_context;
ibr_linux_capture_capabilities vda_linux_capture_capabilities(vda_context *);
void vda_infer_linux_capture(
    vda_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif
