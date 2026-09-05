#if defined(__linux__) && !defined(__ANDROID__)
extern "C" vda_status VDA_CALL vda_infer_stream_bgra8_f32(
    vda_context *context, const uint8_t *bgra, uint64_t stride, int32_t width,
    int32_t height, int32_t size, float *depth, uint64_t count) {
  return vda_infer_stream_bgra8_f32_linux_impl(
      context, bgra, stride, width, height, size, depth, count, nullptr);
}
ibr_linux_capture_capabilities
vda_linux_capture_capabilities(vda_context *context) {
#if defined(VDA_WITH_VULKAN)
  if (context->vulkan)
    return context->vulkan->linux_capture_capabilities();
#endif
  return {};
}
void vda_infer_linux_capture(
    vda_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint32_t size,
    float *output) {
  const auto status = vda_infer_stream_bgra8_f32_linux_impl(
      context, nullptr, source.row_stride, source.width, source.height, size,
      output, uint64_t(source.width) * source.height, &source);
  if (status != 0)
    throw std::runtime_error(vda_last_error());
}
#endif
