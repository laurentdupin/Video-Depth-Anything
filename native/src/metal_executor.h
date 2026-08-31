#pragma once

#include "model.h"
#include "model_config.h"
#include "external_gpu.h"

#include <cstdint>
#include <memory>

namespace vda_native {

class MetalExecutor {
public:
    MetalExecutor(const ModelFile& model, const ModelConfig& config);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    void reset_stream();
    void infer_tensor(
        const float* normalized_rgb_tchw,
        std::uint32_t frames,
        std::uint32_t width,
        std::uint32_t height,
        float* depth_thw);
    void infer_stream(
        const float* normalized_rgb_chw,
        std::uint32_t size,
        float* depth_hw);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vda_native
