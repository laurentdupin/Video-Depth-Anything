#pragma once

#include "video_depth_anything_native.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace vda_native {

struct ModelConfig {
    vda_model_kind kind;
    const char* artifact_name;
    const char* encoder;
    bool metric;
    std::uint32_t embedding;
    std::uint32_t heads;
    std::uint32_t blocks;
    std::array<std::uint32_t, 4> captures;
    std::uint32_t features;
    std::array<std::uint32_t, 4> project_channels;
};

inline const ModelConfig& model_config(vda_model_kind kind) {
    static constexpr ModelConfig configurations[] = {
        {VDA_MODEL_VITS_RELATIVE_32_FRAMES, "video_depth_anything_vits", "vits",
         false, 384, 6, 12, {2, 5, 8, 11}, 64, {48, 96, 192, 384}},
        {VDA_MODEL_VITB_RELATIVE_32_FRAMES, "video_depth_anything_vitb", "vitb",
         false, 768, 12, 12, {2, 5, 8, 11}, 128, {96, 192, 384, 768}},
        {VDA_MODEL_VITL_RELATIVE_32_FRAMES, "video_depth_anything_vitl", "vitl",
         false, 1024, 16, 24, {4, 11, 17, 23}, 256,
         {256, 512, 1024, 1024}},
        {VDA_MODEL_VITS_METRIC_32_FRAMES, "metric_video_depth_anything_vits", "vits",
         true, 384, 6, 12, {2, 5, 8, 11}, 64, {48, 96, 192, 384}},
        {VDA_MODEL_VITB_METRIC_32_FRAMES, "metric_video_depth_anything_vitb", "vitb",
         true, 768, 12, 12, {2, 5, 8, 11}, 128, {96, 192, 384, 768}},
        {VDA_MODEL_VITL_METRIC_32_FRAMES, "metric_video_depth_anything_vitl", "vitl",
         true, 1024, 16, 24, {4, 11, 17, 23}, 256,
         {256, 512, 1024, 1024}},
    };
    const auto index = static_cast<std::uint32_t>(kind);
    if (index >= sizeof(configurations) / sizeof(configurations[0])) {
        throw std::invalid_argument("unsupported VDA model kind");
    }
    return configurations[index];
}

inline bool valid_model_kind(std::uint32_t kind) noexcept {
    return kind <= static_cast<std::uint32_t>(
        VDA_MODEL_VITL_METRIC_32_FRAMES);
}

}  // namespace vda_native
