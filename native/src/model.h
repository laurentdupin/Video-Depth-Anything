#pragma once

#include "video_depth_anything_native.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace vda_native {

struct TensorView {
    const float* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
    std::uint32_t crc32 = 0;
};

struct ModelDerivation {
    bool present = false;
    std::array<std::uint8_t, 32> canonical_sha256{};
    std::string converter;
    std::uint32_t format_version = 0;
    vda_model_kind model = VDA_MODEL_VITS_RELATIVE_32_FRAMES;
};

class ModelFile {
public:
    ModelFile(const std::string& path_utf8, vda_model_kind expected_model);
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ~ModelFile();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    const ModelDerivation& derivation() const { return derivation_; }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::unordered_map<std::string_view, TensorView> tensors_;
    ModelDerivation derivation_;
};

}  // namespace vda_native
