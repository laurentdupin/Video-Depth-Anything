#include "encoder_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vda_native {
namespace {

constexpr std::uint32_t kEmbedding = 384;
constexpr std::uint32_t kHeads = 6;
constexpr std::uint32_t kHeadChannels = 64;

const TensorView& tensor(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error("unexpected encoder tensor rank: " + name);
    }
    return result;
}

void linear(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t input_channels,
    const TensorView& weight,
    const TensorView& bias,
    std::vector<float>& output) {
    if (weight.rank != 2 || bias.rank != 1 ||
        weight.dimensions[1] != input_channels ||
        bias.dimensions[0] != weight.dimensions[0] ||
        input.size() != std::uint64_t(rows) * input_channels) {
        throw std::runtime_error("encoder linear shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    output.resize(std::uint64_t(rows) * output_channels);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * input_channels;
        float* destination =
            output.data() + std::uint64_t(row) * output_channels;
        for (std::uint32_t out = 0; out < output_channels; ++out) {
            const float* kernel =
                weight.data + std::uint64_t(out) * input_channels;
            float value = bias.data[out];
            for (std::uint32_t in = 0; in < input_channels; ++in) {
                value += source[in] * kernel[in];
            }
            destination[out] = value;
        }
    }
}

void layer_norm(
    const std::vector<float>& input,
    std::uint32_t rows,
    const TensorView& scale,
    const TensorView& bias,
    std::vector<float>& output) {
    if (input.size() != std::uint64_t(rows) * kEmbedding ||
        scale.rank != 1 || bias.rank != 1 ||
        scale.dimensions[0] != kEmbedding ||
        bias.dimensions[0] != kEmbedding) {
        throw std::runtime_error("encoder layer norm shape mismatch");
    }
    output.resize(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * kEmbedding;
        float* destination =
            output.data() + std::uint64_t(row) * kEmbedding;
        float mean = 0.0f;
        for (std::uint32_t channel = 0;
             channel < kEmbedding;
             ++channel) {
            mean += source[channel];
        }
        mean /= static_cast<float>(kEmbedding);
        float variance = 0.0f;
        for (std::uint32_t channel = 0;
             channel < kEmbedding;
             ++channel) {
            const float difference = source[channel] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(kEmbedding);
        const float inverse = 1.0f / std::sqrt(variance + 1.0e-6f);
        for (std::uint32_t channel = 0;
             channel < kEmbedding;
             ++channel) {
            destination[channel] =
                (source[channel] - mean) * inverse *
                    scale.data[channel] +
                bias.data[channel];
        }
    }
}

float cubic(float distance) {
    constexpr float coefficient = -0.75f;
    distance = std::abs(distance);
    if (distance <= 1.0f) {
        return ((coefficient + 2.0f) * distance -
                (coefficient + 3.0f)) *
                distance * distance +
            1.0f;
    }
    if (distance < 2.0f) {
        return ((coefficient * distance -
                 5.0f * coefficient) *
                    distance +
                8.0f * coefficient) *
                distance -
            4.0f * coefficient;
    }
    return 0.0f;
}

void add_position(
    const ModelFile& model,
    std::vector<float>& tokens,
    std::uint32_t frames,
    std::uint32_t patch_width,
    std::uint32_t patch_height) {
    const TensorView& position =
        tensor(model, "pretrained.pos_embed", 3);
    if (position.dimensions[0] != 1 ||
        position.dimensions[1] != 1370 ||
        position.dimensions[2] != kEmbedding) {
        throw std::runtime_error("unexpected DINO position shape");
    }
    const std::uint32_t token_count =
        1 + patch_width * patch_height;
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        float* class_token =
            tokens.data() + std::uint64_t(frame) *
                token_count * kEmbedding;
        for (std::uint32_t channel = 0;
             channel < kEmbedding;
             ++channel) {
            class_token[channel] += position.data[channel];
        }
        for (std::uint32_t y = 0; y < patch_height; ++y) {
            const float source_y =
                (static_cast<float>(y) + 0.5f) *
                    37.0f /
                    (static_cast<float>(patch_height) + 0.1f) -
                0.5f;
            const int base_y = static_cast<int>(std::floor(source_y));
            for (std::uint32_t x = 0; x < patch_width; ++x) {
                const float source_x =
                    (static_cast<float>(x) + 0.5f) *
                        37.0f /
                        (static_cast<float>(patch_width) + 0.1f) -
                    0.5f;
                const int base_x =
                    static_cast<int>(std::floor(source_x));
                float* destination = class_token +
                    (1 + std::uint64_t(y) * patch_width + x) *
                        kEmbedding;
                for (std::uint32_t channel = 0;
                     channel < kEmbedding;
                     ++channel) {
                    float value = 0.0f;
                    for (int oy = -1; oy <= 2; ++oy) {
                        const int sample_y = std::clamp(
                            base_y + oy, 0, 36);
                        const float wy = cubic(
                            source_y - static_cast<float>(base_y + oy));
                        for (int ox = -1; ox <= 2; ++ox) {
                            const int sample_x = std::clamp(
                                base_x + ox, 0, 36);
                            const float wx = cubic(
                                source_x -
                                static_cast<float>(base_x + ox));
                            value += wy * wx * position.data[
                                (1 + std::uint64_t(sample_y) * 37 +
                                 sample_x) *
                                    kEmbedding +
                                channel];
                        }
                    }
                    destination[channel] += value;
                }
            }
        }
    }
}

void attention(
    const ModelFile& model,
    const std::string& prefix,
    const std::vector<float>& normalized,
    std::uint32_t frames,
    std::uint32_t token_count,
    std::vector<float>& output) {
    const std::uint32_t rows = frames * token_count;
    std::vector<float> qkv;
    linear(
        normalized, rows, kEmbedding,
        tensor(model, prefix + "qkv.weight", 2),
        tensor(model, prefix + "qkv.bias", 1),
        qkv);
    std::vector<float> attended(
        std::uint64_t(rows) * kEmbedding);
    std::vector<float> scores(token_count);
    constexpr float scale = 0.125f;
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        for (std::uint32_t head = 0; head < kHeads; ++head) {
            for (std::uint32_t query_token = 0;
                 query_token < token_count;
                 ++query_token) {
                const float* query = qkv.data() +
                    (std::uint64_t(frame * token_count + query_token) *
                         3 *
                         kEmbedding) +
                    head * kHeadChannels;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::uint32_t key_token = 0;
                     key_token < token_count;
                     ++key_token) {
                    const float* key = qkv.data() +
                        std::uint64_t(frame * token_count + key_token) *
                            3 *
                            kEmbedding +
                        kEmbedding + head * kHeadChannels;
                    float score = 0.0f;
                    for (std::uint32_t channel = 0;
                         channel < kHeadChannels;
                         ++channel) {
                        score += query[channel] * key[channel];
                    }
                    score *= scale;
                    scores[key_token] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0f;
                for (float& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float* destination = attended.data() +
                    std::uint64_t(frame * token_count + query_token) *
                        kEmbedding +
                    head * kHeadChannels;
                for (std::uint32_t channel = 0;
                     channel < kHeadChannels;
                     ++channel) {
                    float value = 0.0f;
                    for (std::uint32_t source_token = 0;
                         source_token < token_count;
                         ++source_token) {
                        const float* source = qkv.data() +
                            std::uint64_t(
                                frame * token_count + source_token) *
                                3 *
                                kEmbedding +
                            2 * kEmbedding + head * kHeadChannels;
                        value += scores[source_token] / denominator *
                            source[channel];
                    }
                    destination[channel] = value;
                }
            }
        }
    }
    linear(
        attended, rows, kEmbedding,
        tensor(model, prefix + "proj.weight", 2),
        tensor(model, prefix + "proj.bias", 1),
        output);
}

void mlp(
    const ModelFile& model,
    const std::string& prefix,
    const std::vector<float>& normalized,
    std::uint32_t rows,
    std::vector<float>& output) {
    std::vector<float> hidden;
    linear(
        normalized, rows, kEmbedding,
        tensor(model, prefix + "fc1.weight", 2),
        tensor(model, prefix + "fc1.bias", 1),
        hidden);
    constexpr float inverse_sqrt_two =
        0.70710678118654752440f;
    for (float& value : hidden) {
        value = 0.5f * value *
            (1.0f + std::erf(value * inverse_sqrt_two));
    }
    linear(
        hidden, rows, kEmbedding * 4,
        tensor(model, prefix + "fc2.weight", 2),
        tensor(model, prefix + "fc2.bias", 1),
        output);
}

}  // namespace

EncoderCpuOutput encoder_cpu(
    const ModelFile& model,
    const float* input,
    std::uint32_t frames,
    std::uint32_t width,
    std::uint32_t height) {
    if (!input || frames == 0 || width == 0 || height == 0 ||
        width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument("invalid CPU encoder input");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t patches = patch_width * patch_height;
    const std::uint32_t token_count = patches + 1;
    const TensorView& patch_weight = tensor(
        model, "pretrained.patch_embed.proj.weight", 4);
    const TensorView& patch_bias = tensor(
        model, "pretrained.patch_embed.proj.bias", 1);
    const TensorView& cls = tensor(model, "pretrained.cls_token", 3);
    std::vector<float> state(
        std::uint64_t(frames) * token_count * kEmbedding);
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        std::copy_n(
            cls.data, kEmbedding,
            state.data() +
                std::uint64_t(frame) * token_count * kEmbedding);
        for (std::uint32_t py = 0; py < patch_height; ++py) {
            for (std::uint32_t px = 0; px < patch_width; ++px) {
                float* destination = state.data() +
                    (std::uint64_t(frame) * token_count +
                     1 + py * patch_width + px) *
                        kEmbedding;
                for (std::uint32_t out = 0;
                     out < kEmbedding;
                     ++out) {
                    float value = patch_bias.data[out];
                    for (std::uint32_t channel = 0;
                         channel < 3;
                         ++channel) {
                        for (std::uint32_t ky = 0; ky < 14; ++ky) {
                            for (std::uint32_t kx = 0; kx < 14; ++kx) {
                                const float source = input[
                                    ((std::uint64_t(frame) * 3 + channel) *
                                         height +
                                     py * 14 + ky) *
                                        width +
                                    px * 14 + kx];
                                const float kernel = patch_weight.data[
                                    ((std::uint64_t(out) * 3 + channel) *
                                         14 +
                                     ky) *
                                        14 +
                                    kx];
                                value += source * kernel;
                            }
                        }
                    }
                    destination[out] = value;
                }
            }
        }
    }
    EncoderCpuOutput result;
    result.frames = frames;
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.patch_tokens = state;
    add_position(model, state, frames, patch_width, patch_height);
    result.prepared_tokens = state;
    result.features.reserve(4);
    const std::uint32_t rows = frames * token_count;
    const std::uint32_t captures[4] = {2, 5, 8, 11};
    std::uint32_t capture = 0;
    for (std::uint32_t block = 0; block < 12; ++block) {
        const std::string prefix =
            "pretrained.blocks." + std::to_string(block) + ".";
        std::vector<float> normalized;
        layer_norm(
            state, rows,
            tensor(model, prefix + "norm1.weight", 1),
            tensor(model, prefix + "norm1.bias", 1),
            normalized);
        std::vector<float> attended;
        attention(
            model, prefix + "attn.", normalized,
            frames, token_count, attended);
        const TensorView& scale1 =
            tensor(model, prefix + "ls1.gamma", 1);
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t channel = 0;
                 channel < kEmbedding;
                 ++channel) {
                state[std::uint64_t(row) * kEmbedding + channel] +=
                    attended[std::uint64_t(row) * kEmbedding + channel] *
                    scale1.data[channel];
            }
        }
        layer_norm(
            state, rows,
            tensor(model, prefix + "norm2.weight", 1),
            tensor(model, prefix + "norm2.bias", 1),
            normalized);
        std::vector<float> feed_forward;
        mlp(
            model, prefix + "mlp.", normalized,
            rows, feed_forward);
        const TensorView& scale2 =
            tensor(model, prefix + "ls2.gamma", 1);
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t channel = 0;
                 channel < kEmbedding;
                 ++channel) {
                state[std::uint64_t(row) * kEmbedding + channel] +=
                    feed_forward[
                        std::uint64_t(row) * kEmbedding + channel] *
                    scale2.data[channel];
            }
        }

        if (capture < 4 && block == captures[capture]) {
            layer_norm(
                state, rows,
                tensor(model, "pretrained.norm.weight", 1),
                tensor(model, "pretrained.norm.bias", 1),
                normalized);
            std::vector<float> feature(
                std::uint64_t(frames) * patches * kEmbedding);
            for (std::uint32_t frame = 0; frame < frames; ++frame) {
                std::copy_n(
                    normalized.data() +
                        (std::uint64_t(frame) * token_count + 1) *
                            kEmbedding,
                    std::uint64_t(patches) * kEmbedding,
                    feature.data() +
                        std::uint64_t(frame) * patches * kEmbedding);
            }
            result.features.push_back(std::move(feature));
            ++capture;
        }
    }
    return result;
}

}  // namespace vda_native
