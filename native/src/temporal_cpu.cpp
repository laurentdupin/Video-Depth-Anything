#include "temporal_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vda_native {
namespace {

const TensorView& weight(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& tensor = model.tensor(name);
    if (tensor.rank != rank) {
        throw std::runtime_error(
            "unexpected temporal tensor rank: " + name);
    }
    return tensor;
}

void linear(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t input_channels,
    const TensorView& weights,
    const TensorView* bias,
    std::vector<float>& output) {
    if (weights.rank != 2 ||
        weights.dimensions[1] != input_channels ||
        input.size() != std::uint64_t(rows) * input_channels) {
        throw std::runtime_error("temporal linear shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weights.dimensions[0]);
    if (bias &&
        (bias->rank != 1 ||
         bias->dimensions[0] != output_channels)) {
        throw std::runtime_error("temporal linear bias mismatch");
    }
    output.resize(std::uint64_t(rows) * output_channels);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * input_channels;
        float* destination =
            output.data() + std::uint64_t(row) * output_channels;
        for (std::uint32_t out = 0; out < output_channels; ++out) {
            const float* kernel =
                weights.data + std::uint64_t(out) * input_channels;
            float value = bias ? bias->data[out] : 0.0f;
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
    std::uint32_t channels,
    const TensorView& scale,
    const TensorView& bias,
    std::vector<float>& output) {
    if (input.size() != std::uint64_t(rows) * channels ||
        scale.rank != 1 || bias.rank != 1 ||
        scale.dimensions[0] != channels ||
        bias.dimensions[0] != channels) {
        throw std::runtime_error("temporal layer norm shape mismatch");
    }
    output.resize(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * channels;
        float* destination =
            output.data() + std::uint64_t(row) * channels;
        float mean = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            mean += source[channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0f;
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            const float difference = source[channel] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(channels);
        const float inverse = 1.0f / std::sqrt(variance + 1.0e-5f);
        for (std::uint32_t channel = 0; channel < channels; ++channel) {
            destination[channel] =
                (source[channel] - mean) * inverse * scale.data[channel] +
                bias.data[channel];
        }
    }
}

void add_in_place(
    std::vector<float>& destination,
    const std::vector<float>& value) {
    if (destination.size() != value.size()) {
        throw std::runtime_error("temporal residual shape mismatch");
    }
    for (std::size_t index = 0; index < destination.size(); ++index) {
        destination[index] += value[index];
    }
}

std::string transformer_prefix(std::uint32_t module_index) {
    return "head.motion_modules." + std::to_string(module_index) +
        ".temporal_transformer.";
}

void temporal_attention(
    const ModelFile& model,
    const std::string& prefix,
    const std::vector<float>& input,
    std::uint32_t sequences,
    std::uint32_t frames,
    std::uint32_t channels,
    std::vector<float>& output) {
    constexpr std::uint32_t heads = 8;
    if (channels % heads != 0 ||
        input.size() !=
            std::uint64_t(sequences) * frames * channels) {
        throw std::runtime_error("temporal attention shape mismatch");
    }
    const std::uint32_t head_channels = channels / heads;
    const std::uint32_t rows = sequences * frames;
    const TensorView& position = weight(
        model, prefix + "pos_encoder.pe", 3);
    if (position.dimensions[0] != 1 ||
        position.dimensions[1] < frames ||
        position.dimensions[2] != channels) {
        throw std::runtime_error("temporal position shape mismatch");
    }
    std::vector<float> positioned(input);
    for (std::uint32_t sequence = 0; sequence < sequences; ++sequence) {
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            float* row = positioned.data() +
                (std::uint64_t(sequence) * frames + frame) * channels;
            const float* pe =
                position.data + std::uint64_t(frame) * channels;
            for (std::uint32_t channel = 0;
                 channel < channels;
                 ++channel) {
                row[channel] += pe[channel];
            }
        }
    }

    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    linear(
        positioned, rows, channels,
        weight(model, prefix + "to_q.weight", 2), nullptr, query);
    linear(
        positioned, rows, channels,
        weight(model, prefix + "to_k.weight", 2), nullptr, key);
    linear(
        positioned, rows, channels,
        weight(model, prefix + "to_v.weight", 2), nullptr, value);

    std::vector<float> attended(input.size());
    std::vector<float> scores(frames);
    const float scale = 1.0f /
        std::sqrt(static_cast<float>(head_channels));
    for (std::uint32_t sequence = 0;
         sequence < sequences;
         ++sequence) {
        for (std::uint32_t head = 0; head < heads; ++head) {
            const std::uint32_t base_channel = head * head_channels;
            for (std::uint32_t query_frame = 0;
                 query_frame < frames;
                 ++query_frame) {
                const float* q = query.data() +
                    (std::uint64_t(sequence) * frames + query_frame) *
                        channels +
                    base_channel;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::uint32_t key_frame = 0;
                     key_frame < frames;
                     ++key_frame) {
                    const float* k = key.data() +
                        (std::uint64_t(sequence) * frames + key_frame) *
                            channels +
                        base_channel;
                    float score = 0.0f;
                    for (std::uint32_t channel = 0;
                         channel < head_channels;
                         ++channel) {
                        score += q[channel] * k[channel];
                    }
                    score *= scale;
                    scores[key_frame] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0f;
                for (std::uint32_t frame = 0; frame < frames; ++frame) {
                    scores[frame] = std::exp(scores[frame] - maximum);
                    denominator += scores[frame];
                }
                float* destination = attended.data() +
                    (std::uint64_t(sequence) * frames + query_frame) *
                        channels +
                    base_channel;
                for (std::uint32_t channel = 0;
                     channel < head_channels;
                     ++channel) {
                    float result = 0.0f;
                    for (std::uint32_t frame = 0;
                         frame < frames;
                         ++frame) {
                        const float* v = value.data() +
                            (std::uint64_t(sequence) * frames + frame) *
                                channels +
                            base_channel;
                        result +=
                            scores[frame] / denominator * v[channel];
                    }
                    destination[channel] = result;
                }
            }
        }
    }
    linear(
        attended, rows, channels,
        weight(model, prefix + "to_out.0.weight", 2),
        &weight(model, prefix + "to_out.0.bias", 1),
        output);
}

void feed_forward(
    const ModelFile& model,
    const std::string& prefix,
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t channels,
    std::vector<float>& output) {
    std::vector<float> projected;
    linear(
        input, rows, channels,
        weight(model, prefix + "net.0.proj.weight", 2),
        &weight(model, prefix + "net.0.proj.bias", 1),
        projected);
    if (projected.size() !=
        std::uint64_t(rows) * channels * 8) {
        throw std::runtime_error("temporal GEGLU shape mismatch");
    }
    std::vector<float> gated(std::uint64_t(rows) * channels * 4);
    const std::uint32_t inner = channels * 4;
    const float inverse_sqrt_two =
        0.70710678118654752440f;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            projected.data() + std::uint64_t(row) * inner * 2;
        float* destination =
            gated.data() + std::uint64_t(row) * inner;
        for (std::uint32_t channel = 0; channel < inner; ++channel) {
            const float gate = source[inner + channel];
            const float gelu = 0.5f * gate *
                (1.0f + std::erf(gate * inverse_sqrt_two));
            destination[channel] = source[channel] * gelu;
        }
    }
    linear(
        gated, rows, inner,
        weight(model, prefix + "net.2.weight", 2),
        &weight(model, prefix + "net.2.bias", 1),
        output);
}

}  // namespace

void temporal_module_cpu(
    const ModelFile& model,
    std::uint32_t module_index,
    const float* input_cfhw,
    std::uint32_t channels,
    std::uint32_t frames,
    std::uint32_t height,
    std::uint32_t width,
    std::vector<float>& output_cfhw) {
    if (!input_cfhw || module_index >= 4 ||
        channels == 0 || channels % 32 != 0 ||
        frames == 0 || frames > 32 ||
        height == 0 || width == 0) {
        throw std::invalid_argument("invalid temporal module input");
    }
    const std::uint32_t spatial = height * width;
    const std::uint64_t count =
        std::uint64_t(channels) * frames * spatial;
    const std::string prefix = transformer_prefix(module_index);
    const TensorView& group_scale =
        weight(model, prefix + "norm.weight", 1);
    const TensorView& group_bias =
        weight(model, prefix + "norm.bias", 1);
    if (group_scale.dimensions[0] != channels ||
        group_bias.dimensions[0] != channels) {
        throw std::runtime_error("temporal group norm shape mismatch");
    }

    std::vector<float> frame_spatial_channels(count);
    const std::uint32_t channels_per_group = channels / 32;
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        for (std::uint32_t group = 0; group < 32; ++group) {
            float mean = 0.0f;
            const std::uint32_t group_values =
                channels_per_group * spatial;
            for (std::uint32_t local = 0;
                 local < channels_per_group;
                 ++local) {
                const std::uint32_t channel =
                    group * channels_per_group + local;
                for (std::uint32_t position = 0;
                     position < spatial;
                     ++position) {
                    mean += input_cfhw[
                        (std::uint64_t(channel) * frames + frame) *
                            spatial +
                        position];
                }
            }
            mean /= static_cast<float>(group_values);
            float variance = 0.0f;
            for (std::uint32_t local = 0;
                 local < channels_per_group;
                 ++local) {
                const std::uint32_t channel =
                    group * channels_per_group + local;
                for (std::uint32_t position = 0;
                     position < spatial;
                     ++position) {
                    const float difference = input_cfhw[
                        (std::uint64_t(channel) * frames + frame) *
                            spatial +
                        position] - mean;
                    variance += difference * difference;
                }
            }
            variance /= static_cast<float>(group_values);
            const float inverse =
                1.0f / std::sqrt(variance + 1.0e-6f);
            for (std::uint32_t local = 0;
                 local < channels_per_group;
                 ++local) {
                const std::uint32_t channel =
                    group * channels_per_group + local;
                for (std::uint32_t position = 0;
                     position < spatial;
                     ++position) {
                    const float source = input_cfhw[
                        (std::uint64_t(channel) * frames + frame) *
                            spatial +
                        position];
                    frame_spatial_channels[
                        (std::uint64_t(frame) * spatial + position) *
                            channels +
                        channel] =
                        (source - mean) * inverse *
                            group_scale.data[channel] +
                        group_bias.data[channel];
                }
            }
        }
    }

    std::vector<float> projected;
    linear(
        frame_spatial_channels,
        frames * spatial,
        channels,
        weight(model, prefix + "proj_in.weight", 2),
        &weight(model, prefix + "proj_in.bias", 1),
        projected);

    std::vector<float> state(count);
    for (std::uint32_t position = 0; position < spatial; ++position) {
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            std::copy_n(
                projected.data() +
                    (std::uint64_t(frame) * spatial + position) * channels,
                channels,
                state.data() +
                    (std::uint64_t(position) * frames + frame) * channels);
        }
    }

    const std::string block =
        prefix + "transformer_blocks.0.";
    for (std::uint32_t attention = 0; attention < 2; ++attention) {
        const std::string norm =
            block + "norms." + std::to_string(attention);
        std::vector<float> normalized;
        layer_norm(
            state,
            spatial * frames,
            channels,
            weight(model, norm + ".weight", 1),
            weight(model, norm + ".bias", 1),
            normalized);
        std::vector<float> attended;
        temporal_attention(
            model,
            block + "attention_blocks." +
                std::to_string(attention) + ".",
            normalized,
            spatial,
            frames,
            channels,
            attended);
        add_in_place(state, attended);
    }

    std::vector<float> normalized;
    layer_norm(
        state,
        spatial * frames,
        channels,
        weight(model, block + "ff_norm.weight", 1),
        weight(model, block + "ff_norm.bias", 1),
        normalized);
    std::vector<float> fed;
    feed_forward(
        model, block + "ff.", normalized,
        spatial * frames, channels, fed);
    add_in_place(state, fed);

    std::vector<float> frame_order(count);
    for (std::uint32_t position = 0; position < spatial; ++position) {
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            std::copy_n(
                state.data() +
                    (std::uint64_t(position) * frames + frame) * channels,
                channels,
                frame_order.data() +
                    (std::uint64_t(frame) * spatial + position) * channels);
        }
    }
    std::vector<float> projected_out;
    linear(
        frame_order,
        frames * spatial,
        channels,
        weight(model, prefix + "proj_out.weight", 2),
        &weight(model, prefix + "proj_out.bias", 1),
        projected_out);

    output_cfhw.resize(count);
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            for (std::uint32_t position = 0;
                 position < spatial;
                 ++position) {
                const std::uint64_t output_index =
                    (std::uint64_t(channel) * frames + frame) *
                        spatial +
                    position;
                output_cfhw[output_index] =
                    projected_out[
                        (std::uint64_t(frame) * spatial + position) *
                            channels +
                        channel] +
                    input_cfhw[output_index];
            }
        }
    }
}

}  // namespace vda_native
