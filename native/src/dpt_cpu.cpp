#include "dpt_cpu.h"
#include "model_config.h"

#include "temporal_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vda_native {
namespace {

struct Feature {
    std::uint32_t frames = 0;
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<float> data;
};

std::uint64_t count(const Feature& feature) {
    return std::uint64_t(feature.frames) * feature.channels *
        feature.height * feature.width;
}

const TensorView& tensor(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error("unexpected DPT tensor rank: " + name);
    }
    return result;
}

Feature conv(
    const ModelFile& model,
    const Feature& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t padding) {
    const TensorView& weight =
        tensor(model, weight_name, 4);
    const bool has_bias = !bias_name.empty();
    const TensorView* bias = has_bias
        ? &tensor(model, bias_name, 1)
        : nullptr;
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    const std::uint32_t input_channels =
        static_cast<std::uint32_t>(weight.dimensions[1]);
    const std::uint32_t kernel_h =
        static_cast<std::uint32_t>(weight.dimensions[2]);
    const std::uint32_t kernel_w =
        static_cast<std::uint32_t>(weight.dimensions[3]);
    if (input.channels != input_channels ||
        (bias && bias->dimensions[0] != output_channels)) {
        throw std::runtime_error("DPT convolution shape mismatch");
    }
    Feature output{
        input.frames,
        output_channels,
        (input.height + 2 * padding - kernel_h) / stride + 1,
        (input.width + 2 * padding - kernel_w) / stride + 1,
        {},
    };
    output.data.resize(count(output));
    for (std::uint32_t frame = 0; frame < output.frames; ++frame) {
        for (std::uint32_t out = 0; out < output.channels; ++out) {
            for (std::uint32_t y = 0; y < output.height; ++y) {
                for (std::uint32_t x = 0; x < output.width; ++x) {
                    float value = bias ? bias->data[out] : 0.0f;
                    for (std::uint32_t in = 0;
                         in < input.channels;
                         ++in) {
                        for (std::uint32_t ky = 0; ky < kernel_h; ++ky) {
                            const int source_y =
                                static_cast<int>(y * stride + ky) -
                                static_cast<int>(padding);
                            if (source_y < 0 ||
                                source_y >=
                                    static_cast<int>(input.height)) {
                                continue;
                            }
                            for (std::uint32_t kx = 0;
                                 kx < kernel_w;
                                 ++kx) {
                                const int source_x =
                                    static_cast<int>(x * stride + kx) -
                                    static_cast<int>(padding);
                                if (source_x < 0 ||
                                    source_x >=
                                        static_cast<int>(input.width)) {
                                    continue;
                                }
                                const float source = input.data[
                                    ((std::uint64_t(frame) *
                                          input.channels +
                                      in) *
                                         input.height +
                                     source_y) *
                                        input.width +
                                    source_x];
                                const float kernel = weight.data[
                                    ((std::uint64_t(out) *
                                          input.channels +
                                      in) *
                                         kernel_h +
                                     ky) *
                                        kernel_w +
                                    kx];
                                value += source * kernel;
                            }
                        }
                    }
                    output.data[
                        ((std::uint64_t(frame) * output.channels + out) *
                             output.height +
                         y) *
                            output.width +
                        x] = value;
                }
            }
        }
    }
    return output;
}

Feature conv_transpose(
    const ModelFile& model,
    const Feature& input,
    const std::string& prefix,
    std::uint32_t stride) {
    const TensorView& weight =
        tensor(model, prefix + ".weight", 4);
    const TensorView& bias =
        tensor(model, prefix + ".bias", 1);
    if (weight.dimensions[0] != input.channels ||
        weight.dimensions[2] != stride ||
        weight.dimensions[3] != stride) {
        throw std::runtime_error("DPT transpose convolution shape mismatch");
    }
    Feature output{
        input.frames,
        static_cast<std::uint32_t>(weight.dimensions[1]),
        input.height * stride,
        input.width * stride,
        {},
    };
    output.data.resize(count(output));
    for (std::uint32_t frame = 0; frame < output.frames; ++frame) {
        for (std::uint32_t out = 0; out < output.channels; ++out) {
            for (std::uint32_t y = 0; y < output.height; ++y) {
                for (std::uint32_t x = 0; x < output.width; ++x) {
                    output.data[
                        ((std::uint64_t(frame) * output.channels + out) *
                             output.height +
                         y) *
                            output.width +
                        x] = bias.data[out];
                }
            }
        }
    }
    for (std::uint32_t frame = 0; frame < input.frames; ++frame) {
        for (std::uint32_t in = 0; in < input.channels; ++in) {
            for (std::uint32_t y = 0; y < input.height; ++y) {
                for (std::uint32_t x = 0; x < input.width; ++x) {
                    const float source = input.data[
                        ((std::uint64_t(frame) * input.channels + in) *
                             input.height +
                         y) *
                            input.width +
                        x];
                    for (std::uint32_t out = 0;
                         out < output.channels;
                         ++out) {
                        for (std::uint32_t ky = 0; ky < stride; ++ky) {
                            for (std::uint32_t kx = 0; kx < stride; ++kx) {
                                output.data[
                                    ((std::uint64_t(frame) *
                                          output.channels +
                                      out) *
                                         output.height +
                                     y * stride + ky) *
                                        output.width +
                                    x * stride + kx] +=
                                    source * weight.data[
                                        ((std::uint64_t(in) *
                                              output.channels +
                                          out) *
                                             stride +
                                         ky) *
                                            stride +
                                        kx];
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}

Feature bilinear(
    const Feature& input,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    Feature output{
        input.frames, input.channels,
        output_height, output_width, {}};
    output.data.resize(count(output));
    for (std::uint32_t frame = 0; frame < input.frames; ++frame) {
        for (std::uint32_t channel = 0;
             channel < input.channels;
             ++channel) {
            for (std::uint32_t y = 0; y < output_height; ++y) {
                const float source_y = output_height == 1
                    ? 0.0f
                    : static_cast<float>(y) *
                        (input.height - 1) / (output_height - 1);
                const std::uint32_t y0 =
                    static_cast<std::uint32_t>(source_y);
                const std::uint32_t y1 =
                    std::min(y0 + 1, input.height - 1);
                const float fy = source_y - y0;
                for (std::uint32_t x = 0; x < output_width; ++x) {
                    const float source_x = output_width == 1
                        ? 0.0f
                        : static_cast<float>(x) *
                            (input.width - 1) / (output_width - 1);
                    const std::uint32_t x0 =
                        static_cast<std::uint32_t>(source_x);
                    const std::uint32_t x1 =
                        std::min(x0 + 1, input.width - 1);
                    const float fx = source_x - x0;
                    const auto at = [&](std::uint32_t sy,
                                        std::uint32_t sx) {
                        return input.data[
                            ((std::uint64_t(frame) * input.channels +
                              channel) *
                                 input.height +
                             sy) *
                                input.width +
                            sx];
                    };
                    output.data[
                        ((std::uint64_t(frame) * output.channels +
                          channel) *
                             output.height +
                         y) *
                            output.width +
                        x] =
                        (at(y0, x0) * (1.0f - fx) +
                         at(y0, x1) * fx) *
                            (1.0f - fy) +
                        (at(y1, x0) * (1.0f - fx) +
                         at(y1, x1) * fx) *
                            fy;
                }
            }
        }
    }
    return output;
}

void relu(Feature& feature) {
    for (float& value : feature.data) {
        value = std::max(value, 0.0f);
    }
}

Feature residual_unit(
    const ModelFile& model,
    const Feature& input,
    const std::string& prefix) {
    Feature value = input;
    relu(value);
    value = conv(
        model, value, prefix + ".conv1.weight",
        prefix + ".conv1.bias", 1, 1);
    relu(value);
    value = conv(
        model, value, prefix + ".conv2.weight",
        prefix + ".conv2.bias", 1, 1);
    for (std::size_t index = 0; index < value.data.size(); ++index) {
        value.data[index] += input.data[index];
    }
    return value;
}

Feature fusion(
    const ModelFile& model,
    Feature path,
    const Feature* skip,
    const std::string& prefix,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    if (skip) {
        Feature processed = residual_unit(
            model, *skip, prefix + ".resConfUnit1");
        if (processed.data.size() != path.data.size()) {
            throw std::runtime_error("DPT fusion skip shape mismatch");
        }
        for (std::size_t index = 0; index < path.data.size(); ++index) {
            path.data[index] += processed.data[index];
        }
    }
    path = residual_unit(
        model, path, prefix + ".resConfUnit2");
    path = bilinear(path, output_height, output_width);
    return conv(
        model, path, prefix + ".out_conv.weight",
        prefix + ".out_conv.bias", 1, 0);
}

Feature apply_temporal(
    const ModelFile& model,
    std::uint32_t module,
    const Feature& input) {
    std::vector<float> cfhw(count(input));
    const std::uint32_t spatial = input.height * input.width;
    for (std::uint32_t channel = 0;
         channel < input.channels;
         ++channel) {
        for (std::uint32_t frame = 0;
             frame < input.frames;
             ++frame) {
            std::copy_n(
                input.data.data() +
                    (std::uint64_t(frame) * input.channels + channel) *
                        spatial,
                spatial,
                cfhw.data() +
                    (std::uint64_t(channel) * input.frames + frame) *
                        spatial);
        }
    }
    std::vector<float> result;
    temporal_module_cpu(
        model, module, cfhw.data(), input.channels,
        input.frames, input.height, input.width, result);
    Feature output{
        input.frames, input.channels,
        input.height, input.width,
        std::vector<float>(count(input))};
    for (std::uint32_t channel = 0;
         channel < input.channels;
         ++channel) {
        for (std::uint32_t frame = 0;
             frame < input.frames;
             ++frame) {
            std::copy_n(
                result.data() +
                    (std::uint64_t(channel) * input.frames + frame) *
                        spatial,
                spatial,
                output.data.data() +
                    (std::uint64_t(frame) * input.channels + channel) *
                        spatial);
        }
    }
    return output;
}

}  // namespace

std::vector<float> dpt_cpu(
    const ModelFile& model,
    EncoderCpuOutput&& encoded) {
    const ModelConfig& config = model_config(model.model_kind());
    if (encoded.features.size() != 4 ||
        encoded.embedding != config.embedding ||
        encoded.frames == 0) {
        throw std::invalid_argument("invalid CPU DPT encoder output");
    }
    Feature layers[4];
    for (std::uint32_t index = 0; index < 4; ++index) {
        Feature tokens{
            encoded.frames,
            encoded.embedding,
            encoded.patch_height,
            encoded.patch_width,
            std::vector<float>(
                std::uint64_t(encoded.frames) * encoded.embedding *
                encoded.patch_height * encoded.patch_width),
        };
        const std::uint32_t patches =
            encoded.patch_width * encoded.patch_height;
        for (std::uint32_t frame = 0;
             frame < encoded.frames;
             ++frame) {
            for (std::uint32_t position = 0;
                 position < patches;
                 ++position) {
                for (std::uint32_t channel = 0;
                     channel < encoded.embedding;
                     ++channel) {
                    tokens.data[
                        (std::uint64_t(frame) * encoded.embedding +
                         channel) *
                            patches +
                        position] =
                        encoded.features[index][
                            (std::uint64_t(frame) * patches + position) *
                                encoded.embedding +
                            channel];
                }
            }
        }
        const std::string project =
            "head.projects." + std::to_string(index);
        layers[index] = conv(
            model, tokens, project + ".weight",
            project + ".bias", 1, 0);
        if (layers[index].channels != config.project_channels[index]) {
            throw std::runtime_error("DPT project channel mismatch");
        }
        if (index == 0) {
            layers[index] = conv_transpose(
                model, layers[index], "head.resize_layers.0", 4);
        } else if (index == 1) {
            layers[index] = conv_transpose(
                model, layers[index], "head.resize_layers.1", 2);
        } else if (index == 3) {
            layers[index] = conv(
                model, layers[index],
                "head.resize_layers.3.weight",
                "head.resize_layers.3.bias", 2, 1);
        }
    }

    layers[2] = apply_temporal(model, 0, layers[2]);
    layers[3] = apply_temporal(model, 1, layers[3]);
    Feature refined[4];
    for (std::uint32_t index = 0; index < 4; ++index) {
        refined[index] = conv(
            model, layers[index],
            "head.scratch.layer" + std::to_string(index + 1) +
                "_rn.weight",
            "", 1, 1);
    }
    Feature path = fusion(
        model, std::move(refined[3]), nullptr,
        "head.scratch.refinenet4",
        refined[2].height, refined[2].width);
    path = apply_temporal(model, 2, path);
    path = fusion(
        model, std::move(path), &refined[2],
        "head.scratch.refinenet3",
        refined[1].height, refined[1].width);
    path = apply_temporal(model, 3, path);
    path = fusion(
        model, std::move(path), &refined[1],
        "head.scratch.refinenet2",
        refined[0].height, refined[0].width);
    path = fusion(
        model, std::move(path), &refined[0],
        "head.scratch.refinenet1",
        refined[0].height * 2, refined[0].width * 2);
    path = conv(
        model, path,
        "head.scratch.output_conv1.weight",
        "head.scratch.output_conv1.bias", 1, 1);
    path = bilinear(
        path, encoded.patch_height * 14, encoded.patch_width * 14);
    path = conv(
        model, path,
        "head.scratch.output_conv2.0.weight",
        "head.scratch.output_conv2.0.bias", 1, 1);
    relu(path);
    path = conv(
        model, path,
        "head.scratch.output_conv2.2.weight",
        "head.scratch.output_conv2.2.bias", 1, 0);
    relu(path);
    return std::move(path.data);
}

}  // namespace vda_native
