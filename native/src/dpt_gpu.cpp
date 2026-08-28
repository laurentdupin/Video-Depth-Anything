#include "dpt_gpu.h"
#include "temporal_gpu.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace vda_native {
namespace {

std::uint64_t elements(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t channels) {
    return std::uint64_t(width) * height * channels;
}

const VulkanBuffer& weight(
    const GpuModel& model,
    const std::string& name) {
    return model.tensor(name).buffer;
}

}  // namespace

VdaGpuDpt::VdaGpuDpt(
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators,
    const ModelConfig& config)
    : context_(context),
      weights_(weights),
      operators_(operators),
      zero_bias_(context.create_device_buffer(sizeof(float))),
      temporal_(std::make_unique<TemporalGpu>(
          context, weights, operators, config)) {
    const float zero = 0.0f;
    context_.upload(zero_bias_, &zero, sizeof(zero));
    embedding_ = config.embedding;
    features_ = config.features;
    std::copy(
        config.project_channels.begin(), config.project_channels.end(),
        project_channels_);
    convolution_half_weight_ = inferbridge::native::select_fp16_weights(false);
    convolution_block_selected_ = true;
}

VdaGpuDpt::~VdaGpuDpt() = default;

void VdaGpuDpt::select_convolution_block() {
    constexpr std::uint32_t side = 16;
    const VkDeviceSize bytes =
        elements(side, side, features_) * sizeof(float);
    VulkanBuffer input = context_.create_device_buffer(bytes);
    VulkanBuffer output = context_.create_device_buffer(bytes);
    const GpuTensor& convolution_weight = weights_.tensor(
        "head.scratch.refinenet4.resConfUnit2.conv1.weight");
    const VulkanBuffer& convolution_bias = weight(
        weights_,
        "head.scratch.refinenet4.resConfUnit2.conv1.bias");
    const auto run = [&](bool block8, bool half_weight) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            for (int repetition = 0; repetition < 3; ++repetition) {
                operators_.conv2d(
                    output,
                    input,
                    half_weight
                        ? convolution_weight.half_buffer
                        : convolution_weight.buffer,
                    convolution_bias,
                    side,
                    side,
                    features_,
                    features_,
                    3,
                    1,
                    1,
                    true,
                    block8,
                    half_weight);
            }
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct Candidate {
        bool block8;
        bool half_weight;
        std::array<double, 3> samples{};
    };
    std::array<Candidate, 4> candidates{{
        {false, false, {}},
        {true, false, {}},
        {false, true, {}},
        {true, true, {}},
    }};
    for (Candidate& candidate : candidates) {
        run(candidate.block8, candidate.half_weight);
    }
    for (std::size_t sample = 0;
         sample < candidates[0].samples.size();
         ++sample) {
        if ((sample & 1u) == 0) {
            for (Candidate& candidate : candidates) {
                candidate.samples[sample] =
                    run(candidate.block8, candidate.half_weight);
            }
        } else {
            for (auto candidate = candidates.rbegin();
                 candidate != candidates.rend();
                 ++candidate) {
                candidate->samples[sample] =
                    run(candidate->block8, candidate->half_weight);
            }
        }
    }
    Candidate* best_fp32 = nullptr;
    Candidate* best_half = nullptr;
    double best_fp32_time = 0.0;
    double best_half_time = 0.0;
    for (Candidate& candidate : candidates) {
        std::sort(candidate.samples.begin(), candidate.samples.end());
        const double median =
            candidate.samples[candidate.samples.size() / 2];
        Candidate*& best =
            candidate.half_weight ? best_half : best_fp32;
        double& best_time =
            candidate.half_weight ? best_half_time : best_fp32_time;
        if (best == nullptr || median < best_time) {
            best = &candidate;
            best_time = median;
        }
    }
    Candidate* best = inferbridge::native::select_fp16_weights(
        features_ >= 256 && best_half_time < best_fp32_time * 0.96)
        ? best_half : best_fp32;
    convolution_block8_ = best->block8;
    convolution_half_weight_ = best->half_weight;
    weights_.retain_dpt_precision(convolution_half_weight_);
    convolution_block_selected_ = true;
}

const VulkanBuffer& VdaGpuDpt::selected_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    return convolution_half_weight_
        ? tensor.half_buffer
        : tensor.buffer;
}

FeatureMap VdaGpuDpt::conv(
    FeatureMap&& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias) {
    const std::uint32_t output_width =
        (input.width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input.height + 2 * padding - kernel) / stride + 1;
    FeatureMap output{
        context_.create_device_buffer(
            std::uint64_t(input.frames) *
            elements(output_width, output_height, output_channels) *
            sizeof(float)),
        output_width,
        output_height,
        output_channels,
        input.frames,
    };
    const GpuTensor& convolution_weight =
        weights_.tensor(weight_name);
    operators_.conv2d(
        output.buffer,
        input.buffer,
        convolution_half_weight_
            ? convolution_weight.half_buffer
            : convolution_weight.buffer,
        has_bias ? weight(weights_, bias_name) : zero_bias_,
        input.width,
        input.height,
        input.channels,
        output_channels,
        kernel,
        stride,
        padding,
        has_bias,
        convolution_block8_,
        convolution_half_weight_,
        input.frames);
    return output;
}

FeatureMap VdaGpuDpt::residual_unit(
    FeatureMap&& input,
    const std::string& prefix) {
    const std::uint32_t count =
        static_cast<std::uint32_t>(
            elements(input.width, input.height, input.channels));
    const std::uint64_t batch_count =
        std::uint64_t(input.frames) * count;
    FeatureMap activated{
        context_.create_device_buffer(batch_count * sizeof(float)),
        input.width,
        input.height,
        input.channels,
        input.frames,
    };
    operators_.relu(
        activated.buffer, input.buffer,
        static_cast<std::uint32_t>(batch_count));
    FeatureMap first = conv(
        std::move(activated),
        prefix + ".conv1.weight",
        prefix + ".conv1.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.relu(
        first.buffer, first.buffer,
        static_cast<std::uint32_t>(batch_count));
    FeatureMap second = conv(
        std::move(first),
        prefix + ".conv2.weight",
        prefix + ".conv2.bias",
        input.channels,
        3,
        1,
        1,
        true);
    operators_.add(
        second.buffer, second.buffer, input.buffer,
        static_cast<std::uint32_t>(batch_count));
    return second;
}

FeatureMap VdaGpuDpt::fusion(
    FeatureMap&& path,
    FeatureMap&& skip,
    const std::string& prefix,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (skip.buffer.handle() != VK_NULL_HANDLE) {
        FeatureMap processed_skip =
            residual_unit(std::move(skip), prefix + ".resConfUnit1");
        const std::uint32_t count =
            static_cast<std::uint32_t>(
                std::uint64_t(path.frames) *
                elements(path.width, path.height, path.channels));
        operators_.add(
            path.buffer, path.buffer, processed_skip.buffer, count);
    }
    path = residual_unit(
        std::move(path), prefix + ".resConfUnit2");
    FeatureMap resized{
        context_.create_device_buffer(
            std::uint64_t(path.frames) *
            elements(output_width, output_height, features_) *
            sizeof(float)),
        output_width,
        output_height,
        features_,
        path.frames,
    };
    operators_.bilinear_align_true(
        resized.buffer,
        path.buffer,
        path.width,
        path.height,
        output_width,
        output_height,
        features_,
        path.frames);
    return conv(
        std::move(resized),
        prefix + ".out_conv.weight",
        prefix + ".out_conv.bias",
        features_,
        1,
        1,
        0,
        true);
}

FeatureMap VdaGpuDpt::forward(EncoderOutput&& encoded) {
    return forward_impl(std::move(encoded), nullptr, nullptr);
}

FeatureMap VdaGpuDpt::forward_stream(
    EncoderOutput&& encoded,
    const std::vector<const TemporalFrameCache*> history[4],
    TemporalFrameCache* output_cache[4]) {
    return forward_impl(
        std::move(encoded), history, output_cache);
}

FeatureMap VdaGpuDpt::forward_impl(
    EncoderOutput&& encoded,
    const std::vector<const TemporalFrameCache*>* history,
    TemporalFrameCache* const* output_cache) {
    if (encoded.features.size() != 4 ||
        encoded.embedding != embedding_ ||
        encoded.frames == 0 ||
        (history &&
         (!output_cache || !output_cache[0] || !output_cache[1] ||
          !output_cache[2] || !output_cache[3]))) {
        throw std::invalid_argument("invalid DPT encoder output");
    }
    if (!convolution_block_selected_) {
        select_convolution_block();
    }
    FeatureMap layers[4];
    const auto run_projections = [&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            context_.batch([&] {
            FeatureMap projected{
                context_.create_device_buffer(
                    elements(
                        encoded.patch_width,
                        encoded.patch_height,
                        project_channels_[index]) *
                    encoded.frames *
                    sizeof(float)),
                encoded.patch_width,
                encoded.patch_height,
                project_channels_[index],
                encoded.frames,
            };
            const std::string prefix =
                "head.projects." + std::to_string(index);
            operators_.project_tokens(
                projected.buffer,
                encoded.features[index],
                selected_weight(prefix + ".weight"),
                weight(weights_, prefix + ".bias"),
                encoded.patch_width,
                encoded.patch_height,
                embedding_,
                project_channels_[index],
                convolution_half_weight_,
                encoded.frames);
            if (index < 2) {
                const std::uint32_t kernel = index == 0 ? 4 : 2;
                FeatureMap resized{
                    context_.create_device_buffer(
                        elements(
                            projected.width * kernel,
                            projected.height * kernel,
                            projected.channels) *
                        projected.frames *
                        sizeof(float)),
                    projected.width * kernel,
                    projected.height * kernel,
                    projected.channels,
                    projected.frames,
                };
                const std::string resize =
                    "head.resize_layers." + std::to_string(index);
                operators_.conv_transpose_nonoverlap(
                    resized.buffer,
                    projected.buffer,
                    selected_weight(resize + ".weight"),
                    weight(weights_, resize + ".bias"),
                    projected.width,
                    projected.height,
                    projected.channels,
                    projected.channels,
                    kernel,
                    convolution_half_weight_,
                    projected.frames);
                layers[index] = std::move(resized);
            } else if (index == 2) {
                layers[index] = std::move(projected);
            } else {
                layers[index] = conv(
                    std::move(projected),
                    "head.resize_layers.3.weight",
                    "head.resize_layers.3.bias",
                    project_channels_[3],
                    3,
                    2,
                    1,
                    true);
            }
            });
        }
    };
    if (std::uint64_t(encoded.frames) * encoded.tokens > 2000) {
        run_projections();
    } else {
        context_.batch(run_projections);
    }
    context_.batch([&] {
        if (history) {
            layers[2] = temporal_->forward_stream(
                0, std::move(layers[2]),
                history[0], *output_cache[0]);
            layers[3] = temporal_->forward_stream(
                1, std::move(layers[3]),
                history[1], *output_cache[1]);
        } else {
            layers[2] =
                temporal_->forward(0, std::move(layers[2]));
            layers[3] =
                temporal_->forward(1, std::move(layers[3]));
        }
    });

    FeatureMap refined[4];
    FeatureMap path;
    FeatureMap depth;
    const auto run_head = [&] {
    context_.batch([&] {
        for (std::uint32_t index = 0; index < 4; ++index) {
            const std::string prefix =
                "head.scratch.layer" + std::to_string(index + 1) +
                "_rn.weight";
            refined[index] = conv(
                std::move(layers[index]),
                prefix,
                "",
                features_,
                3,
                1,
                1,
                false);
        }
    });

    context_.batch([&] {
        path = fusion(
            std::move(refined[3]),
            FeatureMap{},
            "head.scratch.refinenet4",
            refined[2].width,
            refined[2].height);
        path = history
            ? temporal_->forward_stream(
                2, std::move(path),
                history[2], *output_cache[2])
            : temporal_->forward(2, std::move(path));
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[2]),
            "head.scratch.refinenet3",
            refined[1].width,
            refined[1].height);
        path = history
            ? temporal_->forward_stream(
                3, std::move(path),
                history[3], *output_cache[3])
            : temporal_->forward(3, std::move(path));
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[1]),
            "head.scratch.refinenet2",
            refined[0].width,
            refined[0].height);
    });
    context_.batch([&] {
        path = fusion(
            std::move(path),
            std::move(refined[0]),
            "head.scratch.refinenet1",
            refined[0].width * 2,
            refined[0].height * 2);
    });

    context_.batch([&] {
        path = conv(
            std::move(path),
            "head.scratch.output_conv1.weight",
            "head.scratch.output_conv1.bias",
            features_ / 2,
            3,
            1,
            1,
            true);
        FeatureMap full{
            context_.create_device_buffer(
                elements(
                    encoded.patch_width * 14,
                    encoded.patch_height * 14,
                    features_ / 2) *
                encoded.frames *
                sizeof(float)),
            encoded.patch_width * 14,
            encoded.patch_height * 14,
            features_ / 2,
            encoded.frames,
        };
        operators_.bilinear_align_true(
            full.buffer,
            path.buffer,
            path.width,
            path.height,
            full.width,
            full.height,
            full.channels,
            full.frames);
        full = conv(
            std::move(full),
            "head.scratch.output_conv2.0.weight",
            "head.scratch.output_conv2.0.bias",
            32,
            3,
            1,
            1,
            true);
        operators_.relu(
            full.buffer,
            full.buffer,
            static_cast<std::uint32_t>(
                std::uint64_t(full.frames) *
                elements(full.width, full.height, full.channels)));
        depth = conv(
            std::move(full),
            "head.scratch.output_conv2.2.weight",
            "head.scratch.output_conv2.2.bias",
            1,
            1,
            1,
            0,
            true);
        operators_.relu(
            depth.buffer,
            depth.buffer,
            depth.frames * depth.width * depth.height);
    });
    };
    if (std::uint64_t(encoded.frames) * encoded.tokens > 2000) {
        run_head();
    } else {
        context_.batch(run_head);
    }
    return depth;
}

}  // namespace vda_native
