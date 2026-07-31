#include "encoder_gpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace vda_native {
namespace {

const VulkanBuffer& buffer(
    const GpuModel& weights,
    const std::string& name) {
    return weights.tensor(name).buffer;
}

std::string block_name(std::uint32_t block, const char* suffix) {
    return "pretrained.blocks." + std::to_string(block) + suffix;
}

}  // namespace

VdaGpuEncoder::VdaGpuEncoder(
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators,
    const ModelConfig& config)
    : context_(context),
      weights_(weights),
      operators_(operators) {
    embedding_ = config.embedding;
    heads_ = config.heads;
    blocks_ = config.blocks;
    std::copy(config.captures.begin(), config.captures.end(), capture_);
    linear_tile_selected_ = true;
    if (weights_.tensor("pretrained.cls_token").elements != embedding_ ||
        weights_.tensor("pretrained.pos_embed").elements !=
            std::uint64_t(1370) * embedding_) {
        throw std::runtime_error("encoder tensor dimensions do not match");
    }
}

void VdaGpuEncoder::select_linear_tile() {
    constexpr std::uint32_t rows = 64;
    const VkDeviceSize work_bytes =
        std::uint64_t(rows) * embedding_ * 4 * sizeof(float);
    VulkanBuffer left = context_.create_device_buffer(work_bytes);
    VulkanBuffer right = context_.create_device_buffer(work_bytes);
    const auto selected_weight = [&](
        const std::string& name,
        bool half_weight) -> const VulkanBuffer& {
        const GpuTensor& tensor = weights_.tensor(name);
        return half_weight ? tensor.half_buffer : tensor.buffer;
    };
    const auto run = [&](bool block16, bool half_weight) {
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            operators_.linear(
                right,
                left,
                selected_weight(
                    block_name(0, ".attn.qkv.weight"), half_weight),
                buffer(weights_, block_name(0, ".attn.qkv.bias")),
                rows,
                embedding_,
                embedding_ * 3,
                false,
                block16,
                half_weight);
            operators_.linear(
                left,
                right,
                selected_weight(
                    block_name(0, ".attn.proj.weight"), half_weight),
                buffer(weights_, block_name(0, ".attn.proj.bias")),
                rows,
                embedding_,
                embedding_,
                false,
                block16,
                half_weight);
            operators_.linear(
                right,
                left,
                selected_weight(
                    block_name(0, ".mlp.fc1.weight"), half_weight),
                buffer(weights_, block_name(0, ".mlp.fc1.bias")),
                rows,
                embedding_,
                embedding_ * 4,
                true,
                block16,
                half_weight);
            operators_.linear(
                left,
                right,
                selected_weight(
                    block_name(0, ".mlp.fc2.weight"), half_weight),
                buffer(weights_, block_name(0, ".mlp.fc2.bias")),
                rows,
                embedding_ * 4,
                embedding_,
                false,
                block16,
                half_weight);
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    struct Candidate {
        bool block16;
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
        run(candidate.block16, candidate.half_weight);
    }
    for (std::size_t sample = 0;
         sample < candidates[0].samples.size();
         ++sample) {
        if ((sample & 1u) == 0) {
            for (Candidate& candidate : candidates) {
                candidate.samples[sample] =
                    run(candidate.block16, candidate.half_weight);
            }
        } else {
            for (auto candidate = candidates.rbegin();
                 candidate != candidates.rend();
                 ++candidate) {
                candidate->samples[sample] =
                    run(candidate->block16, candidate->half_weight);
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
    Candidate* best =
        best_half_time < best_fp32_time * 0.96
        ? best_half
        : best_fp32;
    linear_block16_ = best->block16;
    linear_half_weight_ = best->half_weight;
    weights_.retain_transformer_precision(linear_half_weight_);
    linear_tile_selected_ = true;
}

const VulkanBuffer& VdaGpuEncoder::linear_weight(
    const std::string& name) const {
    const GpuTensor& tensor = weights_.tensor(name);
    return linear_half_weight_ ? tensor.half_buffer : tensor.buffer;
}

bool VdaGpuEncoder::select_half_attention(
    const VulkanBuffer& current,
    VulkanBuffer& normalized,
    VulkanBuffer& qkv,
    VulkanBuffer& attention,
    std::uint32_t tokens) {
    context_.batch([&] {
        operators_.layer_norm(
            normalized,
            current,
            buffer(weights_, block_name(0, ".norm1.weight")),
            buffer(weights_, block_name(0, ".norm1.bias")),
            tokens,
            embedding_,
            1.0e-6f);
        operators_.linear(
            qkv,
            normalized,
            linear_weight(block_name(0, ".attn.qkv.weight")),
            buffer(weights_, block_name(0, ".attn.qkv.bias")),
            tokens,
            embedding_,
            embedding_ * 3,
            false,
            linear_block16_,
            linear_half_weight_);
    });

    VulkanBuffer fp32_scratch = context_.create_device_buffer(
        std::uint64_t(heads_) * tokens * tokens * sizeof(float));
    VulkanBuffer half_scratch = context_.create_device_buffer(
        std::uint64_t(heads_) * tokens *
        ((std::uint64_t(tokens) + 1) / 2) *
        sizeof(std::uint32_t));
    const std::uint32_t repetitions =
        tokens < 256 ? 8 : (tokens < 768 ? 4 : 2);
    const auto run = [&](bool half_scores) {
        VulkanBuffer& scratch =
            half_scores ? half_scratch : fp32_scratch;
        const auto start = std::chrono::steady_clock::now();
        context_.batch([&] {
            for (std::uint32_t repetition = 0;
                 repetition < repetitions;
                 ++repetition) {
                operators_.attention_head64(
                    attention,
                    qkv,
                    tokens,
                    heads_,
                    &scratch,
                    half_scores);
            }
        });
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
    };
    run(false);
    run(true);
    std::array<double, 3> fp32_samples{};
    std::array<double, 3> half_samples{};
    for (std::size_t sample = 0; sample < fp32_samples.size(); ++sample) {
        if ((sample & 1u) == 0) {
            fp32_samples[sample] = run(false);
            half_samples[sample] = run(true);
        } else {
            half_samples[sample] = run(true);
            fp32_samples[sample] = run(false);
        }
    }
    std::sort(fp32_samples.begin(), fp32_samples.end());
    std::sort(half_samples.begin(), half_samples.end());
    return half_samples[half_samples.size() / 2] <
        fp32_samples[fp32_samples.size() / 2] * 0.99;
}

EncoderOutput VdaGpuEncoder::forward_batch(
    const VulkanBuffer& image,
    std::uint32_t frames,
    std::uint32_t width,
    std::uint32_t height) {
    if (frames == 0 || width == 0 || height == 0 ||
        width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    if (!linear_tile_selected_) {
        select_linear_tile();
    }
    const std::uint64_t token_elements =
        std::uint64_t(frames) * tokens * embedding_;
    const VkDeviceSize token_bytes = token_elements * sizeof(float);

    VulkanBuffer current = context_.create_device_buffer(token_bytes);
    VulkanBuffer next = context_.create_device_buffer(token_bytes);
    VulkanBuffer normalized = context_.create_device_buffer(token_bytes);
    VulkanBuffer query = context_.create_device_buffer(token_bytes);
    VulkanBuffer attention = context_.create_device_buffer(token_bytes);
    VulkanBuffer qkv =
        context_.create_device_buffer(token_bytes * 3);
    VulkanBuffer hidden =
        context_.create_device_buffer(token_bytes * 4);
    context_.batch([&] {
        operators_.prepare_tokens(
            current,
            image,
            buffer(weights_, "pretrained.patch_embed.proj.weight"),
            buffer(weights_, "pretrained.patch_embed.proj.bias"),
            buffer(weights_, "pretrained.cls_token"),
            buffer(weights_, "pretrained.pos_embed"),
            width,
            height,
            embedding_,
            frames);
    });
    const bool half_attention = false;
    const VkDeviceSize attention_score_bytes = half_attention
        ? std::uint64_t(frames) * heads_ * tokens *
            ((std::uint64_t(tokens) + 1) / 2) *
            sizeof(std::uint32_t)
        : std::uint64_t(frames) * heads_ *
            tokens * tokens * sizeof(float);
    VulkanBuffer attention_scores =
        context_.create_device_buffer(attention_score_bytes);

    EncoderOutput result;
    result.features.reserve(4);
    result.frames = frames;
    result.patch_width = patch_width;
    result.patch_height = patch_height;
    result.tokens = tokens;
    result.embedding = embedding_;
    std::uint32_t capture_index = 0;

    const std::uint32_t blocks_per_submission =
        std::uint64_t(frames) * tokens > 2000 ? 4 : blocks_;
    for (std::uint32_t block_begin = 0;
         block_begin < blocks_;
         block_begin += blocks_per_submission) {
        const std::uint32_t block_end =
            std::min(blocks_, block_begin + blocks_per_submission);
        context_.batch([&, block_begin, block_end] {
        for (std::uint32_t block = block_begin;
             block < block_end;
             ++block) {
            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm1.weight")),
                buffer(weights_, block_name(block, ".norm1.bias")),
                frames * tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                qkv,
                normalized,
                linear_weight(block_name(block, ".attn.qkv.weight")),
                buffer(weights_, block_name(block, ".attn.qkv.bias")),
                frames * tokens,
                embedding_,
                embedding_ * 3,
                false,
                linear_block16_,
                linear_half_weight_);
            operators_.attention_head64(
                attention,
                qkv,
                tokens,
                heads_,
                &attention_scores,
                half_attention,
                frames);
            operators_.linear(
                query,
                attention,
                linear_weight(block_name(block, ".attn.proj.weight")),
                buffer(weights_, block_name(block, ".attn.proj.bias")),
                frames * tokens,
                embedding_,
                embedding_,
                false,
                linear_block16_,
                linear_half_weight_);
            operators_.add_scaled(
                next,
                current,
                query,
                buffer(weights_, block_name(block, ".ls1.gamma")),
                static_cast<std::uint32_t>(token_elements),
                embedding_);
            std::swap(current, next);

            operators_.layer_norm(
                normalized,
                current,
                buffer(weights_, block_name(block, ".norm2.weight")),
                buffer(weights_, block_name(block, ".norm2.bias")),
                frames * tokens,
                embedding_,
                1.0e-6f);
            operators_.linear(
                hidden,
                normalized,
                linear_weight(block_name(block, ".mlp.fc1.weight")),
                buffer(weights_, block_name(block, ".mlp.fc1.bias")),
                frames * tokens,
                embedding_,
                embedding_ * 4,
                true,
                linear_block16_,
                linear_half_weight_);
            operators_.linear(
                query,
                hidden,
                linear_weight(block_name(block, ".mlp.fc2.weight")),
                buffer(weights_, block_name(block, ".mlp.fc2.bias")),
                frames * tokens,
                embedding_ * 4,
                embedding_,
                false,
                linear_block16_,
                linear_half_weight_);
            operators_.add_scaled(
                next,
                current,
                query,
                buffer(weights_, block_name(block, ".ls2.gamma")),
                static_cast<std::uint32_t>(token_elements),
                embedding_);
            std::swap(current, next);

            if (capture_index < 4 && block == capture_[capture_index]) {
                VulkanBuffer feature =
                    context_.create_device_buffer(token_bytes);
                operators_.layer_norm(
                    feature,
                    current,
                    buffer(weights_, "pretrained.norm.weight"),
                    buffer(weights_, "pretrained.norm.bias"),
                    frames * tokens,
                    embedding_,
                    1.0e-6f);
                result.features.push_back(std::move(feature));
                ++capture_index;
            }
        }
        });
    }
    if (result.features.size() != 4) {
        throw std::runtime_error("encoder did not produce four features");
    }
    return result;
}

EncoderOutput VdaGpuEncoder::forward(
    const VulkanBuffer& image,
    std::uint32_t frames,
    std::uint32_t width,
    std::uint32_t height) {
    if (frames == 0 || width == 0 || height == 0 ||
        width % 14 != 0 || height % 14 != 0) {
        throw std::invalid_argument(
            "encoder dimensions must be positive multiples of 14");
    }
    const std::uint32_t patch_width = width / 14;
    const std::uint32_t patch_height = height / 14;
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    const std::uint64_t score_bytes_per_frame =
        std::uint64_t(heads_) * tokens * tokens * sizeof(float);
    constexpr std::uint64_t score_budget =
        std::uint64_t(128) * 1024 * 1024;
    std::uint32_t chunk_frames = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(
            1, score_budget /
                std::max<std::uint64_t>(score_bytes_per_frame, 1)));
    chunk_frames = std::min(chunk_frames, 8u);
    if (tokens > 500) {
        chunk_frames = std::min(chunk_frames, 2u);
    }
    chunk_frames = std::min(chunk_frames, frames);
    if (chunk_frames == frames) {
        return forward_batch(image, frames, width, height);
    }

    EncoderOutput combined;
    combined.frames = frames;
    combined.patch_width = patch_width;
    combined.patch_height = patch_height;
    combined.tokens = tokens;
    combined.embedding = embedding_;
    const VkDeviceSize feature_bytes_per_frame =
        std::uint64_t(tokens) * embedding_ * sizeof(float);
    for (std::uint32_t level = 0; level < 4; ++level) {
        combined.features.push_back(
            context_.create_device_buffer(
                feature_bytes_per_frame * frames));
    }
    const VkDeviceSize image_bytes_per_frame =
        std::uint64_t(3) * width * height * sizeof(float);
    for (std::uint32_t frame_begin = 0;
         frame_begin < frames;
         frame_begin += chunk_frames) {
        const std::uint32_t count =
            std::min(chunk_frames, frames - frame_begin);
        VulkanBuffer chunk = context_.create_device_buffer(
            image_bytes_per_frame * count);
        context_.copy(
            chunk, 0, image,
            image_bytes_per_frame * frame_begin,
            image_bytes_per_frame * count);
        EncoderOutput encoded =
            forward_batch(chunk, count, width, height);
        for (std::uint32_t level = 0; level < 4; ++level) {
            context_.copy(
                combined.features[level],
                feature_bytes_per_frame * frame_begin,
                encoded.features[level],
                0,
                feature_bytes_per_frame * count);
        }
    }
    return combined;
}

}  // namespace vda_native
