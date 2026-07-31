#include "temporal_gpu.h"
#include "model_config.h"

#include "temporal_attention_spv.h"
#include "temporal_attention_stream_spv.h"
#include "temporal_geglu_spv.h"
#include "temporal_group_norm_spv.h"
#include "temporal_output_spv.h"
#include "temporal_position_spv.h"
#include "temporal_transpose_spv.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vda_native {
namespace {

std::uint32_t divide_up(
    std::uint32_t value,
    std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::uint64_t feature_count(const FeatureMap& feature) {
    return std::uint64_t(feature.frames) * feature.channels *
        feature.height * feature.width;
}

}  // namespace

TemporalGpu::TemporalGpu(
    VulkanContext& context,
    GpuModel& weights,
    VulkanOperators& operators,
    const ModelConfig& config)
    : context_(context),
      weights_(weights),
      operators_(operators),
      zero_bias_(context.create_device_buffer(
          std::max({config.features, config.project_channels[2],
                    config.project_channels[3]}) * sizeof(float))),
      group_norm_(context.create_pipeline(
          vda_temporal_group_norm_spv,
          vda_temporal_group_norm_spv_size, 4, 16)),
      transpose_(context.create_pipeline(
          vda_temporal_transpose_spv,
          vda_temporal_transpose_spv_size, 2, 16)),
      position_(context.create_pipeline(
          vda_temporal_position_spv,
          vda_temporal_position_spv_size, 3, 12)),
      attention_(context.create_pipeline(
          vda_temporal_attention_spv,
          vda_temporal_attention_spv_size, 4, 16)),
      attention_stream_(context.create_pipeline(
          vda_temporal_attention_stream_spv,
          vda_temporal_attention_stream_spv_size, 4, 20)),
      geglu_(context.create_pipeline(
          vda_temporal_geglu_spv,
          vda_temporal_geglu_spv_size, 2, 8)),
      output_(context.create_pipeline(
          vda_temporal_output_spv,
          vda_temporal_output_spv_size, 3, 12)) {
    const std::vector<float> zero(
        std::max({config.features, config.project_channels[2],
                  config.project_channels[3]}),
        0.0f);
    context_.upload(
        zero_bias_, zero.data(), zero.size() * sizeof(float));
    group_norm_.set_debug_name("temporal_group_norm");
    transpose_.set_debug_name("temporal_transpose");
    position_.set_debug_name("temporal_position");
    attention_.set_debug_name("temporal_attention");
    attention_stream_.set_debug_name("temporal_attention_stream");
    geglu_.set_debug_name("temporal_geglu");
    output_.set_debug_name("temporal_output");
}

const VulkanBuffer& TemporalGpu::weight(
    const std::string& name) const {
    return weights_.tensor(name).buffer;
}

std::string TemporalGpu::prefix(
    std::uint32_t module_index) const {
    return "head.motion_modules." +
        std::to_string(module_index) +
        ".temporal_transformer.";
}

FeatureMap TemporalGpu::forward(
    std::uint32_t module_index,
    FeatureMap&& input) {
    return forward_impl(
        module_index, std::move(input), nullptr, nullptr);
}

FeatureMap TemporalGpu::forward_stream(
    std::uint32_t module_index,
    FeatureMap&& input,
    const std::vector<const TemporalFrameCache*>& history,
    TemporalFrameCache& output_cache) {
    return forward_impl(
        module_index, std::move(input), &history, &output_cache);
}

FeatureMap TemporalGpu::forward_impl(
    std::uint32_t module_index,
    FeatureMap&& input,
    const std::vector<const TemporalFrameCache*>* history,
    TemporalFrameCache* output_cache) {
    const bool streaming = history != nullptr;
    if (module_index >= 4 || input.frames == 0 ||
        input.frames > 32 || input.channels == 0 ||
        input.channels % 32 != 0 ||
        input.width == 0 || input.height == 0 ||
        (streaming &&
         (input.frames != 1 ||
          (!history->empty() && history->size() != 31) ||
          output_cache == nullptr))) {
        throw std::invalid_argument(
            "invalid GPU temporal module input");
    }
    if (streaming) {
        for (const TemporalFrameCache* cache : *history) {
            if (!cache || cache->width != input.width ||
                cache->height != input.height ||
                cache->channels != input.channels) {
                throw std::invalid_argument(
                    "incompatible GPU temporal cache");
            }
        }
        output_cache->width = input.width;
        output_cache->height = input.height;
        output_cache->channels = input.channels;
    }
    const std::uint32_t frames = input.frames;
    const std::uint32_t channels = input.channels;
    const std::uint32_t spatial = input.width * input.height;
    const std::uint32_t rows = frames * spatial;
    const std::uint64_t count = feature_count(input);
    const VkDeviceSize bytes = count * sizeof(float);
    const std::string base = prefix(module_index);
    VulkanBuffer frame_spatial =
        context_.create_device_buffer(bytes);
    struct Shape {
        std::uint32_t frames;
        std::uint32_t channels;
        std::uint32_t spatial;
        std::uint32_t value;
    };
    const Shape group_shape{frames, channels, spatial, 32};
    context_.dispatch(
        group_norm_,
        {
            &frame_spatial,
            &input.buffer,
            &weight(base + "norm.weight"),
            &weight(base + "norm.bias"),
        },
        &group_shape,
        sizeof(group_shape),
        frames * 32);

    VulkanBuffer projected =
        context_.create_device_buffer(bytes);
    operators_.linear(
        projected,
        frame_spatial,
        weight(base + "proj_in.weight"),
        weight(base + "proj_in.bias"),
        rows,
        channels,
        channels,
        false);
    VulkanBuffer state = context_.create_device_buffer(bytes);
    const Shape to_sequence{frames, channels, spatial, 1};
    context_.dispatch(
        transpose_,
        {&state, &projected},
        &to_sequence,
        sizeof(to_sequence),
        divide_up(static_cast<std::uint32_t>(count), 256));

    const std::string block =
        base + "transformer_blocks.0.";
    VulkanBuffer normalized =
        context_.create_device_buffer(bytes);
    const bool cached_stream =
        streaming && !history->empty();
    const std::uint32_t attention_frames =
        cached_stream ? 32 : frames;
    const std::uint32_t attention_rows =
        attention_frames * spatial;
    const std::uint64_t attention_count =
        std::uint64_t(attention_frames) * spatial * channels;
    const VkDeviceSize attention_bytes =
        attention_count * sizeof(float);
    VulkanBuffer combined =
        cached_stream
        ? context_.create_device_buffer(attention_bytes)
        : VulkanBuffer{};
    VulkanBuffer positioned =
        context_.create_device_buffer(attention_bytes);
    VulkanBuffer query =
        context_.create_device_buffer(attention_bytes);
    VulkanBuffer key =
        context_.create_device_buffer(attention_bytes);
    VulkanBuffer value =
        context_.create_device_buffer(attention_bytes);
    VulkanBuffer attended =
        context_.create_device_buffer(bytes);
    VulkanBuffer attention_output =
        context_.create_device_buffer(bytes);
    struct AttentionShape {
        std::uint32_t sequences;
        std::uint32_t frames;
        std::uint32_t channels;
        std::uint32_t heads;
    };
    const AttentionShape attention_shape{
        spatial, attention_frames, channels, 8};
    for (std::uint32_t attention = 0;
         attention < 2;
         ++attention) {
        const std::string norm =
            block + "norms." + std::to_string(attention);
        operators_.layer_norm(
            normalized,
            state,
            weight(norm + ".weight"),
            weight(norm + ".bias"),
            rows,
            channels,
            1.0e-5f);
        if (streaming) {
            output_cache->attention[attention] =
                context_.create_device_buffer(bytes);
            context_.copy(
                output_cache->attention[attention], 0,
                normalized, 0, bytes);
        }
        const std::string attention_base =
            block + "attention_blocks." +
            std::to_string(attention) + ".";
        const VulkanBuffer* attention_input = &normalized;
        if (cached_stream) {
            VulkanBuffer frame_order =
                context_.create_device_buffer(attention_bytes);
            for (std::size_t index = 0;
                 index < history->size(); ++index) {
                context_.copy(
                    frame_order,
                    static_cast<VkDeviceSize>(index) * bytes,
                    (*history)[index]->attention[attention],
                    0, bytes);
            }
            context_.copy(
                frame_order, 31 * bytes,
                normalized, 0, bytes);
            const Shape join_shape{
                32, channels, spatial, 1};
            context_.dispatch(
                transpose_,
                {&combined, &frame_order},
                &join_shape,
                sizeof(join_shape),
                divide_up(
                    static_cast<std::uint32_t>(attention_count),
                    256));
            attention_input = &combined;
        }
        const struct PositionShape {
            std::uint32_t sequences;
            std::uint32_t frames;
            std::uint32_t channels;
        } position_shape{
            spatial, attention_frames, channels};
        context_.dispatch(
            position_,
            {
                &positioned,
                attention_input,
                &weight(attention_base + "pos_encoder.pe"),
            },
            &position_shape,
            sizeof(position_shape),
            divide_up(
                static_cast<std::uint32_t>(attention_count), 256));
        operators_.linear(
            query, positioned,
            weight(attention_base + "to_q.weight"),
            zero_bias_, attention_rows, channels, channels, false);
        operators_.linear(
            key, positioned,
            weight(attention_base + "to_k.weight"),
            zero_bias_, attention_rows, channels, channels, false);
        operators_.linear(
            value, positioned,
            weight(attention_base + "to_v.weight"),
            zero_bias_, attention_rows, channels, channels, false);
        if (cached_stream) {
            const struct StreamAttentionShape {
                std::uint32_t sequences;
                std::uint32_t frames;
                std::uint32_t channels;
                std::uint32_t heads;
                std::uint32_t query_frame;
            } stream_shape{
                spatial, 32, channels, 8, 31};
            context_.dispatch(
                attention_stream_,
                {&attended, &query, &key, &value},
                &stream_shape,
                sizeof(stream_shape),
                divide_up(channels, 8), 1, spatial);
        } else {
            context_.dispatch(
                attention_,
                {&attended, &query, &key, &value},
                &attention_shape,
                sizeof(attention_shape),
                divide_up(channels, 8),
                divide_up(frames, 8),
                spatial);
        }
        operators_.linear(
            attention_output,
            attended,
            weight(attention_base + "to_out.0.weight"),
            weight(attention_base + "to_out.0.bias"),
            rows,
            channels,
            channels,
            false);
        operators_.add(
            state, state, attention_output,
            static_cast<std::uint32_t>(count));
    }

    operators_.layer_norm(
        normalized,
        state,
        weight(block + "ff_norm.weight"),
        weight(block + "ff_norm.bias"),
        rows,
        channels,
        1.0e-5f);
    VulkanBuffer expanded = context_.create_device_buffer(
        count * 8 * sizeof(float));
    operators_.linear(
        expanded,
        normalized,
        weight(block + "ff.net.0.proj.weight"),
        weight(block + "ff.net.0.proj.bias"),
        rows,
        channels,
        channels * 8,
        false);
    VulkanBuffer gated = context_.create_device_buffer(
        count * 4 * sizeof(float));
    const struct GegluShape {
        std::uint32_t rows;
        std::uint32_t inner;
    } geglu_shape{rows, channels * 4};
    context_.dispatch(
        geglu_,
        {&gated, &expanded},
        &geglu_shape,
        sizeof(geglu_shape),
        divide_up(rows * channels * 4, 256));
    VulkanBuffer fed = context_.create_device_buffer(bytes);
    operators_.linear(
        fed,
        gated,
        weight(block + "ff.net.2.weight"),
        weight(block + "ff.net.2.bias"),
        rows,
        channels * 4,
        channels,
        false);
    operators_.add(
        state, state, fed,
        static_cast<std::uint32_t>(count));

    VulkanBuffer frame_order =
        context_.create_device_buffer(bytes);
    const Shape to_frame{frames, channels, spatial, 0};
    context_.dispatch(
        transpose_,
        {&frame_order, &state},
        &to_frame,
        sizeof(to_frame),
        divide_up(static_cast<std::uint32_t>(count), 256));
    VulkanBuffer projected_out =
        context_.create_device_buffer(bytes);
    operators_.linear(
        projected_out,
        frame_order,
        weight(base + "proj_out.weight"),
        weight(base + "proj_out.bias"),
        rows,
        channels,
        channels,
        false);
    FeatureMap output{
        context_.create_device_buffer(bytes),
        input.width,
        input.height,
        input.channels,
        input.frames,
    };
    const struct OutputShape {
        std::uint32_t frames;
        std::uint32_t channels;
        std::uint32_t spatial;
    } output_shape{frames, channels, spatial};
    context_.dispatch(
        output_,
        {&output.buffer, &projected_out, &input.buffer},
        &output_shape,
        sizeof(output_shape),
        divide_up(static_cast<std::uint32_t>(count), 256));
    return output;
}

}  // namespace vda_native
