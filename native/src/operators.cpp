#include "operators.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv2d_pointwise_gemm_spv.h"
#include "conv2d_pointwise_gemm_half_spv.h"
#include "conv2d8_tiled16x8_spv.h"
#include "conv2d8_tiled16x8_half_spv.h"
#include "conv2d_implicit_gemm_spv.h"
#include "conv2d_implicit_gemm_half_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_vec4_spv.h"
#include "linear_vec4_half_spv.h"
#include "linear_vec8_spv.h"
#include "linear_vec8_half_spv.h"
#include "linear_vec16_spv.h"
#include "linear_vec16_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "quantize_rows_int8_spv.h"
#include "inferbridge/native_harness_precision.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace vda_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          vda_linear_spv, vda_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          vda_linear16_spv, vda_linear16_spv_size, 4, 12)),
      linear_half_(context.create_pipeline(
          vda_linear_half_spv, vda_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          vda_linear16_half_spv,
          vda_linear16_half_spv_size,
          4,
          12)),
      linear_vec4_(context.create_pipeline(
          vda_linear_vec4_spv,
          vda_linear_vec4_spv_size,
          4,
          12)),
      linear_vec4_half_(context.create_pipeline(
          vda_linear_vec4_half_spv,
          vda_linear_vec4_half_spv_size,
          4,
          12)),
      linear_vec8_(context.create_pipeline(
          vda_linear_vec8_spv,
          vda_linear_vec8_spv_size,
          4,
          12)),
      linear_vec8_half_(context.create_pipeline(
          vda_linear_vec8_half_spv,
          vda_linear_vec8_half_spv_size,
          4,
          12)),
      linear_vec16_(context.create_pipeline(
          vda_linear_vec16_spv,
          vda_linear_vec16_spv_size,
          4,
          12)),
      linear_vec16_half_(context.create_pipeline(
          vda_linear_vec16_half_spv,
          vda_linear_vec16_half_spv_size,
          4,
          12)),
      quantize_rows_int8_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(vda_quantize_rows_int8_spv,
                vda_quantize_rows_int8_spv_size,
              {
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            },
                            {
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                            },
                            4)
          : VulkanPipeline{}),
      linear_int8_tiled_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(vda_linear_int8_tiled_spv,
                vda_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      gelu_(context.create_pipeline(
          vda_gelu_spv, vda_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          vda_layer_norm_spv, vda_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          vda_add_scaled_spv, vda_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          vda_bmm_spv, vda_bmm_spv_size, 3, 44)),
      bmm_score_half_(context.create_pipeline(
          vda_bmm_score_half_spv,
          vda_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          vda_bmm_value_half_spv,
          vda_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          vda_softmax_lastdim_spv,
          vda_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          vda_softmax_lastdim_half_spv,
          vda_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          vda_prepare_tokens_spv,
          vda_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          vda_position_bicubic_spv,
          vda_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          vda_add_position_spv,
          vda_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          vda_add_spv, vda_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          vda_project_tokens_spv,
          vda_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          vda_project_tokens_half_spv,
          vda_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          vda_conv2d_spv, vda_conv2d_spv_size, 4, 48)),
      conv2d8_(context.create_pipeline(
          vda_conv2d8_spv, vda_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          vda_conv2d_half_spv, vda_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          vda_conv2d8_half_spv,
          vda_conv2d8_half_spv_size,
          4,
          48)),
      conv2d_pointwise_gemm_(context.create_pipeline(
          vda_conv2d_pointwise_gemm_spv,
          vda_conv2d_pointwise_gemm_spv_size,
          4,
          48)),
      conv2d_pointwise_gemm_half_(context.create_pipeline(
          vda_conv2d_pointwise_gemm_half_spv,
          vda_conv2d_pointwise_gemm_half_spv_size,
          4,
          48)),
      conv2d8_tiled16x8_(context.create_pipeline(
          vda_conv2d8_tiled16x8_spv,
          vda_conv2d8_tiled16x8_spv_size,
          4,
          48)),
      conv2d8_tiled16x8_half_(context.create_pipeline(
          vda_conv2d8_tiled16x8_half_spv,
          vda_conv2d8_tiled16x8_half_spv_size,
          4,
          48)),
      conv2d_implicit_gemm_(context.create_pipeline(
          vda_conv2d_implicit_gemm_spv,
          vda_conv2d_implicit_gemm_spv_size,
          4,
          48)),
      conv2d_implicit_gemm_half_(context.create_pipeline(
          vda_conv2d_implicit_gemm_half_spv,
          vda_conv2d_implicit_gemm_half_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          vda_conv_transpose_nonoverlap_spv,
          vda_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          vda_conv_transpose_nonoverlap_half_spv,
          vda_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          vda_bilinear_align_true_spv,
          vda_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          vda_bilinear_align_true_image_spv,
          vda_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          vda_relu_spv, vda_relu_spv_size, 2, 4)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    linear_vec4_.set_debug_name("linear_vec4");
    linear_vec4_half_.set_debug_name("linear_vec4_half");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_vec8_half_.set_debug_name("linear_vec8_half");
    linear_vec16_.set_debug_name("linear_vec16");
    linear_vec16_half_.set_debug_name("linear_vec16_half");
    if (context.supports_packed_int8_dot() &&
        inferbridge::native::requested_precision() ==
            inferbridge::native::Precision::int8) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv2d_pointwise_gemm_.set_debug_name(
        "conv2d_pointwise_gemm");
    conv2d_pointwise_gemm_half_.set_debug_name(
        "conv2d_pointwise_gemm_half");
    conv2d8_tiled16x8_.set_debug_name("conv2d8_tiled16x8");
    conv2d8_tiled16x8_half_.set_debug_name(
        "conv2d8_tiled16x8_half");
    conv2d_implicit_gemm_.set_debug_name(
        "conv2d_implicit_gemm");
    conv2d_implicit_gemm_half_.set_debug_name(
        "conv2d_implicit_gemm_half");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
}

#include <inferbridge/vulkan/operators/vulkan_operators_linear_int8_variant_2.h>

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool block16,
    bool half_weight) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
    } parameters{rows, input_columns, output_columns};
    const bool vectorized = input_columns % 4u == 0u;
    VulkanPipeline& pipeline = vectorized
        ? (half_weight ? linear_vec8_half_ : linear_vec8_)
        : (half_weight
            ? (block16 ? linear16_half_ : linear_half_)
            : (block16 ? linear16_ : linear_));
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        vectorized
            ? divide_up(divide_up(output_columns, 4), 16)
            : divide_up(divide_up(output_columns, 4), 8),
        vectorized
            ? divide_up(divide_up(rows, 7), 8)
            : divide_up(divide_up(rows, 4), 8));
    if (gelu) {
        struct GeluParameters {
            std::uint32_t count;
        } gelu_parameters{rows * output_columns};
        context_.dispatch(
            gelu_,
            {&output, &output},
            &gelu_parameters,
            sizeof(gelu_parameters),
            divide_up(gelu_parameters.count, 256));
    }
}

#include <inferbridge/vulkan/operators/vulkan_operators_layer_norm_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_scaled.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_head64_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_prepare_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_project_tokens_variant_1.h>

void VulkanOperators::conv2d(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias,
    bool block8,
    bool half_weight,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2 * padding < kernel ||
        input_height + 2 * padding < kernel ||
        batches == 0) {
        throw std::invalid_argument("invalid convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * input_channels,
        "convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * output_channels,
        "convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
        std::uint32_t has_bias;
        std::uint32_t batches;
        std::uint32_t output_channel_blocks;
    };
    const bool pointwise =
        kernel == 1 && stride == 1 && padding == 0 &&
        output_width == input_width && output_height == input_height;
    const bool tiled =
        kernel == 3 && stride == 1 && padding == 1 &&
        output_width == input_width && output_height == input_height;
    const bool implicit_gemm =
        kernel == 3 && stride == 2;
    const std::uint32_t output_channel_blocks =
        divide_up(output_channels, tiled ? 8 : (block8 ? 8 : 4));
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u,
        batches, output_channel_blocks,
    };
    VulkanPipeline& pipeline = pointwise
        ? (half_weight
            ? conv2d_pointwise_gemm_half_
            : conv2d_pointwise_gemm_)
        : (tiled
            ? (half_weight
                ? conv2d8_tiled16x8_half_
                : conv2d8_tiled16x8_)
            : (implicit_gemm
                ? (half_weight
                    ? conv2d_implicit_gemm_half_
                    : conv2d_implicit_gemm_)
                : (half_weight
                    ? (block8 ? conv2d8_half_ : conv2d_half_)
                    : (block8 ? conv2d8_ : conv2d_))));
    context_.dispatch(
        pipeline,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        pointwise
            ? divide_up(output_width * output_height, 32)
            : implicit_gemm
                ? divide_up(batches * output_width * output_height, 32)
            : divide_up(output_width, tiled ? 16 : 8),
        pointwise
            ? divide_up(output_channels, 32)
            : implicit_gemm
                ? divide_up(output_channels, 32)
            : divide_up(output_height, 8),
        (pointwise || implicit_gemm)
            ? (pointwise ? batches : 1u)
            : output_channel_blocks * batches);
}

#include <inferbridge/vulkan/operators/vulkan_operators_conv_transpose_nonoverlap_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_image.h>

#include <inferbridge/vulkan/operators/vulkan_operators_relu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add.h>

}  // namespace vda_native
