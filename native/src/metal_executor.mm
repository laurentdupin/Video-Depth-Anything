#include "metal_executor.h"
#include "inferbridge/native_harness_metal_texture.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vda_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t index = 0; index < tensor.rank; ++index)
        [result addObject:@(tensor.dimensions[index])];
    return result;
}

NSString* ns(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

float cubic1(float x) {
    constexpr float a = -0.75f;
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
}

float cubic2(float x) {
    constexpr float a = -0.75f;
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
}

std::array<float, 4> cubic_coefficients(float value) {
    return {cubic2(value + 1.0f), cubic1(value),
            cubic1(1.0f - value), cubic2(2.0f - value)};
}

std::vector<float> position_embedding(
    const TensorView& source, int patch_height, int patch_width,
    int embedding) {
    if (source.rank != 3 || source.dimensions[0] != 1 ||
        source.dimensions[1] != 1370 ||
        source.dimensions[2] != static_cast<std::uint64_t>(embedding))
        throw std::runtime_error("invalid VDA DINO position embedding");
    std::vector<float> result(
        static_cast<std::size_t>(patch_height * patch_width + 1) * embedding);
    std::copy_n(source.data, embedding, result.data());
    const float scale_y = 37.0f / (static_cast<float>(patch_height) + 0.1f);
    const float scale_x = 37.0f / (static_cast<float>(patch_width) + 0.1f);
    for (int y = 0; y < patch_height; ++y) {
        const float sy = (y + 0.5f) * scale_y - 0.5f;
        const int by = static_cast<int>(std::floor(sy));
        const auto cy = cubic_coefficients(sy - by);
        for (int x = 0; x < patch_width; ++x) {
            const float sx = (x + 0.5f) * scale_x - 0.5f;
            const int bx = static_cast<int>(std::floor(sx));
            const auto cx = cubic_coefficients(sx - bx);
            float* destination = result.data() +
                static_cast<std::size_t>(1 + y * patch_width + x) * embedding;
            for (int channel = 0; channel < embedding; ++channel) {
                float value = 0.0f;
                for (int ky = 0; ky < 4; ++ky) {
                    const int py = std::clamp(by - 1 + ky, 0, 36);
                    for (int kx = 0; kx < 4; ++kx) {
                        const int px = std::clamp(bx - 1 + kx, 0, 36);
                        value += cy[ky] * cx[kx] * source.data[
                            (static_cast<std::size_t>(1 + py * 37 + px) *
                             embedding) + channel];
                    }
                }
                destination[channel] = value;
            }
        }
    }
    return result;
}

struct CacheShape {
    int spatial = 0;
    int channels = 0;
};

class GraphBuilder {
public:
    GraphBuilder(
        const ModelFile& model, const ModelConfig& config,
        int frames, int width, int height, bool fp16, bool cached)
        : model_(model), config_(config), frames_(frames), width_(width),
          height_(height), patch_width_(width / 14),
          patch_height_(height / 14), tokens_(patch_width_ * patch_height_ + 1),
          fp16_(fp16), cached_(cached), graph_([MPSGraph new]) {}

    void build() {
        input_ = [graph_ placeholderWithShape:shape({frames_, 3, height_, width_})
                                     dataType:MPSDataTypeFloat32
                                         name:@"normalized_rgb_tchw"];
        MPSGraphTensor* current = internal(input_);
        current = conv(current, "pretrained.patch_embed.proj", 14, 0);
        current = [graph_ reshapeTensor:current
                              withShape:shape({frames_, config_.embedding,
                                               patch_height_ * patch_width_})
                                   name:nil];
        current = [graph_ transposeTensor:current dimension:1
                            withDimension:2 name:nil];
        MPSGraphTensor* cls = constant("pretrained.cls_token");
        if (frames_ != 1)
            cls = [graph_ tileTensor:cls
                      withMultiplier:shape({frames_, 1, 1}) name:nil];
        current = [graph_ concatTensors:@[cls, current] dimension:1 name:nil];
        const auto positions = position_embedding(
            model_.tensor("pretrained.pos_embed"), patch_height_,
            patch_width_, config_.embedding);
        current = add(current, owned_constant(
            positions, shape({1, tokens_, config_.embedding})));

        std::array<MPSGraphTensor*, 4> captured{};
        int capture = 0;
        for (int block = 0; block < static_cast<int>(config_.blocks); ++block) {
            const std::string prefix =
                "pretrained.blocks." + std::to_string(block);
            MPSGraphTensor* normalized = layer_norm(
                current, prefix + ".norm1", 1.0e-6f);
            MPSGraphTensor* qkv = linear(normalized, prefix + ".attn.qkv");
            qkv = [graph_ reshapeTensor:qkv
                              withShape:shape({frames_, tokens_, 3,
                                  config_.heads, 64}) name:nil];
            qkv = [graph_ transposeTensor:qkv
                              permutation:@[@2, @0, @3, @1, @4] name:nil];
            MPSGraphTensor* query = [graph_ sliceTensor:qkv dimension:0
                                                   start:0 length:1 name:nil];
            MPSGraphTensor* key = [graph_ sliceTensor:qkv dimension:0
                                                 start:1 length:1 name:nil];
            MPSGraphTensor* value = [graph_ sliceTensor:qkv dimension:0
                                                   start:2 length:1 name:nil];
            query = [graph_ reshapeTensor:query withShape:shape({frames_,
                config_.heads, tokens_, 64}) name:nil];
            key = [graph_ reshapeTensor:key withShape:shape({frames_,
                config_.heads, tokens_, 64}) name:nil];
            value = [graph_ reshapeTensor:value withShape:shape({frames_,
                config_.heads, tokens_, 64}) name:nil];
            query = multiply(query, scalar(0.125f));
            key = [graph_ transposeTensor:key dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:query
                secondaryTensor:key name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attention = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:value name:nil];
            attention = [graph_ transposeTensor:attention
                                      dimension:1 withDimension:2 name:nil];
            attention = [graph_ reshapeTensor:attention
                                    withShape:shape({frames_, tokens_,
                                        config_.embedding}) name:nil];
            MPSGraphTensor* projected = linear(
                attention, prefix + ".attn.proj");
            current = add(current, multiply(
                projected, constant(prefix + ".ls1.gamma")));
            normalized = layer_norm(current, prefix + ".norm2", 1.0e-6f);
            MPSGraphTensor* hidden = gelu(linear(
                normalized, prefix + ".mlp.fc1"));
            hidden = linear(hidden, prefix + ".mlp.fc2");
            current = add(current, multiply(
                hidden, constant(prefix + ".ls2.gamma")));
            if (capture < 4 && block == static_cast<int>(config_.captures[capture]))
                captured[capture++] = layer_norm(
                    current, "pretrained.norm", 1.0e-6f);
        }
        if (capture != 4)
            throw std::runtime_error("VDA Metal encoder capture mismatch");
        output_ = build_dpt(captured);
        output_ = external(output_);
        targets_.push_back(output_);
        for (MPSGraphTensor* cache : cache_outputs_)
            targets_.push_back(external(cache));
    }

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }
    const std::vector<MPSGraphTensor*>& histories() const { return histories_; }
    const std::vector<MPSGraphTensor*>& targets() const { return targets_; }
    const std::vector<CacheShape>& cache_shapes() const { return cache_shapes_; }

private:
    MPSGraphTensor* internal(MPSGraphTensor* value) {
        return fp16_ ? [graph_ castTensor:value
                                  toType:MPSDataTypeFloat16 name:nil] : value;
    }
    MPSGraphTensor* external(MPSGraphTensor* value) {
        return value.dataType == MPSDataTypeFloat32 ? value :
            [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor* constant(const std::string& tensor_name) {
        const TensorView& tensor = model_.tensor(tensor_name);
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        return internal([graph_ constantWithData:data shape:shape(tensor)
                                        dataType:MPSDataTypeFloat32]);
    }
    MPSGraphTensor* owned_constant(
        const std::vector<float>& values, MPSShape* dimensions) {
        NSData* data = [NSData dataWithBytes:values.data()
                                      length:values.size() * sizeof(float)];
        return internal([graph_ constantWithData:data shape:dimensions
                                        dataType:MPSDataTypeFloat32]);
    }
    MPSGraphTensor* scalar(float value) {
        return internal([graph_ constantWithScalar:value
                                          dataType:MPSDataTypeFloat32]);
    }
    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* linear(MPSGraphTensor* value, const std::string& prefix) {
        MPSGraphTensor* weight = [graph_ transposeTensor:constant(prefix + ".weight")
                                                  dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:ns(prefix)];
        return model_.contains(prefix + ".bias")
            ? add(result, constant(prefix + ".bias")) : result;
    }
    MPSGraphTensor* layer_norm(
        MPSGraphTensor* value, const std::string& prefix, float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(prefix + ".weight")
            betaTensor:constant(prefix + ".bias") epsilon:epsilon name:ns(prefix)];
    }
    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(static_cast<float>(M_SQRT1_2))) name:nil];
        return multiply(multiply(value, scalar(0.5f)),
                        add(error, scalar(1.0f)));
    }
    MPSGraphConvolution2DOpDescriptor* conv_descriptor(int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride
            dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:padding paddingRight:padding
            paddingTop:padding paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }
    MPSGraphTensor* conv(
        MPSGraphTensor* value, const std::string& prefix,
        int stride, int padding, bool bias = true) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(prefix + ".weight")
            descriptor:conv_descriptor(stride, padding) name:ns(prefix)];
        if (bias && model_.contains(prefix + ".bias")) {
            const int channels = static_cast<int>(
                model_.tensor(prefix + ".bias").elements);
            result = add(result, [graph_ reshapeTensor:constant(prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        }
        return result;
    }
    MPSGraphTensor* conv_transpose(
        MPSGraphTensor* value, const std::string& prefix,
        int channels, int source_height, int source_width, int stride) {
        MPSGraphTensor* result = [graph_
            convolutionTranspose2DWithSourceTensor:value
            weightsTensor:constant(prefix + ".weight")
            outputShape:shape({frames_, channels, source_height * stride,
                               source_width * stride})
            descriptor:conv_descriptor(stride, 0) name:ns(prefix)];
        return add(result, [graph_ reshapeTensor:constant(prefix + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil]);
    }
    MPSGraphTensor* resize(MPSGraphTensor* value, int height, int width) {
        return [graph_ resizeTensor:value size:shape({height, width})
                              mode:MPSGraphResizeBilinear centerResult:NO
                      alignCorners:YES layout:MPSGraphTensorNamedDataLayoutNCHW
                              name:nil];
    }
    MPSGraphTensor* residual_unit(
        MPSGraphTensor* value, const std::string& prefix) {
        MPSGraphTensor* path = [graph_ reLUWithTensor:value name:nil];
        path = conv(path, prefix + ".conv1", 1, 1);
        path = [graph_ reLUWithTensor:path name:nil];
        return add(value, conv(path, prefix + ".conv2", 1, 1));
    }
    MPSGraphTensor* fusion(
        MPSGraphTensor* path, MPSGraphTensor* skip,
        const std::string& prefix, int height, int width) {
        if (skip != nil)
            path = add(path, residual_unit(skip, prefix + ".resConfUnit1"));
        path = residual_unit(path, prefix + ".resConfUnit2");
        return conv(resize(path, height, width), prefix + ".out_conv", 1, 0);
    }
    MPSGraphTensor* temporal(
        MPSGraphTensor* input, int module, int channels, int height, int width) {
        const int spatial = height * width;
        const std::string base = "head.motion_modules." +
            std::to_string(module) + ".temporal_transformer.";
        MPSGraphTensor* grouped = [graph_ reshapeTensor:input
            withShape:shape({frames_, 32, channels / 32, spatial}) name:nil];
        NSArray<NSNumber*>* axes = @[@2, @3];
        MPSGraphTensor* mean = [graph_ meanOfTensor:grouped axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:grouped
            meanTensor:mean axes:axes name:nil];
        MPSGraphTensor* gamma = [graph_ reshapeTensor:constant(base + "norm.weight")
            withShape:shape({1, 32, channels / 32, 1}) name:nil];
        MPSGraphTensor* beta = [graph_ reshapeTensor:constant(base + "norm.bias")
            withShape:shape({1, 32, channels / 32, 1}) name:nil];
        grouped = [graph_ normalizationWithTensor:grouped meanTensor:mean
            varianceTensor:variance gammaTensor:gamma betaTensor:beta
            epsilon:1.0e-6f name:nil];
        MPSGraphTensor* frame_spatial = [graph_ reshapeTensor:grouped
            withShape:shape({frames_, channels, height, width}) name:nil];
        frame_spatial = [graph_ transposeTensor:frame_spatial
            permutation:@[@0, @2, @3, @1] name:nil];
        frame_spatial = [graph_ reshapeTensor:frame_spatial
            withShape:shape({frames_ * spatial, channels}) name:nil];
        MPSGraphTensor* projected = linear(frame_spatial, base + "proj_in");
        projected = [graph_ reshapeTensor:projected
            withShape:shape({frames_, spatial, channels}) name:nil];
        MPSGraphTensor* state = [graph_ transposeTensor:projected
            dimension:0 withDimension:1 name:nil];
        const std::string block = base + "transformer_blocks.0.";
        for (int attention = 0; attention < 2; ++attention) {
            MPSGraphTensor* normalized = layer_norm(state,
                block + "norms." + std::to_string(attention), 1.0e-5f);
            MPSGraphTensor* attention_input = normalized;
            int attention_frames = frames_;
            if (cached_) {
                MPSGraphTensor* history = [graph_ placeholderWithShape:
                    shape({spatial, 31, channels}) dataType:MPSDataTypeFloat32
                    name:ns("history_" + std::to_string(module) + "_" +
                            std::to_string(attention))];
                histories_.push_back(history);
                cache_shapes_.push_back({spatial, channels});
                cache_outputs_.push_back(normalized);
                attention_input = [graph_ concatTensors:
                    @[internal(history), normalized] dimension:1 name:nil];
                attention_frames = 32;
            } else if (frames_ == 1) {
                cache_shapes_.push_back({spatial, channels});
                cache_outputs_.push_back(normalized);
            }
            const std::string ap = block + "attention_blocks." +
                std::to_string(attention) + ".";
            MPSGraphTensor* position = constant(ap + "pos_encoder.pe");
            if (attention_frames < 32)
                position = [graph_ sliceTensor:position dimension:1
                    start:0 length:attention_frames name:nil];
            MPSGraphTensor* positioned = add(attention_input, position);
            MPSGraphTensor* query = linear(positioned, ap + "to_q");
            MPSGraphTensor* key = linear(positioned, ap + "to_k");
            MPSGraphTensor* value = linear(positioned, ap + "to_v");
            query = [graph_ reshapeTensor:query withShape:
                shape({spatial, attention_frames, 8, channels / 8}) name:nil];
            key = [graph_ reshapeTensor:key withShape:
                shape({spatial, attention_frames, 8, channels / 8}) name:nil];
            value = [graph_ reshapeTensor:value withShape:
                shape({spatial, attention_frames, 8, channels / 8}) name:nil];
            query = [graph_ transposeTensor:query
                permutation:@[@0, @2, @1, @3] name:nil];
            key = [graph_ transposeTensor:key
                permutation:@[@0, @2, @3, @1] name:nil];
            value = [graph_ transposeTensor:value
                permutation:@[@0, @2, @1, @3] name:nil];
            if (cached_)
                query = [graph_ sliceTensor:query dimension:2
                    start:31 length:1 name:nil];
            query = multiply(query, scalar(
                1.0f / std::sqrt(static_cast<float>(channels / 8))));
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:query
                secondaryTensor:key name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attended = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:value name:nil];
            attended = [graph_ transposeTensor:attended
                permutation:@[@0, @2, @1, @3] name:nil];
            attended = [graph_ reshapeTensor:attended
                withShape:shape({spatial, cached_ ? 1 : frames_, channels})
                name:nil];
            state = add(state, linear(attended, ap + "to_out.0"));
        }
        MPSGraphTensor* normalized = layer_norm(
            state, block + "ff_norm", 1.0e-5f);
        MPSGraphTensor* expanded = linear(normalized, block + "ff.net.0.proj");
        MPSGraphTensor* first = [graph_ sliceTensor:expanded dimension:2
            start:0 length:channels * 4 name:nil];
        MPSGraphTensor* gate = [graph_ sliceTensor:expanded dimension:2
            start:channels * 4 length:channels * 4 name:nil];
        state = add(state, linear(multiply(first, gelu(gate)),
                                  block + "ff.net.2"));
        MPSGraphTensor* ordered = [graph_ transposeTensor:state
            dimension:0 withDimension:1 name:nil];
        ordered = [graph_ reshapeTensor:ordered
            withShape:shape({frames_, height, width, channels}) name:nil];
        ordered = [graph_ transposeTensor:ordered
            permutation:@[@0, @3, @1, @2] name:nil];
        MPSGraphTensor* rows = [graph_ transposeTensor:ordered
            permutation:@[@0, @2, @3, @1] name:nil];
        rows = [graph_ reshapeTensor:rows
            withShape:shape({frames_ * spatial, channels}) name:nil];
        rows = linear(rows, base + "proj_out");
        rows = [graph_ reshapeTensor:rows
            withShape:shape({frames_, height, width, channels}) name:nil];
        rows = [graph_ transposeTensor:rows
            permutation:@[@0, @3, @1, @2] name:nil];
        return add(input, rows);
    }
    MPSGraphTensor* build_dpt(
        const std::array<MPSGraphTensor*, 4>& features) {
        std::array<MPSGraphTensor*, 4> layers{};
        const std::array<int, 4> heights{patch_height_ * 4,
            patch_height_ * 2, patch_height_, (patch_height_ + 1) / 2};
        const std::array<int, 4> widths{patch_width_ * 4,
            patch_width_ * 2, patch_width_, (patch_width_ + 1) / 2};
        for (int index = 0; index < 4; ++index) {
            MPSGraphTensor* patches = [graph_ sliceTensor:features[index]
                dimension:1 start:1 length:patch_width_ * patch_height_ name:nil];
            patches = [graph_ transposeTensor:patches
                dimension:1 withDimension:2 name:nil];
            patches = [graph_ reshapeTensor:patches withShape:
                shape({frames_, config_.embedding, patch_height_, patch_width_})
                name:nil];
            const std::string project = "head.projects." + std::to_string(index);
            patches = conv(patches, project, 1, 0);
            if (index < 2) {
                const int scale = index == 0 ? 4 : 2;
                patches = conv_transpose(patches,
                    "head.resize_layers." + std::to_string(index),
                    config_.project_channels[index], patch_height_,
                    patch_width_, scale);
            } else if (index == 3) {
                patches = conv(patches, "head.resize_layers.3", 2, 1);
            }
            layers[index] = patches;
        }
        layers[2] = temporal(layers[2], 0, config_.project_channels[2],
                             heights[2], widths[2]);
        layers[3] = temporal(layers[3], 1, config_.project_channels[3],
                             heights[3], widths[3]);
        for (int index = 0; index < 4; ++index)
            layers[index] = conv(layers[index], "head.scratch.layer" +
                std::to_string(index + 1) + "_rn", 1, 1, false);
        MPSGraphTensor* path = fusion(layers[3], nil,
            "head.scratch.refinenet4", heights[2], widths[2]);
        path = temporal(path, 2, config_.features, heights[2], widths[2]);
        path = fusion(path, layers[2], "head.scratch.refinenet3",
                      heights[1], widths[1]);
        path = temporal(path, 3, config_.features, heights[1], widths[1]);
        path = fusion(path, layers[1], "head.scratch.refinenet2",
                      heights[0], widths[0]);
        path = fusion(path, layers[0], "head.scratch.refinenet1",
                      heights[0] * 2, widths[0] * 2);
        path = conv(path, "head.scratch.output_conv1", 1, 1);
        path = resize(path, height_, width_);
        path = conv(path, "head.scratch.output_conv2.0", 1, 1);
        path = [graph_ reLUWithTensor:path name:nil];
        path = conv(path, "head.scratch.output_conv2.2", 1, 0);
        return [graph_ reLUWithTensor:path name:nil];
    }

    const ModelFile& model_;
    const ModelConfig& config_;
    int frames_;
    int width_;
    int height_;
    int patch_width_;
    int patch_height_;
    int tokens_;
    bool fp16_;
    bool cached_;
    MPSGraph* graph_;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
    std::vector<MPSGraphTensor*> histories_;
    std::vector<MPSGraphTensor*> cache_outputs_;
    std::vector<MPSGraphTensor*> targets_;
    std::vector<CacheShape> cache_shapes_;
};

struct PlanKey {
    int frames;
    int width;
    int height;
    bool cached;
    bool operator==(const PlanKey& other) const {
        return frames == other.frames && width == other.width &&
            height == other.height &&
            cached == other.cached;
    }
};
struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
        std::size_t result = static_cast<std::size_t>(key.frames);
        result = result * 1315423911u + key.width;
        result = result * 1315423911u + key.height;
        return result * 2u + static_cast<std::size_t>(key.cached);
    }
};
struct Plan {
    MPSGraph* graph = nil;
    MPSGraphExecutable* executable = nil;
    std::vector<CacheShape> cache_shapes;
    int frames = 0;
    int width = 0;
    int height = 0;
    bool cached = false;
};
struct StreamEntry {
    std::array<std::vector<float>, 8> attention;
    std::array<id<MTLBuffer>, 8> metal_attention{};
    std::array<MPSGraphTensorData*, 8> metal_data{};
    std::shared_ptr<
        inferbridge::native_harness::metal::AuxiliaryTensorPool::Lease>
        metal_lease;
};

class MetalExternalJob final : public ExternalJob {
public:
    explicit MetalExternalJob(
        std::shared_ptr<inferbridge::native_harness::metal::Submission> value)
        : submission_(std::move(value)) {}
    ExternalJobState state() const override {
        if (submission_->cancelled()) return ExternalJobState::cancelled;
        return submission_->complete() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { submission_->cancel(); }
private:
    std::shared_ptr<inferbridge::native_harness::metal::Submission> submission_;
};

}  // namespace

class MetalExecutor::Impl {
public:
    Impl(const ModelFile& model, const ModelConfig& config)
        : model_(model), config_(config) {
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("VDA Metal does not support INT8 yet");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil)
            throw std::runtime_error("Metal is unavailable on this Mac");
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize VDA Metal");
        inferbridge::native_harness::metal::label_queue(
            queue_, "Video Depth Anything");
        tensor_pool_ = std::make_shared<
            inferbridge::native_harness::metal::AuxiliaryTensorPool>(
                device_, "Video Depth Anything");
        create_texture_pipelines();
    }

    void reset_stream() {
        std::lock_guard<std::mutex> guard(mutex_);
        cache_.clear();
        stream_id_ = -1;
        stream_size_ = 0;
    }

    void infer_tensor(
        const float* input, std::uint32_t frames,
        std::uint32_t width, std::uint32_t height, float* depth) {
        if (frames != 32)
            throw std::invalid_argument(
                "VDA Metal tensor inference requires 32 frames");
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            std::array<std::vector<float>, 8> unused;
            run(get_plan(32, static_cast<int>(width),
                         static_cast<int>(height), false), input,
                nullptr, depth, unused);
        }
    }

    void infer_stream(const float* input, std::uint32_t size, float* depth) {
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            if (!cache_.empty() && stream_size_ != size)
                throw std::invalid_argument(
                    "VDA Metal stream size changed without reset");
            if (cache_.empty()) {
                auto seed = std::make_shared<StreamEntry>();
                run(get_plan(1, static_cast<int>(size),
                             static_cast<int>(size), false), input,
                    nullptr, depth, seed->attention);
                cache_.assign(32, seed);
                stream_size_ = size;
            }
            std::vector<std::shared_ptr<StreamEntry>> selected;
            selected.reserve(31);
            selected.push_back(cache_[0]);
            selected.push_back(cache_[1]);
            const std::size_t tail = cache_.size() - 29;
            selected.insert(selected.end(), cache_.begin() + tail, cache_.end());
            std::array<std::vector<float>, 8> histories;
            const Plan& plan = get_plan(1, static_cast<int>(size),
                                        static_cast<int>(size), true);
            for (std::size_t cache_index = 0; cache_index < 8; ++cache_index) {
                const CacheShape cs = plan.cache_shapes[cache_index];
                histories[cache_index].resize(
                    static_cast<std::size_t>(cs.spatial) * 31 * cs.channels);
                for (int spatial = 0; spatial < cs.spatial; ++spatial) {
                    for (int frame = 0; frame < 31; ++frame) {
                        const float* source = selected[frame]->attention[cache_index].data() +
                            static_cast<std::size_t>(spatial) * cs.channels;
                        float* destination = histories[cache_index].data() +
                            (static_cast<std::size_t>(spatial) * 31 + frame) * cs.channels;
                        std::copy_n(source, cs.channels, destination);
                    }
                }
            }
            auto current = std::make_shared<StreamEntry>();
            run(plan, input, &histories, depth, current->attention);
            cache_.push_back(current);
            ++stream_id_;
            if (stream_id_ + 32 > 42)
                cache_.erase(cache_.begin() + 1);
        }
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) {
        if (!request.shared_texture_handle || !request.output_texture_handle ||
            !request.signal_fence_handle || !request.signal_fence_value ||
            !request.width || !request.height || !request.process_resolution ||
            request.process_resolution % 14u != 0u ||
            request.output_width != request.width ||
            request.output_height != request.height)
            throw std::invalid_argument("invalid VDA Metal texture request");
        inferbridge::native_harness::metal::Prepared prepared;
        prepared.input_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.shared_texture_handle));
        prepared.output_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.output_texture_handle));
        prepared.wait_event = request.wait_fence_handle ?
            (__bridge id<MTLSharedEvent>)(reinterpret_cast<void*>(
                request.wait_fence_handle)) : nil;
        prepared.signal_event = (__bridge id<MTLSharedEvent>)(
            reinterpret_cast<void*>(request.signal_fence_handle));
        prepared.signal_value = request.signal_fence_value;
        if (prepared.input_texture.device.registryID != device_.registryID ||
            prepared.output_texture.device.registryID != device_.registryID ||
            prepared.input_texture.textureType != MTLTextureType2D ||
            prepared.input_texture.width != request.width ||
            prepared.input_texture.height != request.height ||
            prepared.input_texture.pixelFormat != MTLPixelFormatBGRA8Unorm ||
            prepared.output_texture.textureType != MTLTextureType2D ||
            prepared.output_texture.width != request.output_width ||
            prepared.output_texture.height != request.output_height ||
            prepared.output_texture.pixelFormat != MTLPixelFormatR32Float)
            throw std::invalid_argument("VDA Metal texture descriptor mismatch");
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            if (request.reset) {
                cache_.clear(); stream_id_ = -1; stream_size_ = 0;
            }
            const std::uint32_t size = request.process_resolution;
            if (!cache_.empty() && stream_size_ != size)
                throw std::invalid_argument(
                    "VDA Metal stream size changed without reset");
            const Plan& plan=get_plan(1,size,size,true);
            std::vector<inferbridge::native_harness::metal::TensorSpec> specs;
            specs.push_back({{1,3,(NSInteger)size,(NSInteger)size},
                MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,"Network Input"});
            for(std::size_t module=0;module<8;++module){
                const CacheShape cs=plan.cache_shapes[module];
                specs.push_back({{cs.spatial,31,cs.channels},
                    MPSDataTypeFloat32,sizeof(float),
                    MTLResourceStorageModePrivate,
                    "Attention History " + std::to_string(module)});
            }
            specs.push_back({{1,1,(NSInteger)size,(NSInteger)size},
                MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,"Network Depth"});
            specs.push_back({{1,1,(NSInteger)size,(NSInteger)size},
                MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,"Seed Depth"});
            specs.push_back({{1,1,(NSInteger)request.height,
                (NSInteger)request.width},MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,"Resized Depth"});
            specs.push_back({{2},MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,"Depth Range"});
            auto tensors=tensor_pool_->acquire(specs);
            prepared.retained_resources.push_back(tensors);
            id<MTLBuffer> input=tensors->buffer(0);
            id<MTLCommandBuffer> preprocess = [queue_ commandBuffer];
            inferbridge::native_harness::metal::label_command(
                preprocess,"Video Depth Anything","Preprocess");
            if (prepared.wait_event)
                [preprocess encodeWaitForEvent:prepared.wait_event
                    value:request.wait_fence_value];
            id<MTLComputeCommandEncoder> encoder =
                [preprocess computeCommandEncoder];
            inferbridge::native_harness::metal::label_encoder(
                encoder,"Video Depth Anything","Preprocess");
            struct PreprocessParameters { uint32_t width,height,size; } pp{
                request.width,request.height,size};
            [encoder setComputePipelineState:preprocess_pipeline_];
            [encoder setTexture:prepared.input_texture atIndex:0];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBytes:&pp length:sizeof(pp) atIndex:1];
            dispatch(encoder,preprocess_pipeline_,size,size,1);
            [encoder endEncoding]; [preprocess commit];

            if (cache_.empty()) {
                const Plan& seed_plan = get_plan(1,size,size,false);
                auto seed = make_metal_entry(seed_plan);
                run_external(seed_plan,tensors->data(0),nullptr,
                    tensors->data(10),*seed);
                cache_.assign(32,seed); stream_size_=size;
            }
            std::vector<std::shared_ptr<StreamEntry>> selected;
            selected.reserve(31); selected.push_back(cache_[0]);
            selected.push_back(cache_[1]);
            const std::size_t tail=cache_.size()-29;
            selected.insert(selected.end(),cache_.begin()+tail,cache_.end());
            std::array<id<MTLBuffer>,8> histories{};
            std::array<MPSGraphTensorData*,8> history_data{};
            id<MTLCommandBuffer> packing=[queue_ commandBuffer];
            inferbridge::native_harness::metal::label_command(
                packing,"Video Depth Anything","Pack Attention History");
            encoder=[packing computeCommandEncoder];
            inferbridge::native_harness::metal::label_encoder(
                encoder,"Video Depth Anything","Pack Attention History");
            for(std::size_t module=0;module<8;++module){
                const CacheShape cs=plan.cache_shapes[module];
                histories[module]=tensors->buffer(1+module);
                history_data[module]=tensors->data(1+module);
                for(uint32_t frame=0;frame<31;++frame){
                    struct PackParameters{uint32_t spatial,channels,frame;} p{
                        static_cast<uint32_t>(cs.spatial),
                        static_cast<uint32_t>(cs.channels),frame};
                    [encoder setComputePipelineState:pack_pipeline_];
                    [encoder setBuffer:selected[frame]->metal_attention[module]
                        offset:0 atIndex:0];
                    [encoder setBuffer:histories[module] offset:0 atIndex:1];
                    [encoder setBytes:&p length:sizeof(p) atIndex:2];
                    dispatch(encoder,pack_pipeline_,cs.spatial,cs.channels,1);
                }
            }
            [encoder endEncoding]; [packing commit];
            auto current=make_metal_entry(plan);
            id<MTLBuffer> network_depth=tensors->buffer(9);
            run_external(plan,tensors->data(0),&history_data,
                tensors->data(9),*current);
            for(const auto& entry:selected)
                prepared.retained_resources.push_back(entry);
            prepared.retained_resources.push_back(current);
            cache_.push_back(current); ++stream_id_;
            if(stream_id_+32>42)cache_.erase(cache_.begin()+1);

            id<MTLBuffer> resized=tensors->buffer(11);
            id<MTLBuffer> range=tensors->buffer(12);
            id<MTLCommandBuffer> completion=[queue_ commandBuffer];
            inferbridge::native_harness::metal::label_command(
                completion,"Video Depth Anything","Resize and Present Depth");
            encoder=[completion computeCommandEncoder];
            inferbridge::native_harness::metal::label_encoder(
                encoder,"Video Depth Anything","Resize and Present Depth");
            struct ResizeParameters{uint32_t sw,sh,width,height;} rp{
                size,size,request.width,request.height};
            [encoder setComputePipelineState:resize_pipeline_];
            [encoder setBuffer:network_depth offset:0 atIndex:0];
            [encoder setBuffer:resized offset:0 atIndex:1];
            [encoder setBytes:&rp length:sizeof(rp) atIndex:2];
            dispatch(encoder,resize_pipeline_,request.width,request.height,1);
            if(!config_.metric){
                uint32_t count=request.width*request.height;
                [encoder setComputePipelineState:reduce_pipeline_];
                [encoder setBuffer:resized offset:0 atIndex:0];
                [encoder setBuffer:range offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                dispatch(encoder,reduce_pipeline_,256,1,1);
            }
            struct OutputParameters{uint32_t width,height,normalize;} op{
                request.width,request.height,config_.metric?0u:1u};
            [encoder setComputePipelineState:output_pipeline_];
            [encoder setBuffer:resized offset:0 atIndex:0];
            [encoder setBuffer:range offset:0 atIndex:1];
            [encoder setTexture:prepared.output_texture atIndex:0];
            [encoder setBytes:&op length:sizeof(op) atIndex:2];
            dispatch(encoder,output_pipeline_,request.width,request.height,1);
            [encoder endEncoding];
            [completion encodeSignalEvent:prepared.signal_event
                value:prepared.signal_value];
            [completion commit];
            return std::make_shared<MetalExternalJob>(
                std::make_shared<inferbridge::native_harness::metal::Submission>(
                    prepared,completion));
        }
    }

private:
    static void dispatch(id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline, NSUInteger width,
        NSUInteger height, NSUInteger depth) {
        const NSUInteger x=pipeline.threadExecutionWidth;
        const NSUInteger y=std::max<NSUInteger>(1,
            pipeline.maxTotalThreadsPerThreadgroup/x);
        [encoder dispatchThreads:MTLSizeMake(width,height,depth)
            threadsPerThreadgroup:MTLSizeMake(x,y,1)];
    }

    std::shared_ptr<StreamEntry> make_metal_entry(const Plan& plan) {
        auto result=std::make_shared<StreamEntry>();
        std::vector<inferbridge::native_harness::metal::TensorSpec> specs;
        for(std::size_t index=0;index<8;++index){
            const CacheShape cs=plan.cache_shapes[index];
            specs.push_back({{cs.spatial,1,cs.channels},
                MPSDataTypeFloat32,sizeof(float),
                MTLResourceStorageModePrivate,
                "Current Attention " + std::to_string(index)});
        }
        result->metal_lease=tensor_pool_->acquire(specs);
        for(std::size_t index=0;index<8;++index){
            result->metal_attention[index]=result->metal_lease->buffer(index);
            result->metal_data[index]=result->metal_lease->data(index);
        }
        return result;
    }

    void run_external(const Plan& plan,MPSGraphTensorData* input,
        const std::array<MPSGraphTensorData*,8>* histories,
        MPSGraphTensorData* depth,
        StreamEntry& current) {
        NSMutableArray<MPSGraphTensorData*>* values=[NSMutableArray array];
        [values addObject:input];
        if(histories){
            for(std::size_t index=0;index<8;++index){
                [values addObject:(*histories)[index]];
            }
        }
        NSMutableArray<MPSGraphTensorData*>* outputs=[NSMutableArray array];
        [outputs addObject:depth];
        for(std::size_t index=0;index<8;++index){
            [outputs addObject:current.metal_data[index]];
        }
        MPSGraphExecutableExecutionDescriptor* execution=
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted=NO;
        NSArray* results=[plan.executable runAsyncWithMTLCommandQueue:queue_
            inputsArray:values resultsArray:outputs executionDescriptor:execution];
        if(results.count!=9)
            throw std::runtime_error("VDA Metal output binding failed");
    }

    void create_texture_pipelines(){
        static constexpr char source_text[]=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct PreprocessParameters{uint width,height,size;};
kernel void preprocess(texture2d<float,access::read>src[[texture(0)]],
 device float*dst[[buffer(0)]],constant PreprocessParameters&p[[buffer(1)]],
 uint2 q[[thread_position_in_grid]]){
 if(q.x>=p.size||q.y>=p.size)return;
 uint sx=min(p.width-1,q.x*p.width/p.size),sy=min(p.height-1,q.y*p.height/p.size);
 float4 pixel=src.read(uint2(sx,sy));uint i=q.y*p.size+q.x,plane=p.size*p.size;
 dst[i]=(pixel.z-0.485f)/0.229f;dst[plane+i]=(pixel.y-0.456f)/0.224f;
 dst[2*plane+i]=(pixel.x-0.406f)/0.225f;
}
struct PackParameters{uint spatial,channels,frame;};
kernel void pack_history(device const float*src[[buffer(0)]],device float*dst[[buffer(1)]],
 constant PackParameters&p[[buffer(2)]],uint2 q[[thread_position_in_grid]]){
 if(q.x>=p.spatial||q.y>=p.channels)return;
 dst[(q.x*31+p.frame)*p.channels+q.y]=src[q.x*p.channels+q.y];
}
struct ResizeParameters{uint sw,sh,width,height;};
kernel void resize_depth(device const float*src[[buffer(0)]],device float*dst[[buffer(1)]],
 constant ResizeParameters&p[[buffer(2)]],uint2 q[[thread_position_in_grid]]){
 if(q.x>=p.width||q.y>=p.height)return;
 float fx=p.width>1?float(q.x)*float(p.sw-1)/float(p.width-1):0.0f;
 float fy=p.height>1?float(q.y)*float(p.sh-1)/float(p.height-1):0.0f;
 uint x0=uint(floor(fx)),y0=uint(floor(fy)),x1=min(x0+1,p.sw-1),y1=min(y0+1,p.sh-1);
 float top=mix(src[y0*p.sw+x0],src[y0*p.sw+x1],fx-float(x0));
 float bottom=mix(src[y1*p.sw+x0],src[y1*p.sw+x1],fx-float(x0));
 dst[q.y*p.width+q.x]=mix(top,bottom,fy-float(y0));
}
kernel void reduce_range(device const float*src[[buffer(0)]],device float*range[[buffer(1)]],
 constant uint&count[[buffer(2)]],uint gid[[thread_position_in_grid]]){
 if(gid)return;float lo=INFINITY,hi=-INFINITY;
 for(uint i=0;i<count;++i){lo=min(lo,src[i]);hi=max(hi,src[i]);}
 range[0]=lo;range[1]=hi;
}
struct OutputParameters{uint width,height,normalize;};
kernel void write_output(device const float*src[[buffer(0)]],device const float*range[[buffer(1)]],
 texture2d<float,access::write>out[[texture(0)]],constant OutputParameters&p[[buffer(2)]],
 uint2 q[[thread_position_in_grid]]){
 if(q.x>=p.width||q.y>=p.height)return;float v=src[q.y*p.width+q.x];
 if(p.normalize){float r=range[1]-range[0];v=r>0?(v-range[0])/r:0.0f;}
 out.write(float4(v),q);
}
)METAL";
        NSError* error=nil;
        id<MTLLibrary> library=[device_ newLibraryWithSource:
            [NSString stringWithUTF8String:source_text] options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String?:
            "could not compile VDA Metal texture kernels");
        auto make=[&](NSString*name){
            id<MTLComputePipelineState> result=[device_ newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&error];
            if(!result)throw std::runtime_error(error.localizedDescription.UTF8String?:
                "could not create VDA Metal texture pipeline");return result;};
        preprocess_pipeline_=make(@"preprocess");pack_pipeline_=make(@"pack_history");
        resize_pipeline_=make(@"resize_depth");reduce_pipeline_=make(@"reduce_range");
        output_pipeline_=make(@"write_output");
    }

    const Plan& get_plan(
        int frames, int width, int height, bool cached) {
        const PlanKey key{frames, width, height, cached};
        auto existing = plans_.find(key);
        if (existing != plans_.end()) return existing->second;
        GraphBuilder builder(
            model_, config_, frames, width, height, fp16_, cached);
        builder.build();
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
            [NSMutableDictionary dictionary];
        feeds[builder.input()] = [[MPSGraphShapedType alloc]
            initWithShape:shape({frames, 3, height, width})
            dataType:MPSDataTypeFloat32];
        const auto& histories = builder.histories();
        const auto& cache_shapes = builder.cache_shapes();
        for (std::size_t index = 0; index < histories.size(); ++index)
            feeds[histories[index]] = [[MPSGraphShapedType alloc]
                initWithShape:shape({cache_shapes[index].spatial, 31,
                    cache_shapes[index].channels}) dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        // Keep the temporal graph on Metal. Level 1 attempts an ANE partition
        // which the Apple compiler rejects after several seconds before
        // falling back to the same Metal execution path.
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        NSMutableArray<MPSGraphTensor*>* targets = [NSMutableArray array];
        for (MPSGraphTensor* target : builder.targets())
            [targets addObject:target];
        MPSGraphExecutable* executable = [builder.graph()
            compileWithDevice:graph_device_ feeds:feeds targetTensors:targets
            targetOperations:nil compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error("failed to compile VDA Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        Plan plan;
        plan.graph = builder.graph();
        plan.executable = executable;
        plan.cache_shapes = cache_shapes;
        plan.frames = frames;
        plan.width = width;
        plan.height = height;
        plan.cached = cached;
        return plans_.emplace(key, std::move(plan)).first->second;
    }

    void run(
        const Plan& plan, const float* input,
        const std::array<std::vector<float>, 8>* histories,
        float* depth, std::array<std::vector<float>, 8>& caches) {
        const NSUInteger input_bytes = static_cast<NSUInteger>(
            static_cast<std::uint64_t>(plan.frames) * 3 * plan.width *
            plan.height * sizeof(float));
        id<MTLBuffer> input_buffer = [device_ newBufferWithBytes:input
            length:input_bytes options:MTLResourceStorageModeShared];
        if (input_buffer == nil) throw std::bad_alloc();
        NSMutableArray<MPSGraphTensorData*>* values = [NSMutableArray array];
        [values addObject:[[MPSGraphTensorData alloc]
            initWithMTLBuffer:input_buffer
            shape:shape({plan.frames, 3, plan.height, plan.width})
            dataType:MPSDataTypeFloat32]];
        std::vector<id<MTLBuffer>> history_buffers;
        if (histories != nullptr) {
            history_buffers.reserve(8);
            for (std::size_t index = 0; index < 8; ++index) {
                const auto& history = (*histories)[index];
                id<MTLBuffer> buffer = [device_ newBufferWithBytes:history.data()
                    length:history.size() * sizeof(float)
                    options:MTLResourceStorageModeShared];
                if (buffer == nil) throw std::bad_alloc();
                history_buffers.push_back(buffer);
                const CacheShape cs = plan.cache_shapes[index];
                [values addObject:[[MPSGraphTensorData alloc]
                    initWithMTLBuffer:buffer shape:shape({cs.spatial, 31,
                        cs.channels}) dataType:MPSDataTypeFloat32]];
            }
        }
        MPSGraphExecutableExecutionDescriptor* execution =
            [MPSGraphExecutableExecutionDescriptor new];
        execution.waitUntilCompleted = YES;
        NSArray<MPSGraphTensorData*>* results = [plan.executable
            runWithMTLCommandQueue:queue_ inputsArray:values
            resultsArray:nil executionDescriptor:execution];
        const NSUInteger expected = plan.frames == 1 ? 9 : 1;
        if (results.count != expected)
            throw std::runtime_error("VDA Metal graph returned incomplete outputs");
        [results[0].mpsndarray readBytes:depth strideBytes:nil];
        if (plan.frames == 1) {
            for (std::size_t index = 0; index < 8; ++index) {
                const CacheShape cs = plan.cache_shapes[index];
                caches[index].resize(
                    static_cast<std::size_t>(cs.spatial) * cs.channels);
                [results[index + 1].mpsndarray
                    readBytes:caches[index].data() strideBytes:nil];
            }
        }
    }

    const ModelFile& model_;
    const ModelConfig& config_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::shared_ptr<inferbridge::native_harness::metal::AuxiliaryTensorPool>
        tensor_pool_;
    std::unordered_map<PlanKey, Plan, PlanKeyHash> plans_;
    std::vector<std::shared_ptr<StreamEntry>> cache_;
    std::int64_t stream_id_ = -1;
    std::uint32_t stream_size_ = 0;
    std::mutex mutex_;
    id<MTLComputePipelineState> preprocess_pipeline_=nil;
    id<MTLComputePipelineState> pack_pipeline_=nil;
    id<MTLComputePipelineState> resize_pipeline_=nil;
    id<MTLComputePipelineState> reduce_pipeline_=nil;
    id<MTLComputePipelineState> output_pipeline_=nil;
};

MetalExecutor::MetalExecutor(
    const ModelFile& model, const ModelConfig& config)
    : impl_(std::make_unique<Impl>(model, config)) {}
MetalExecutor::~MetalExecutor() = default;
void MetalExecutor::reset_stream() { impl_->reset_stream(); }
void MetalExecutor::infer_tensor(
    const float* input, std::uint32_t frames, std::uint32_t width,
    std::uint32_t height, float* depth) {
    impl_->infer_tensor(input, frames, width, height, depth);
}
void MetalExecutor::infer_stream(
    const float* input, std::uint32_t size, float* depth) {
    impl_->infer_stream(input, size, depth);
}
std::shared_ptr<ExternalJob> MetalExecutor::submit_texture(
    const ExternalTextureRequest& request) {
    return impl_->submit_texture(request);
}

}  // namespace vda_native
