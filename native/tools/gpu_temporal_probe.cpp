#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "temporal_gpu.h"
#include "vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<float> read_floats(
    const std::string& path,
    std::uint64_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input ||
        input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid tensor file: " + path);
    }
    return values;
}
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr
            << "usage: vda_gpu_temporal_probe model device module "
               "channels frames height width input reference tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t device = std::stoul(argv[2]);
        const std::uint32_t module = std::stoul(argv[3]);
        const std::uint32_t channels = std::stoul(argv[4]);
        const std::uint32_t frames = std::stoul(argv[5]);
        const std::uint32_t height = std::stoul(argv[6]);
        const std::uint32_t width = std::stoul(argv[7]);
        const double tolerance = std::stod(argv[10]);
        const std::uint64_t spatial =
            std::uint64_t(height) * width;
        const std::uint64_t count =
            std::uint64_t(channels) * frames * spatial;
        const std::vector<float> input_cfhw =
            read_floats(argv[8], count);
        const std::vector<float> reference =
            read_floats(argv[9], count);
        std::vector<float> input_fchw(
            static_cast<std::size_t>(count));
        for (std::uint32_t channel = 0;
             channel < channels; ++channel) {
            for (std::uint32_t frame = 0;
                 frame < frames; ++frame) {
                std::copy_n(
                    input_cfhw.data() +
                        (std::uint64_t(channel) * frames + frame) *
                            spatial,
                    spatial,
                    input_fchw.data() +
                        (std::uint64_t(frame) * channels + channel) *
                            spatial);
            }
        }
        vda_native::ModelFile model_file(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES);
        vda_native::VulkanContext context(device);
        vda_native::GpuModel weights(model_file, context);
        vda_native::VulkanOperators operators(context);
        vda_native::TemporalGpu temporal(
            context, weights, operators);
        vda_native::FeatureMap input{
            context.create_device_buffer(count * sizeof(float)),
            width, height, channels, frames};
        context.upload(
            input.buffer, input_fchw.data(), count * sizeof(float));
        vda_native::FeatureMap output =
            temporal.forward(module, std::move(input));
        std::vector<float> output_fchw(
            static_cast<std::size_t>(count));
        context.download(
            output.buffer, output_fchw.data(), count * sizeof(float));
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::uint32_t channel = 0;
             channel < channels; ++channel) {
            for (std::uint32_t frame = 0;
                 frame < frames; ++frame) {
                for (std::uint64_t position = 0;
                     position < spatial; ++position) {
                    const std::uint64_t reference_index =
                        (std::uint64_t(channel) * frames + frame) *
                            spatial +
                        position;
                    const std::uint64_t output_index =
                        (std::uint64_t(frame) * channels + channel) *
                            spatial +
                        position;
                    const float difference = std::abs(
                        output_fchw[output_index] -
                        reference[reference_index]);
                    absolute_sum += difference;
                    reference_sum +=
                        std::abs(reference[reference_index]);
                    maximum = std::max(maximum, difference);
                }
            }
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "relative_l1=" << relative
                  << "\nmaximum_absolute=" << maximum << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
