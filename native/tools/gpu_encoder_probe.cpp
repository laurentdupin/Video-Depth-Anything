#include "encoder_gpu.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
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
    if (argc != 7) {
        std::cerr
            << "usage: vda_gpu_encoder_probe model.vda device size "
               "input.bin reference-prefix tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t device =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const double tolerance = std::stod(argv[6]);
        const std::vector<float> input = read_floats(
            argv[4], std::uint64_t(3) * size * size);
        vda_native::ModelFile model_file(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES);
        vda_native::VulkanContext context(device);
        vda_native::GpuModel weights(model_file, context);
        vda_native::VulkanOperators operators(context);
        vda_native::VdaGpuEncoder encoder(
            context, weights, operators);
        vda_native::VulkanBuffer image =
            context.create_device_buffer(input.size() * sizeof(float));
        context.upload(
            image, input.data(), input.size() * sizeof(float));
        vda_native::EncoderOutput output =
            encoder.forward(image, 1, size, size);
        const std::uint64_t patches =
            std::uint64_t(output.patch_width) * output.patch_height;
        const std::uint64_t feature_elements =
            patches * output.embedding;
        double worst_relative = 0.0;
        float worst_maximum = 0.0f;
        for (std::uint32_t level = 0; level < 4; ++level) {
            std::vector<float> captured(
                static_cast<std::size_t>(
                    (patches + 1) * output.embedding));
            context.download(
                output.features[level], captured.data(),
                captured.size() * sizeof(float));
            const std::vector<float> reference = read_floats(
                std::string(argv[5]) + ".feature" +
                    std::to_string(level) + ".bin",
                feature_elements);
            double absolute_sum = 0.0;
            double reference_sum = 0.0;
            float maximum = 0.0f;
            for (std::uint64_t index = 0;
                 index < feature_elements; ++index) {
                const float difference = std::abs(
                    captured[
                        static_cast<std::size_t>(
                            output.embedding + index)] -
                    reference[static_cast<std::size_t>(index)]);
                absolute_sum += difference;
                reference_sum += std::abs(
                    reference[static_cast<std::size_t>(index)]);
                maximum = std::max(maximum, difference);
            }
            const double relative =
                absolute_sum /
                std::max(reference_sum, 1.0e-30);
            worst_relative = std::max(worst_relative, relative);
            worst_maximum = std::max(worst_maximum, maximum);
            std::cout << "feature" << level
                      << "_relative_l1=" << relative
                      << "\nfeature" << level
                      << "_maximum_absolute=" << maximum << "\n";
        }
        std::cout << "worst_relative_l1=" << worst_relative
                  << "\nworst_maximum_absolute=" << worst_maximum
                  << "\n";
        return worst_relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
