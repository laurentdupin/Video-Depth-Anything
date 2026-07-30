#include "encoder_cpu.h"
#include "model.h"

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
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid raw tensor file: " + path);
    }
    return values;
}
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: vda_encoder_probe model.vda size "
                     "input.bin reference-prefix tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const float tolerance = std::stof(argv[5]);
        const std::vector<float> input =
            read_floats(argv[3], std::uint64_t(3) * size * size);
        vda_native::ModelFile model(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES);
        const vda_native::EncoderCpuOutput output =
            vda_native::encoder_cpu(
                model, input.data(), 1, size, size);
        const std::uint64_t prepared_count =
            std::uint64_t(
                1 + output.patch_width * output.patch_height) *
            output.embedding;
        const std::uint64_t patch_count =
            std::uint64_t(output.patch_width) *
            output.patch_height * output.embedding;
        const std::vector<float> patch_reference = read_floats(
            std::string(argv[4]) + ".patch.bin", patch_count);
        double patch_absolute_sum = 0.0;
        double patch_reference_sum = 0.0;
        float patch_maximum = 0.0f;
        for (std::size_t index = 0;
             index < patch_reference.size();
             ++index) {
            const float difference = std::abs(
                output.patch_tokens[output.embedding + index] -
                patch_reference[index]);
            patch_maximum = std::max(patch_maximum, difference);
            patch_absolute_sum += difference;
            patch_reference_sum += std::abs(patch_reference[index]);
        }
        std::cout << "patch_relative_l1="
                  << patch_absolute_sum /
                      std::max(patch_reference_sum, 1.0e-30)
                  << "\npatch_maximum_absolute="
                  << patch_maximum << "\n";
        const std::vector<float> prepared_reference = read_floats(
            std::string(argv[4]) + ".prepared.bin", prepared_count);
        double prepared_absolute_sum = 0.0;
        double prepared_reference_sum = 0.0;
        float prepared_maximum = 0.0f;
        for (std::size_t index = 0;
             index < prepared_reference.size();
             ++index) {
            const float difference = std::abs(
                output.prepared_tokens[index] -
                prepared_reference[index]);
            prepared_maximum = std::max(
                prepared_maximum, difference);
            prepared_absolute_sum += difference;
            prepared_reference_sum += std::abs(
                prepared_reference[index]);
        }
        std::cout << "prepared_relative_l1="
                  << prepared_absolute_sum /
                      std::max(prepared_reference_sum, 1.0e-30)
                  << "\nprepared_maximum_absolute="
                  << prepared_maximum << "\n";
        const std::uint64_t feature_count =
            std::uint64_t(output.patch_width) *
            output.patch_height * output.embedding;
        double worst_relative = 0.0;
        float worst_absolute = 0.0f;
        for (std::uint32_t feature = 0; feature < 4; ++feature) {
            const std::vector<float> reference = read_floats(
                std::string(argv[4]) + ".feature" +
                    std::to_string(feature) + ".bin",
                feature_count);
            double absolute_sum = 0.0;
            double reference_sum = 0.0;
            float maximum = 0.0f;
            for (std::size_t index = 0;
                 index < reference.size();
                 ++index) {
                const float difference = std::abs(
                    output.features[feature][index] - reference[index]);
                maximum = std::max(maximum, difference);
                absolute_sum += difference;
                reference_sum += std::abs(reference[index]);
            }
            const double relative =
                absolute_sum / std::max(reference_sum, 1.0e-30);
            worst_relative = std::max(worst_relative, relative);
            worst_absolute = std::max(worst_absolute, maximum);
            std::cout << "feature" << feature
                      << "_relative_l1=" << relative
                      << "\nfeature" << feature
                      << "_maximum_absolute=" << maximum << "\n";
        }
        std::cout << "worst_relative_l1=" << worst_relative
                  << "\nworst_maximum_absolute=" << worst_absolute
                  << "\n";
        return worst_relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
