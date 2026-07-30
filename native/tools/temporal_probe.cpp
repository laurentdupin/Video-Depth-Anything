#include "model.h"
#include "temporal_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
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
    if (argc != 9) {
        std::cerr << "usage: vda_temporal_probe model.vda module "
                     "frames height width input.bin reference.bin tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t module =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t frames =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const std::uint32_t height =
            static_cast<std::uint32_t>(std::stoul(argv[4]));
        const std::uint32_t width =
            static_cast<std::uint32_t>(std::stoul(argv[5]));
        const float tolerance = std::stof(argv[8]);
        const std::uint32_t module_channels[4] = {192, 384, 64, 64};
        if (module >= 4) {
            throw std::invalid_argument("invalid temporal module");
        }
        const std::uint64_t count =
            std::uint64_t(module_channels[module]) * frames *
            height * width;
        const std::vector<float> input = read_floats(argv[6], count);
        const std::vector<float> reference = read_floats(argv[7], count);
        vda_native::ModelFile model(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES);
        std::vector<float> output;
        vda_native::temporal_module_cpu(
            model, module, input.data(), module_channels[module],
            frames, height, width, output);
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[index]);
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "maximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
