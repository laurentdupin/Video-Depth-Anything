#include "video_depth_anything_native.h"

#include <algorithm>
#include <chrono>
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
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: vda_gpu_full_graph_probe model.vda device frames size "
                     "input.bin reference.bin tolerance [iterations]\n";
        return 2;
    }
    try {
        const std::uint32_t frames =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[4]));
        const float tolerance = std::stof(argv[7]);
        const std::uint32_t iterations = argc == 9
            ? static_cast<std::uint32_t>(std::stoul(argv[8]))
            : 1;
        if (iterations == 0) {
            throw std::invalid_argument("iterations must be positive");
        }
        const std::uint64_t input_elements =
            std::uint64_t(frames) * 3 * size * size;
        const std::vector<float> input =
            std::string(argv[5]) == "-"
            ? std::vector<float>(
                  static_cast<std::size_t>(input_elements), 0.0f)
            : read_floats(argv[5], input_elements);
        const std::vector<float> reference =
            std::string(argv[6]) == "-"
            ? std::vector<float>{}
            : read_floats(
                  argv[6], std::uint64_t(frames) * size * size);
        vda_context* context = nullptr;
        const vda_status create_status = vda_create_vulkan(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES,
            static_cast<uint32_t>(std::stoul(argv[2])), &context);
        if (create_status != VDA_STATUS_OK) {
            throw std::runtime_error(vda_last_error());
        }
        std::vector<float> output(
            std::uint64_t(frames) * size * size);
        std::vector<double> timings;
        timings.reserve(iterations);
        vda_status infer_status = VDA_STATUS_OK;
        for (std::uint32_t iteration = 0;
             iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            infer_status = vda_infer_tensor_f32(
                context, input.data(),
                static_cast<int32_t>(frames),
                static_cast<int32_t>(size),
                static_cast<int32_t>(size),
                output.data(), output.size());
            timings.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count());
            if (infer_status != VDA_STATUS_OK) {
                break;
            }
        }
        vda_destroy(context);
        if (infer_status != VDA_STATUS_OK) {
            throw std::runtime_error(vda_last_error());
        }
        std::sort(timings.begin(), timings.end());
        const double seconds =
            timings[timings.size() / 2];
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        if (reference.empty()) {
            std::cout << "iterations=" << timings.size()
                      << "\nmedian_seconds=" << seconds
                      << "\noutput_elements=" << output.size()
                      << "\n";
            return 0;
        }
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[index]);
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "iterations=" << timings.size()
                  << "\nmedian_seconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
