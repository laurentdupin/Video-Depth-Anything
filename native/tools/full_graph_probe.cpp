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
    if (argc != 7) {
        std::cerr << "usage: vda_full_graph_probe model.vda frames size "
                     "input.bin reference.bin tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t frames =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const float tolerance = std::stof(argv[6]);
        const std::vector<float> input = read_floats(
            argv[4], std::uint64_t(frames) * 3 * size * size);
        const std::vector<float> reference = read_floats(
            argv[5], std::uint64_t(frames) * size * size);
        vda_context* context = nullptr;
        const vda_status create_status = vda_create(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES, &context);
        if (create_status != VDA_STATUS_OK) {
            throw std::runtime_error(vda_last_error());
        }
        const auto start = std::chrono::steady_clock::now();
        std::vector<float> output(
            std::uint64_t(frames) * size * size);
        const vda_status infer_status = vda_infer_tensor_f32(
            context, input.data(),
            static_cast<int32_t>(frames),
            static_cast<int32_t>(size),
            static_cast<int32_t>(size),
            output.data(), output.size());
        vda_destroy(context);
        if (infer_status != VDA_STATUS_OK) {
            throw std::runtime_error(vda_last_error());
        }
        const double seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
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
        std::cout << "seconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
