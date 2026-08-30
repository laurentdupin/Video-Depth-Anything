#include "video_depth_anything_native.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: vda_stream_probe model.vda kind size frames output.bin\n";
        return 2;
    }
    try {
        const auto kind = static_cast<vda_model_kind>(std::stoul(argv[2]));
        const std::uint32_t size = std::stoul(argv[3]);
        const std::uint32_t frames = std::stoul(argv[4]);
        vda_context* context = nullptr;
        vda_status status = vda_create_vulkan(argv[1], kind, 0, &context);
        if (status != VDA_STATUS_OK) throw std::runtime_error(vda_last_error());
        std::vector<std::uint8_t> bgra(
            static_cast<std::size_t>(size) * size * 4);
        std::vector<float> depth(
            static_cast<std::size_t>(frames) * size * size);
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            for (std::uint32_t y = 0; y < size; ++y) {
                for (std::uint32_t x = 0; x < size; ++x) {
                    const std::size_t pixel =
                        (static_cast<std::size_t>(y) * size + x) * 4;
                    bgra[pixel + 0] = static_cast<std::uint8_t>(
                        (x * 7 + y * 3 + frame * 11) & 255);
                    bgra[pixel + 1] = static_cast<std::uint8_t>(
                        (x * 2 + y * 9 + frame * 5) & 255);
                    bgra[pixel + 2] = static_cast<std::uint8_t>(
                        (x * 5 + y * 4 + frame * 13) & 255);
                    bgra[pixel + 3] = 255;
                }
            }
            status = vda_infer_stream_bgra8_f32(
                context, bgra.data(), static_cast<std::uint64_t>(size) * 4,
                size, size, size,
                depth.data() + static_cast<std::size_t>(frame) * size * size,
                static_cast<std::uint64_t>(size) * size);
            if (status != VDA_STATUS_OK) {
                const std::string message = vda_last_error();
                vda_destroy(context);
                throw std::runtime_error(message);
            }
        }
        vda_destroy(context);
        std::ofstream output(argv[5], std::ios::binary);
        output.write(reinterpret_cast<const char*>(depth.data()),
                     static_cast<std::streamsize>(depth.size() * sizeof(float)));
        if (!output) throw std::runtime_error("could not write probe output");
        std::cout << "frames=" << frames << " elements=" << depth.size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
