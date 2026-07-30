#include "model.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: vda_model_probe model.vda\n";
        return 2;
    }
    try {
        vda_native::ModelFile model(
            argv[1], VDA_MODEL_VITS_RELATIVE_32_FRAMES);
        std::cout << "tensors=" << model.tensor_count() << "\nsha256=";
        for (std::uint8_t byte :
             model.derivation().canonical_sha256) {
            std::cout << std::hex << std::setw(2)
                      << std::setfill('0')
                      << static_cast<unsigned>(byte);
        }
        std::cout << "\nconverter="
                  << model.derivation().converter << "\n";
        return model.tensor_count() == 351 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
