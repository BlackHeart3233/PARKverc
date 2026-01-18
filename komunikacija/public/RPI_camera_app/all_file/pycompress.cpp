#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "decompress.h"

namespace py = pybind11;

// C++ funkcije
std::vector<uint8_t> compress_image_u8(
    const uint8_t* img,
    int height,
    int width
);

std::vector<uint8_t> Decompress(BitReader& reader);
Header DecodeHeader(BitReader& reader);

PYBIND11_MODULE(compressor, m) {

    // =============================
    // compress(img) -> np.ndarray uint8 (1D)
    // =============================
    m.def("compress", [](py::array_t<uint8_t> img) {
        auto buf = img.request();

        // ---- validacija ----
        if (buf.ndim != 2)
            throw std::runtime_error("Image must be 2D grayscale");

        if (img.dtype().kind() != 'u' || img.itemsize() != 1)
            throw std::runtime_error("Image must be uint8");

        int H = static_cast<int>(buf.shape[0]);
        int W = static_cast<int>(buf.shape[1]);

        // ---- klic C++ kompresije ----
        std::vector<uint8_t> compressed =
            compress_image_u8(
                static_cast<const uint8_t*>(buf.ptr),
                H,
                W
            );

        // ---- brez kopije: Python prevzame ownership ----
        auto* data = new std::vector<uint8_t>(std::move(compressed));

        return py::array_t<uint8_t>(
            { data->size() },          // shape
            { 1 },                     // stride
            data->data(),              // pointer
            py::capsule(data, [](void* p) {
                delete static_cast<std::vector<uint8_t>*>(p);
                })
        );
        });

    // =============================
    // decompress(bytes) -> np.ndarray uint8 (H, W)
    // =============================
    m.def("decompress", [](py::bytes input) {

        // ---- bytes -> vector<uint8_t> ----
        std::string buf = input;
        std::vector<uint8_t> compressed(buf.begin(), buf.end());

        // ---- dekodiranje ----
        BitReader reader(compressed);
        std::vector<uint8_t> img = Decompress(reader);

        // ---- header za dimenzije ----
        BitReader r2(compressed);
        Header h = DecodeHeader(r2);

        int H = h.visina;
        int W = h.n / h.visina;

        // ---- brez kopije ----
        auto* data = new std::vector<uint8_t>(std::move(img));

        return py::array_t<uint8_t>(
            { H, W },                  // shape
            { W, 1 },                  // stride (row-major)
            data->data(),
            py::capsule(data, [](void* p) {
                delete static_cast<std::vector<uint8_t>*>(p);
                })
        );
        });
}
