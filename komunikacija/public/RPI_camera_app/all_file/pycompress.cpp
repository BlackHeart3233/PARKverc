#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "decompress.h"

namespace py = pybind11;

std::vector<uint8_t> Decompress(BitReader& reader);
Header DecodeHeader(BitReader& reader);

PYBIND11_MODULE(compressor, m) {

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
