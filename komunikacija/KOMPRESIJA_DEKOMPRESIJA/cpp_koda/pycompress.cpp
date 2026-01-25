#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "decompress.h"

namespace py = pybind11;


/*
Deklaracija za kompresijo sivinske slike.
Funkcija sprejme surove slikovne podatke in dimenzije slike
ter vrne kompresirane podatke v obliki bajtnega vektorja.
*/

std::vector<uint8_t> compress_image_u8(
    const uint8_t* img,
    int height,
    int width
);

/*
 Deklaracija glavne dekompresijske funkcije.
 Funkcija iz BitReaderja rekonstruira originalno sliko
 in jo vrne kot vektor 8-bitnih vrednosti.
*/

std::vector<uint8_t> Decompress(BitReader& reader);

/*
 Deklaracija funkcije za dekodiranje glave kompresiranih podatkov.
 Iz bitnega toka prebere osnovne parametre slike, potrebne
 za pravilno oblikovanje izhodnega NumPy polja v Pythonu.
*/
Header DecodeHeader(BitReader& reader);

PYBIND11_MODULE(compressor, m) {
      /*
      Python funkcija compress.
      Sprejme 2D NumPy polje tipa uint8 (sivinska slika),
      preveri pravilnost oblike in tipa podatkov,
      nato pokliče funkcijo za kompresijo.
      Rezultat se vrne kot 1D NumPy polje bajtov.
     */
    m.def("compress", [](py::array_t<uint8_t> img) {
        auto buf = img.request();

        if (buf.ndim != 2)
            throw std::runtime_error("Image must be 2D grayscale");

        if (img.dtype().kind() != 'u' || img.itemsize() != 1)
            throw std::runtime_error("Image must be uint8");

        int H = static_cast<int>(buf.shape[0]);
        int W = static_cast<int>(buf.shape[1]);


        std::vector<uint8_t> compressed =
            compress_image_u8(
                static_cast<const uint8_t*>(buf.ptr),
                H,
                W
            );

        auto* data = new std::vector<uint8_t>(std::move(compressed));

        return py::array_t<uint8_t>(
            { data->size() },
            { 1 },
            data->data(),
            py::capsule(data, [](void* p) {
                delete static_cast<std::vector<uint8_t>*>(p);
                })
        );
        });

    /*
    Sprejme kompresirane podatke v obliki Python bytes objekta,
    jih pretvori v C++ vektor, izvede dekompresijo ter
    rekonstruirano sliko vrne kot 2D NumPy polje tipa uint8.
    Dimenzije slike se določijo iz dekodirane glave podatkov.
   */
    m.def("decompress", [](py::bytes input) {

        std::string buf = input;
        std::vector<uint8_t> compressed(buf.begin(), buf.end());

        BitReader reader(compressed);
        std::vector<uint8_t> img = Decompress(reader);

        BitReader r2(compressed);
        Header h = DecodeHeader(r2);

        int H = h.visina;
        int W = h.n / h.visina;

        auto* data = new std::vector<uint8_t>(std::move(img));

        return py::array_t<uint8_t>(
            { H, W },
            { W, 1 },
            data->data(),
            py::capsule(data, [](void* p) {
                delete static_cast<std::vector<uint8_t>*>(p);
                })
        );
        });
}
