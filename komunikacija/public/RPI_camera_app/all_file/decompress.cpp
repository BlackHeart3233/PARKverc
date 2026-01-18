#include "decompress.h"
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using std::vector;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::int32_t;
constexpr int Q = 1;   //MORA BITI ISTI kot v kompresiji

BitReader::BitReader(const std::vector<uint8_t>& data)
    : data(data), pos(0), size(data.size() * 8) {
}

uint32_t BitReader::read(int bits) {
    if (pos + bits > size)
        throw std::runtime_error("BitReader overflow");

    uint32_t value = 0;

    while (bits >= 8 && (pos & 7) == 0) {
        value = (value << 8) | data[pos >> 3];
        pos += 8;
        bits -= 8;
    }

    while (bits > 0) {
        uint8_t byte = data[pos >> 3];
        int shift = 7 - (pos & 7);
        value = (value << 1) | ((byte >> shift) & 1);
        pos++;
        bits--;
    }

    return value;
}



Header DecodeHeader(BitReader& reader) {
    Header h;
    h.visina = reader.read(16);
    h.prvi = reader.read(8);
    h.zadnji = reader.read(32);
    h.n = reader.read(32);
    return h;
}

vector<int32_t> InitializeC(int32_t prvi, int32_t zadnji, int32_t n) {
    vector<int32_t> C(n, 0);
    C[0] = prvi;
    C[n - 1] = zadnji;
    return C;
}

void DelC(BitReader& reader, vector<int32_t>& C) {
    vector<std::pair<int, int>> st;
    st.reserve(C.size());
    st.push_back({ 0, (int)C.size() - 1 });

    while (!st.empty()) {
        auto [L, H] = st.back();
        st.pop_back();

        if (H - L <= 1)
            continue;

        if (C[H] == C[L]) {
            int32_t v = C[L];
            for (int i = L + 1; i < H; ++i)
                C[i] = v;
            continue;
        }

        int m = (L + H) >> 1;
        int range = C[H] - C[L] + 1;

        if (range == 1) {
            C[m] = C[L];
        }
        else {
            unsigned long idx;
            _BitScanReverse(&idx, range);
            int g = idx + 1;
            C[m] = C[L] + reader.read(g);
        }

        st.push_back({ L, m });
        st.push_back({ m, H });
    }
}



vector<int64_t> inv_JPG_LES(const vector<int32_t>& E, int H, int W) {
    vector<int64_t> P(H * W, 0);

    auto idx = [&](int y, int x) {
        return y * W + x; //row-major
        };

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int i = idx(y, x);

            if (x == 0 && y == 0) {
                P[i] = E[i];
            }
            else if (y == 0) {
                P[i] = P[idx(y, x - 1)] - (E[i] << Q);
            }
            else if (x == 0) {
                P[i] = P[idx(y - 1, x)] - (E[i] << Q);
            }
            else {
                int64_t A = P[idx(y, x - 1)];
                int64_t B = P[idx(y - 1, x)];
                int64_t C = P[idx(y - 1, x - 1)];

                int64_t pred;
                if (C >= A && C >= B)
                    pred = std::min(A, B);
                else if (C <= A && C <= B)
                    pred = std::max(A, B);
                else
                    pred = A + B - C;

                P[i] = pred - (E[i] << Q);
            }
        }
    }

    return P;
}

void saveBMP(const std::string& filename,
    const vector<uint8_t>& img,
    int W,
    int H) {
    int pad = (4 - (W % 4)) % 4;
    int filesize = 54 + 1024 + (W + pad) * H;

    std::ofstream f(filename, std::ios::binary);

    uint8_t bmpfileheader[14] = {
        'B','M',
        (uint8_t)(filesize),
        (uint8_t)(filesize >> 8),
        (uint8_t)(filesize >> 16),
        (uint8_t)(filesize >> 24),
        0,0,0,0,
        54 + 1024,0,0,0
    };

    uint8_t bmpinfoheader[40] = {
        40,0,0,0,
        (uint8_t)(W),
        (uint8_t)(W >> 8),
        (uint8_t)(W >> 16),
        (uint8_t)(W >> 24),
        (uint8_t)(H),
        (uint8_t)(H >> 8),
        (uint8_t)(H >> 16),
        (uint8_t)(H >> 24),
        1,0,
        8,0
    };

    f.write((char*)bmpfileheader, 14);
    f.write((char*)bmpinfoheader, 40);

    for (int i = 0; i < 256; i++) {
        uint8_t c[4] = { (uint8_t)i, (uint8_t)i, (uint8_t)i, 0 };
        f.write((char*)c, 4);
    }

    for (int y = H - 1; y >= 0; y--) {
        f.write((char*)&img[y * W], W);
        uint8_t p[3] = { 0,0,0 };
        f.write((char*)p, pad);
    }
}

std::vector<uint8_t> Decompress(BitReader& reader){
    using clock = std::chrono::high_resolution_clock;

    auto t_start = clock::now();

    Header h = DecodeHeader(reader);
    int sirina = h.n / h.visina;

    auto t0 = clock::now();

    vector<int32_t> C = InitializeC(h.prvi, h.zadnji, h.n);
    DelC(reader, C);

    auto t1 = clock::now();

    vector<int32_t> E(h.n);
    E[0] = C[0];

    for (int i = 1; i < h.n; i++) {
        int d = C[i] - C[i - 1];
        if (d < 0)
            throw std::runtime_error("Invalid C sequence (d < 0)");

        if ((d & 1) == 0)
            E[i] = d >> 1;
        else
            E[i] = -((d + 1) >> 1);
    }

    auto t2 = clock::now();

    // column-major -> row-major
    vector<int32_t> E2(h.visina * sirina);
    for (int x = 0; x < sirina; x++) {
        for (int y = 0; y < h.visina; y++) {
            E2[y * sirina + x] = E[y + x * h.visina];
        }
    }

    auto t3 = clock::now();

    vector<int64_t> P = inv_JPG_LES(E2, h.visina, sirina);

    auto t4 = clock::now();

    vector<uint8_t> img(h.n);
    for (int i = 0; i < h.n; i++)
        img[i] = (uint8_t)std::clamp<int64_t>(P[i], 0, 255);

    auto t_end = clock::now();  

    /*std::cout << "DecodeHeader + init: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t0 - t_start).count()
        << " ms\n";

    std::cout << "DelC: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
        << " ms\n";

    std::cout << "C -> E: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " ms\n";

    std::cout << "E -> E2: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
        << " ms\n";

    std::cout << "inv_JPG_LES: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
        << " ms\n";

    std::cout << "Clamp + img: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t4).count()
        << " ms\n";

    std::cout << "SKUPAJ: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()
        << " ms\n";
    saveBMP("decompressed.bmp", img, sirina, h.visina);

    cv::Mat image(h.visina, sirina, CV_8UC1, img.data());
    cv::imshow("Decompressed image", image);
    cv::waitKey(0);*/
    return img;
}

std::vector<uint8_t> DecompressFromPython(const uint8_t* data, size_t size) {

    std::vector<uint8_t> compressed(data, data + size);
    BitReader reader(compressed);

    return Decompress(reader);
}


/* ===================== main ===================== */
/*int main() {
    std::ifstream f("compressed.bin", std::ios::binary);
    if (!f) {
        std::cerr << "Ne morem odpreti compressed.bin\n";
        return 1;
    }

    vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );

    try {
        BitReader reader(data);

        auto start = std::chrono::high_resolution_clock::now();

        Decompress(reader);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Dekompresija OK\n";
        std::cout << "Cas dekompresije: "
            << elapsed.count() << " ms\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Napaka: " << e.what() << "\n";
    }


    cv::waitKey(0);
    return 0;
}*/
