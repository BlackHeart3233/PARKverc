#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <utility>
#include <intrin.h>   
#include <iostream>
#include <chrono>
#include "decompress.h"
#include <opencv2/opencv.hpp>

constexpr int Q = 1;   //MORA BITI ISTI kot v kompresiji


class BitWriter {
public:
    void write(uint32_t value, int bits) {
        while (bits > 0) {
            int free = 8 - nbits;
            int take = std::min(bits, free);
            uint32_t shift = bits - take;

            cur = (cur << take) |
                ((value >> shift) & ((1u << take) - 1));

            nbits += take;
            bits -= take;

            if (nbits == 8) {
                buf.push_back(cur);
                cur = 0;
                nbits = 0;
            }
        }
    }

    std::vector<uint8_t> finish() {
        if (nbits > 0) {
            buf.push_back(cur << (8 - nbits));
        }
        return buf;
    }


private:
    std::vector<uint8_t> buf;
    uint8_t cur = 0;
    int nbits = 0;
};

void WriteHeader(BitWriter& w, int32_t visina, const std::vector<int32_t>& C) {
    w.write(visina, 16);
    w.write(C[0] & 0xFF, 8);
    w.write(C.back(), 32);
    w.write((uint32_t)C.size(), 32);
}

void IC(BitWriter& writer, const std::vector<int32_t>& C) {
    std::vector<std::pair<int, int>> st;
    st.reserve(C.size());
    st.push_back({ 0, (int)C.size() - 1 });

    while (!st.empty()) {
        auto [L, H] = st.back();
        st.pop_back();

        if (H - L <= 1)
            continue;

        if (C[H] == C[L])
            continue;

        int m = (L + H) >> 1;
        int range = C[H] - C[L] + 1;

        unsigned long idx;
        _BitScanReverse(&idx, range);   
        int g = idx + 1;

        writer.write((uint32_t)(C[m] - C[L]), g);

        // isti vrstni red
        st.push_back({ L, m });
        st.push_back({ m, H });
    }
}
std::vector<uint8_t> Compress(const std::vector<int32_t>& P, int H, int W) {
    // 1. predikcija
    std::vector<int32_t> E(H * W);
    auto idx = [&](int y, int x) { return y * W + x; };

    E[0] = P[0];

    for (int x = 1; x < W; x++) {
        int d = P[x - 1] - P[x];
        E[x] = (d >= 0)
            ? ((d + (1 << (Q - 1))) >> Q)
            : -(((-d + (1 << (Q - 1))) >> Q));
    }

    for (int y = 1; y < H; y++) {
        int d = P[(y - 1) * W] - P[y * W];
        E[y * W] = (d >= 0)
            ? ((d + (1 << (Q - 1))) >> Q)
            : -(((-d + (1 << (Q - 1))) >> Q));
    }

    for (int y = 1; y < H; y++) {
        for (int x = 1; x < W; x++) {
            int A = P[idx(y, x - 1)];
            int B = P[idx(y - 1, x)];
            int C = P[idx(y - 1, x - 1)];

            int pred;
            if (C >= A && C >= B) pred = std::min(A, B);
            else if (C <= A && C <= B) pred = std::max(A, B);
            else pred = A + B - C;

            int d = pred - P[idx(y, x)];
            E[idx(y, x)] = (d >= 0)
                ? ((d + (1 << (Q - 1))) >> Q)
                : -(((-d + (1 << (Q - 1))) >> Q));
        }
    }

    // 2. stolpčno prepletanje + vsota
    std::vector<int32_t> Cvec(H * W);
    int k = 0;
    int32_t s = 0;

    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            int v = E[y * W + x];
            if (k == 0) {
                s = v;
            }
            else {
                if (v >= 0) s += v << 1;
                else s += ((-v) << 1) - 1;
            }
            Cvec[k++] = s;
        }
    }

    // 3. zapis → V RAM
    BitWriter writer;
    WriteHeader(writer, H, Cvec);
    IC(writer, Cvec);

    return writer.finish();   // ✅ KLJUČNA VRSTICA
}


std::vector<uint8_t> compress_image_u8(
    const uint8_t* img,
    int height,
    int width
) {
    std::vector<int32_t> P(height * width);
    for (int i = 0; i < height * width; i++) {
        P[i] = img[i];
    }

    return Compress(P, height, width);
}



/*
int main() {
    cv::Mat img = cv::imread("Man.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Napaka: slike ni mogoce odpreti\n";
        return 1;
    }

    if (!img.isContinuous()) {
        img = img.clone();
    }

    int H = img.rows;
    int W = img.cols;

    std::vector<int32_t> P(H * W);

    std::transform(
        img.data,
        img.data + H * W,
        P.begin(),
        [](uint8_t v) { return (int32_t)v; }
    );

    //KOMPRESIJA 
    //auto t0 = std::chrono::high_resolution_clock::now();
    Compress(P, H, W);
    //auto t1 = std::chrono::high_resolution_clock::now();

    //DEKOMPRESIJA
    std::ifstream f("compressed.bin", std::ios::binary);
    std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );

    BitReader reader(data);

    //auto t2 = std::chrono::high_resolution_clock::now();
    Decompress(reader);
    /*auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "Kompresija:   "
        << std::chrono::duration<double, std::milli>(t1 - t0).count()
        << " ms\n";

    std::cout << "Dekompresija:"
        << std::chrono::duration<double, std::milli>(t3 - t2).count()
        << " ms\n";

    return 0;
}
*/

