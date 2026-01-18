#include "decompress.h"
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>

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

    return img;
}

std::vector<uint8_t> DecompressFromPython(const uint8_t* data, size_t size) {

    std::vector<uint8_t> compressed(data, data + size);
    BitReader reader(compressed);

    return Decompress(reader);
}

