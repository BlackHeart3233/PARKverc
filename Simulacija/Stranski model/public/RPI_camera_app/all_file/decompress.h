#pragma once

#include <vector>
#include <cstdint>

class BitReader {
public:
    BitReader(const std::vector<uint8_t>& data);
    uint32_t read(int bits);

private:
    const std::vector<uint8_t>& data;
    size_t pos;
    size_t size;
};

struct Header {
    int32_t visina;
    int32_t prvi;
    int32_t zadnji;
    int32_t n;
};

Header DecodeHeader(BitReader& reader);
std::vector<int32_t> InitializeC(int32_t prvi, int32_t zadnji, int32_t n);
void DelC(BitReader& reader, std::vector<int32_t>& C);
std::vector<uint8_t> Decompress(BitReader& reader);

