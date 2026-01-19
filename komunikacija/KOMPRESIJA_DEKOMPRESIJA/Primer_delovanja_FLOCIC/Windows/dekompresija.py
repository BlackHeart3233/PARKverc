from PIL import Image
import numpy as np
import math
from numba import njit
from bitstring import BitArray

class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.size = len(data) * 8

    def read(self, bits):
        if self.pos + bits > self.size:
            raise RuntimeError("BitReader overflow")

        value = 0
        while bits >= 8 and (self.pos & 7) == 0:
            value = (value << 8) | self.data[self.pos >> 3]
            self.pos += 8
            bits -= 8

        while bits > 0:
            byte = self.data[self.pos >> 3]
            shift = 7 - (self.pos & 7)
            value = (value << 1) | ((byte >> shift) & 1)
            self.pos += 1
            bits -= 1

        return value




def DecodeHeader(reader):
    visina = reader.read(16)
    prvi = reader.read(8)
    zadnji = reader.read(32)
    n = reader.read(32)
    return visina, prvi, zadnji, n


def InitalizeC(prvi_element,zadnji_element,n):
    C = np.zeros(n, dtype=np.int32)
    C[0] = prvi_element
    C[-1] = zadnji_element
    return C

def DelC(reader, C):
    stack = [(0, C.size - 1)]
    while stack:
        L, H = stack.pop()
        if H - L <= 1:
            continue
        if C[H] == C[L]:
            for i in range(L + 1, H):
                C[i] = C[L]
            continue

        m = (L + H) >> 1
        g = int(C[H] - C[L] + 1).bit_length()
        C[m] = C[L] + reader.read(g)

        stack.append((L, m))
        stack.append((m, H))




@njit(cache=True)
def inv_JPG_LES(E):
    X, Y = E.shape
    P = np.zeros((X, Y), dtype=np.int32)

    for y in range(X):
        for x in range(Y):
            if x == 0 and y == 0:
                P[y, x] = E[y, x]
            elif y == 0:
                P[y, x] = P[y, x-1] - E[y, x]
            elif x == 0:
                P[y, x] = P[y-1, x] - E[y, x]
            else:
                A = P[y, x-1]
                B = P[y-1, x]
                C = P[y-1, x-1]

                if C >= A and C >= B:
                    pred = A if A < B else B
                elif C <= A and C <= B:
                    pred = A if A > B else B
                else:
                    pred = A + B - C

                P[y, x] = pred - E[y, x]
    return P




def Decompress(reader):
    visina, prvi_element, zadnji_element, n = DecodeHeader(reader)
    sirina = n // visina

    C = InitalizeC(prvi_element, zadnji_element, n)
    DelC(reader, C)

    E = np.empty(n, dtype=np.int32)
    E[0] = C[0]

    for i in range(1, n):
        d = C[i] - C[i - 1]
        if (d & 1) == 0:
            E[i] = d >> 1
        else:
            E[i] = -(d + 1) >> 1

    E = E.reshape((visina, sirina), order='F')
    P = inv_JPG_LES(E)

    P = np.clip(P, 0, 255).astype(np.uint8)
    Image.fromarray(P, mode='L').save("decompressed.bmp")






