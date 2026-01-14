from PIL import Image
import numpy as np
import math
from bitstring import BitArray

def DecodeHeader(B):
    visina = B[0:16].int              # 0–15
    prvi_element = B[16:32].int       # 16–31
    zadnji_element = B[32:64].int     # 32–63
    stevilo_elementov = B[64:96].int  # 64–95
    return visina, prvi_element, zadnji_element, stevilo_elementov

def InitalizeC(prvi_element,zadnji_element,n):
    C = np.zeros(n, dtype=np.int32)
    C[0] = prvi_element
    C[-1] = zadnji_element
    return C

def Decode(B, g, offset):
    value = B[offset : offset + g].uint
    return value, offset + g


def DelC(B, C, L, H, offset):
    if H - L <= 1:
        return offset

    if C[H] == C[L]:
        for i in range(L + 1, H):
            C[i] = C[L]
        return offset

    m = math.floor(0.5 * (L + H))
    g = math.ceil(math.log2(C[H] - C[L] + 1))

    value, offset = Decode(B, g, offset)
    C[m] = C[L] + value

    if L < m:
        offset = DelC(B, C, L, m, offset)
    if m < H:
        offset = DelC(B, C, m, H, offset)

    return offset

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

                if C >= max(A, B):
                    pred = min(A, B)
                elif C <= min(A, B):
                    pred = max(A, B)
                else:
                    pred = A + B - C
                P[y, x] = pred - E[y, x]
    return P



def Decompress(B):
    visina, prvi_element, zadnji_element, n = DecodeHeader(B)
    sirina = n // visina
    C = InitalizeC(prvi_element, zadnji_element, n)
    offset = 96  #header porabi 88 bitov
    offset = DelC(B, C, 0, n - 1, offset)

    N = np.zeros(n, dtype=np.int32)
    N[0] = C[0]
    for i in range(1,n):
        N[i] = C[i]-C[i-1]

    E = np.zeros(n, dtype=np.int32)
    E[0] = N[0]

    for i in range(1, n):
        if N[i] % 2 == 0:
            E[i] = N[i] // 2
        else:
            E[i] = -(N[i] + 1) // 2 #// pomeni celostevilcno cene bi bila float

    E = E.reshape((visina, sirina), order='F')
    P = inv_JPG_LES(E)
    P = np.clip(P, 0, 255).astype(np.uint8)
    return P




