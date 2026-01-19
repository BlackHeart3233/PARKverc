from PIL import Image
import numpy as np
import math
from bitstring import BitArray


P = np.array([
    [23, 21, 21, 23, 23],
    [24, 22, 22, 20, 24],
    [23, 22, 22, 19, 23],
    [26, 25, 21, 19, 22]
], dtype=np.int32)

#print(P.shape)      # (H, W)
#print(P.dtype)      # int32

def JPG_LES(img):
    X,Y = img.shape #visina in sirina
    E = np.zeros((X,Y), dtype=np.int32)
    for y in range(X):
        for x in range(Y):
            if x == 0 and y == 0:
                E[y, x] = img[y, x]
            elif y == 0:
                E[y, x] = img[y, x-1] - img[y, x]
            elif x == 0:
                E[y, x] = img[y-1, x] - img[y, x]
            else:
                A = img[y, x-1]
                B = img[y-1, x]
                C = img[y-1, x-1]
                if C >= max(A, B):
                    pred = min(A, B)
                elif C <= min(A, B):
                    pred = max(A, B)
                else:
                    pred = A + B - C

                E[y, x] = pred - img[y, x]

    return E

def prepletanje(E):
    e = E.flatten(order='F') #po stolpcih
    n = e.size
    N = np.zeros(n, dtype=np.int32)
    N[0] = e[0]
    for i in range(1,n):
        if(e[i] >=0):
            N[i] = e[i] * 2
        else:
            N[i] = (2 * abs(e[i])) - 1
    return N

def komutativna_vsota(N):
    c = N.size
    C = np.zeros(c, dtype=np.int32)
    C[0] = N[0]
    for i in range(1,c):
        C[i] = C[i-1] + N[i]
    return C

def SetHeader(visina, C):
    B = BitArray()
    B.append(BitArray(int=visina, length=16))     # 0–15
    B.append(BitArray(int=C[0], length=16))       # 16–31
    B.append(BitArray(int=C[-1], length=32))      # 32–63
    B.append(BitArray(int=C.size, length=32))     # 64–95
    return B


def Encode(B, g, value):
    B.append(BitArray(uint=value, length=g))

def IC(B, C, L, H):
    if H - L <= 1:
        return

    if C[H] != C[L]:
        m = math.floor(0.5 * (L + H)) #zaokrozi dol
        g = math.ceil(math.log2(C[H] - C[L] + 1)) #zaokrozi gor
        Encode(B, g, C[m] - C[L])
        if L < m:
            IC(B, C, L, m)
        if m < H:
            IC(B, C, m, H)


def kompresija(P):
    visina,sirina = P.shape
    E = JPG_LES(P)
    N = prepletanje(E)
    C = komutativna_vsota(N)
    B = SetHeader(visina,C)
    IC(B,C,0,C.size-1)
    return B

