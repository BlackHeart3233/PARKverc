import math
from PIL import  Image
import numpy as np
import os
import time




def ocena(P, X, Y):
    E=np.zeros((X,Y), dtype=int)

    for i in range(X):
        for j in range(Y):
            if i ==0 and j==0:
                predict=0
            elif i == 0:
                predict= P[i, j-1]
            elif j == 0:
                predict= P[i-1, j]
            else:
                A= P[i, j-1]
                B= P[i-1, j]
                C= P[i-1, j-1]

                if C>=max(A,B):
                    predict= min(A,B)
                elif C<=min(A,B):
                    predict= max(A,B)
                else:
                    predict = A + B - C

            E[i, j] = predict - P[i, j]

    return E.flatten()

class Bit_Writer:
    def __init__(self):
        self.bits = []

    def write(self, value, n_bits):
        for i in reversed(range(n_bits)):
            bit = (value >> i) & 1
            self.bits.append(bit)

    def to_bytes(self):
        while len(self.bits) % 8 != 0:
            self.bits.append(0)

        data = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for bit in self.bits[i:i+8]:
                byte = byte * 2 +  bit
            data.append(byte)
        return data


def glava(X, C0, Cn_1, n):
    B=bytearray()

    B+= int(X).to_bytes(2, byteorder="big", signed=False)
    B+= int(C0).to_bytes(1, byteorder="big", signed=True)
    B+= int(Cn_1).to_bytes(4, byteorder="big", signed=True)
    B+= int(n).to_bytes(4, byteorder="big", signed=False)

    return B

def IC(writer, C, L, H):
    if H - L > 1 and C[H] != C[L]:
        m = (L + H) // 2
        g = math.ceil(math.log2(C[H] - C[L] + 1))
        writer.write(C[m] - C[L], g)

        if L < m:
            IC(writer, C, L, m)
        if m < H:
            IC(writer, C, m, H)


def kompresija(P,X,Y):
    E = ocena(P, X, Y)

    n = X * Y
    N = np.zeros(len(E), dtype=int)
    N[0] = E[0]

    for i in range(1, len(E)):
        if E[i] >= 0:
            N[i] = 2 * E[i]
        else:
            N[i] = 2 * abs(E[i]) - 1


    C = np.zeros(len(N), dtype=int)
    C[0] = N[0]
    for i in range(1, len(N)):
        C[i] = C[i - 1] + N[i]

    writer = Bit_Writer()
    header = glava(X, C[0], C[-1], n)
    IC(writer, C, 0, n-1)

    bitstream = header + writer.to_bytes()
    return bitstream


def de_glava(B):
    X=int.from_bytes(B[0:2], byteorder="big", signed=False)
    C0=int.from_bytes(B[2:3], byteorder="big", signed=True)
    Cn_1=int.from_bytes(B[3:7], byteorder="big", signed=True)
    n= int.from_bytes(B[7:11], byteorder="big", signed=False)
    return X, C0, Cn_1, n, B[11:]

def delIC(reader, C, L, H):
    if H - L > 1:
        if C[H] == C[L]:
            for i in range(L + 1, H):
                C[i] = C[L]
        else:
            m = (L + H) // 2
            g = math.ceil(math.log2(C[H] - C[L] + 1))
            C[m] = C[L] + reader.read(g)

            if L < m:
                delIC(reader, C, L, m)
            if m < H:
                delIC(reader, C, m, H)

class BitReader:
    def __init__(self, data):
        self.bits = []
        for b in data:
            for i in range(7, -1, -1):
                self.bits.append((b >> i) & 1)
        self.pos = 0

    def read(self, g):
        v = 0
        for _ in range(g):
            v = (v << 1) | self.bits[self.pos]
            self.pos += 1
        return v

def predict_inverse(E, X, Y):
    P = np.zeros((X, Y), dtype=int)
    E = E.reshape((X, Y))

    for i in range(X):
        for j in range(Y):
            if i == 0 and j == 0:
                predict = 0
            elif i == 0:
                predict = P[i, j-1]
            elif j == 0:
                predict = P[i-1, j]
            else:
                A = P[i, j-1]
                B = P[i-1, j]
                C = P[i-1, j-1]

                if C >= max(A, B):
                    predict = min(A, B)
                elif C <= min(A, B):
                    predict = max(A, B)
                else:
                    predict = A + B - C

            P[i, j] = predict - E[i, j]

    return P

def dekompresija(bitstream):
    X, C0, Cn_1, n, payload = de_glava(bitstream)
    Y = n // X

    C = np.zeros(n, dtype=int)
    C[0] = C0
    C[-1] = Cn_1

    reader = BitReader(payload)
    delIC(reader, C, 0, n-1)

    N = np.zeros(len(C), dtype=int)
    N[0] = C[0]
    for i in range(1, len(C)):
        N[i] = C[i] - C[i - 1]

    E = np.zeros(len(N), dtype=int)
    for i in range(len(N)):
        if N[i] % 2 == 0:
            E[i] = N[i] // 2
        else:
            E[i] = -(N[i] + 1) // 2

    P = predict_inverse(E, X, Y)
    return P
