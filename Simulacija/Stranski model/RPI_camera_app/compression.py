from scipy.fftpack import dct
import cv2
import numpy as np


def diagonalno_branje(matrika):
    vrstice, stolpci = matrika.shape
    rezultat = []
    for diag_indeks in range(vrstice + stolpci - 1):
        diagonala = []
        i_zacetek = max(0, diag_indeks - stolpci + 1)
        i_konec = min(vrstice, diag_indeks + 1)
        for i in range(i_zacetek, i_konec):
            j = diag_indeks - i
            diagonala.append(matrika[i][j])
        if diag_indeks % 2 == 0:
            diagonala.reverse()
        rezultat.extend(diagonala)
    return np.array(rezultat)


class BitWriter:
    """
    Incremental bit packer:
    - avoids building giant strings
    - writes bits into a bytearray with a shifting buffer
    """

    def __init__(self):
        self.out = bytearray()
        self.buf = 0
        self.nbits = 0

    def write_bits(self, value: int, n: int) -> None:
        """Write lowest n bits of value to stream (MSB-first in the stream)."""
        if n <= 0:
            return
        value &= (1 << n) - 1

        self.buf = (self.buf << n) | value
        self.nbits += n

        while self.nbits >= 8:
            shift = self.nbits - 8
            byte = (self.buf >> shift) & 0xFF
            self.out.append(byte)

            self.nbits -= 8
            if self.nbits > 0:
                self.buf &= (1 << self.nbits) - 1
            else:
                self.buf = 0

    def write_signed_twos_complement(self, value: int, n: int) -> None:
        """Write signed int in two's complement on n bits."""
        if value < 0:
            value = (1 << n) + value
        self.write_bits(value, n)

    def finish(self) -> bytes:
        """Pad with 0s to next byte boundary."""
        if self.nbits > 0:
            self.write_bits(0, 8 - self.nbits)
        return bytes(self.out)


def kodiraj_ac_koeficiente_v_pisatelj(cikcak_vektor, pisatelj: BitWriter) -> None:
    """
    Writes AC coefficients using the SAME format as your original string encoder:

    Case A (non-zero, no preceding zeros):
      '1' + len(4) + value(len)

    Case B (non-zero with preceding run of zeros):
      '0' + zeros(6) + len(4) + value(len)

    End-of-block (remaining zeros):
      '0' + zeros(6)=0
    """
    stevilo_nicel = 0

    for ac_vrednost_raw in cikcak_vektor:
        ac_vrednost = int(ac_vrednost_raw)

        if ac_vrednost == 0:
            stevilo_nicel += 1
            continue

        ac_dolzina = abs(ac_vrednost).bit_length()  # number of bits for magnitude

        if stevilo_nicel > 0:
            # 0 + zeros(6) + len(4) + value(len)
            pisatelj.write_bits(0, 1)
            pisatelj.write_bits(stevilo_nicel, 6)
            pisatelj.write_bits(ac_dolzina, 4)
            pisatelj.write_signed_twos_complement(ac_vrednost, ac_dolzina)
            stevilo_nicel = 0
        else:
            # 1 + len(4) + value(len)
            pisatelj.write_bits(1, 1)
            pisatelj.write_bits(ac_dolzina, 4)
            pisatelj.write_signed_twos_complement(ac_vrednost, ac_dolzina)

    # End-of-block marker if there are trailing zeros
    if stevilo_nicel > 0:
        pisatelj.write_bits(0, 1)
        pisatelj.write_bits(0, 6)


def stisni_sliko_bytes(slika_bgr: np.ndarray, faktor_stiskanja: int) -> bytes:
    """
    Input:  BGR numpy image (uint8)
    Output: compressed bytes
    """
    slika = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2RGB)
    slika = slika.astype(np.int16) - 128

    visina, sirina, kanali = slika.shape

    dopolnilo_v = (8 - visina % 8) % 8
    dopolnilo_s = (8 - sirina % 8) % 8
    slika_dopolnjena = np.pad(
        slika,
        ((0, dopolnilo_v), (0, dopolnilo_s), (0, 0)),
        mode='constant',
        constant_values=0
    )

    v_dopolnjena, s_dopolnjena, _ = slika_dopolnjena.shape

    pisatelj = BitWriter()
    pisatelj.write_bits(sirina, 16)
    pisatelj.write_bits(visina, 16)

    for ch in range(kanali):
        for i in range(0, v_dopolnjena, 8):
            for j in range(0, s_dopolnjena, 8):
                blok = slika_dopolnjena[i:i + 8, j:j + 8, ch]

                blok_f = blok.astype(np.float32)
                dct_blok = cv2.dct(blok_f)
                dct_blok = np.round(dct_blok).astype(np.int16)

                cikcak = diagonalno_branje(dct_blok)

                if 0 < faktor_stiskanja < 64:
                    cikcak[-faktor_stiskanja:] = 0

                pisatelj.write_signed_twos_complement(int(cikcak[0]), 12)
                kodiraj_ac_koeficiente_v_pisatelj(cikcak[1:], pisatelj)

    return pisatelj.finish()
