from scipy.fftpack import dct
import cv2
import numpy as np


def inverzno_diagonalno_branje(vektor):
    matrika = np.zeros((8, 8), dtype=np.int16)
    vrstice, stolpci = 8, 8
    indeks_vektorja = 0

    for diag_indeks in range(vrstice + stolpci - 1):
        diagonala_koordinate = []
        i_zacetek = max(0, diag_indeks - stolpci + 1)
        i_konec = min(vrstice, diag_indeks + 1)

        for i in range(i_zacetek, i_konec):
            j = diag_indeks - i
            diagonala_koordinate.append((i, j))

        if diag_indeks % 2 == 0:
            diagonala_koordinate.reverse()

        for i, j in diagonala_koordinate:
            if indeks_vektorja < len(vektor):
                matrika[i, j] = vektor[indeks_vektorja]
                indeks_vektorja += 1
            else:
                break

    return matrika


def dekodiraj_ac_koeficiente(bralnik):
    cikcak_vektor = []
    stevec_vrednosti = 0

    while stevec_vrednosti < 63:
        prvi_bit = bralnik.preberi_bite(1)

        if prvi_bit == '1':
            dolzina = bralnik.preberi_int(4)
            if dolzina == 0:
                ac_vrednost = 0
            else:
                ac_vrednost = bralnik.preberi_int(dolzina, predznaceno=True)
            cikcak_vektor.append(ac_vrednost)
            stevec_vrednosti += 1
        else:
            stevilo_nicel = bralnik.preberi_int(6)

            if stevilo_nicel == 0:
                cikcak_vektor.extend([0] * (63 - stevec_vrednosti))
                stevec_vrednosti = 63
            else:
                stevec_vrednosti += stevilo_nicel
                if stevec_vrednosti < 63:
                    dolzina = bralnik.preberi_int(4)
                    if dolzina == 0:
                        ac_vrednost = 0
                    else:
                        ac_vrednost = bralnik.preberi_int(dolzina, predznaceno=True)
                    cikcak_vektor.extend([0] * stevilo_nicel)
                    cikcak_vektor.append(ac_vrednost)
                    stevec_vrednosti += 1
                else:
                    cikcak_vektor.extend([0] * (stevilo_nicel))

    return np.array(cikcak_vektor[:63])


class BralnikBitov:
    def __init__(self, podatki_bytes):
        self.biti = ''.join(f'{b:08b}' for b in podatki_bytes)
        self.polozaj = 0

    def preberi_bite(self, n):
        if self.polozaj + n > len(self.biti):
            return ''
        vrednost = self.biti[self.polozaj:self.polozaj + n]
        self.polozaj += n
        return vrednost

    def preberi_int(self, n, predznaceno=False):
        biti = self.preberi_bite(n)
        if not predznaceno:
            return int(biti, 2)
        if biti[0] == '0':
            return int(biti, 2)
        return int(biti, 2) - (1 << n)


def razsiri_sliko_bytes(podatki: bytes) -> np.ndarray:
    """
    Input:  compressed bytes
    Output: BGR numpy image (uint8)
    """
    if len(podatki) < 4:
        return None

    sirina = int.from_bytes(podatki[0:2], byteorder='big')
    visina = int.from_bytes(podatki[2:4], byteorder='big')
    podatki_brez_glave = podatki[4:]

    dopolnilo_v = (8 - visina % 8) % 8
    dopolnilo_s = (8 - sirina % 8) % 8
    v_dopolnjena = visina + dopolnilo_v
    s_dopolnjena = sirina + dopolnilo_s

    slika_rekonstruirana = np.zeros((v_dopolnjena, s_dopolnjena, 3), dtype=np.uint8)
    bralnik = BralnikBitov(podatki_brez_glave)

    for ch in range(3):
        for i in range(0, v_dopolnjena, 8):
            for j in range(0, s_dopolnjena, 8):
                dc_koeficient = bralnik.preberi_int(12, predznaceno=True)
                ac_koeficienti = dekodiraj_ac_koeficiente(bralnik)

                cikcak = np.concatenate([[dc_koeficient], ac_koeficienti])
                blok_dct = inverzno_diagonalno_branje(cikcak)

                blok_idct = dct(dct(blok_dct.T, norm='ortho').T, norm='ortho', type=3)
                blok_idct = np.round(blok_idct + 128)
                blok_idct = np.clip(blok_idct, 0, 255).astype(np.uint8)

                slika_rekonstruirana[i:i + 8, j:j + 8, ch] = blok_idct

    slika_koncna = slika_rekonstruirana[:visina, :sirina, :]
    return cv2.cvtColor(slika_koncna, cv2.COLOR_RGB2BGR)
