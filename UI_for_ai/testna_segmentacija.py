import numpy as np
import matplotlib.pyplot as plt
import cv2

def izostri_sliko(slika_rgb):
    # Če je slika normalizirana (0..1), pretvori v 0..255 uint8
    if slika_rgb.max() <= 1.0:
        slika = (slika_rgb * 255).astype(np.uint8)
    else:
        slika = slika_rgb.astype(np.uint8)

    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    izostrena = np.zeros_like(slika)
    for i in range(3):
        izostrena[:, :, i] = cv2.filter2D(slika[:, :, i], -1, kernel)

    # Nazaj v 0..1 float za prikaz
    return izostrena / 255.0

def segmentacijaSivinskeSlike(slika_rgb):
    izostrena_slika = izostri_sliko(slika_rgb)

    # Sivinska slika iz izostrene
    if izostrena_slika.max() <= 1.0:
        slika_siva = (izostrena_slika.mean(axis=2) * 255).astype(np.uint8)
    else:
        slika_siva = izostrena_slika.mean(axis=2).astype(np.uint8)

    hist, bins = np.histogram(slika_siva, bins=200, range=(0, 256))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(slika_rgb)
    ax[0].set_title('Originalna barvna slika')
    ax[0].axis('off')

    ax[1].imshow(izostrena_slika)
    ax[1].set_title('Izostrjena barvna slika')
    ax[1].axis('off')

    ax[2].plot(bins[:-1], hist)
    ax[2].set_title('Histogram sivinske slike (izostrjene)')
    ax[2].set_xlabel('Sivina')
    ax[2].set_ylabel('Število pikslov')
    plt.tight_layout()
    plt.show()
    
    # Segmentacija
    prag_min = 150
    prag_max = 200
    objekt = (slika_siva > prag_min) & (slika_siva < prag_max)

    # Priprava RGB maske (rdeča barva)
    maska_rgb = np.zeros_like(slika_rgb, dtype=np.uint8)
    maska_rgb[objekt] = [255, 0, 0]

    # Originalna slika v 0..255 uint8
    if slika_rgb.max() <= 1.0:
        slika_rgb_255 = (slika_rgb * 255).astype(np.uint8)
    else:
        slika_rgb_255 = slika_rgb.astype(np.uint8)

    # Združevanje originala in maske (alpha blending)
    alpha = 0.4
    kombinirana = cv2.addWeighted(slika_rgb_255, 1.0, maska_rgb, alpha, 0)

    return kombinirana

if __name__ == "__main__":
    img_path = "UI_for_ai/test.png"
    slika_rgb = plt.imread(img_path)
    if slika_rgb.shape[-1] == 4:
        slika_rgb = slika_rgb[:, :, :3] 

    rezultat = segmentacijaSivinskeSlike(slika_rgb)

    # Prikaz rezultata
    plt.figure(figsize=(6,6))
    plt.imshow(rezultat)
    plt.title('Segmentirani objekt na originalni sliki (z masko)')
    plt.axis('off')
    plt.show()

    # Shrani rezultat (v RGB -> BGR za OpenCV)
    cv2.imwrite("rezultat_segmentacije.png", cv2.cvtColor(rezultat, cv2.COLOR_RGB2BGR))
