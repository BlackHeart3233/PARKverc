import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from pathlib import Path

def transform(slika_path,savePath):
    if not os.path.exists(slika_path):
        print(f"Napaka: Datoteka ne obstaja: {slika_path}")
        return

    #(RGB, dtype=float32 ali uint8)
    image = plt.imread(slika_path)
    filename = os.path.basename(slika_path)
    #Če ima float vrednosti (0.0–1.0), pretvori v uint8 (0–255)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    #alfa kanal (RGBA) -> odstrani
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    x = random.randint(0, 2)
    if x == 0:
        #Poudarjene črte
        transform = A.Compose([
            A.CLAHE(clip_limit=8.0, tile_grid_size=(8, 8), p=1.0),  #izboljša kontrast
            A.Sharpen(alpha=(0.5, 1.0), lightness=(1.0, 1.5), p=1.0),  #izostrimo robove belih črt
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(0.6, 1.0), p=1.0)  #povečamo svetlost in kontrast
        ])
    elif x == 1:
        #dodajanje dežja in meglice
        transform = A.Compose([
            A.RandomFog(p=1.0), #dodamo "fog" z vrjetnostjo 100%
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.3, p=0.7), #povečamo svetlost in kontrast
            # brightness_limit=0.15 lahko svetlost naraste ali pade do 15%.
            # contrast_limit=0.3 lahko kontrast naraste ali pade do 30%.
            # p=0.7 transformacija uporabi v 70% primerih.
            A.GaussianBlur(blur_limit=(2, 5), p=0.2) #zamegljenost (blur)
            #(2, 5) velikost zamegljevalnega jedra naključno izbrana med 2 in 5.
            # p=0.5 uporabi v 20% primerih.
        ])
    else:
        #tranformiramo na nizko kakovost kako bi recimo izgledala ob kompresiji
        transform = A.Compose([
            A.ImageCompression(quality_range=(10, 15), p=1.0),
            #zelo nizka kakovost med 10 in 15 (100 = brez izgube)
            # p=1.0 vrjetnost transformacije 100%
        ])


    for i in range(5):
        transformed = transform(image=image)
        transformed_image = transformed["image"]   # izhodna slika je shranjena pod ključem 'image'
        #plt.figure()
        #plt.imshow (transformed_image)


    '''plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Originalna slika')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Poudarjene rumene črte')
    plt.imshow(transformed_image)
    plt.axis('off')
    plt.show()'''

    return {'image': transformed_image}
    #save_path = os.path.join(savePath, filename)
    #plt.imsave(save_path, transformed_image)


def main(jpg_files, savePath):
    for i in range(10):
        chosenFile = random.choice(jpg_files)
        agumenPic = transform(str(chosenFile), savePath)
        save_path = os.path.join(savePath, chosenFile.name)
        plt.imsave(save_path, agumenPic['image'])
    print("Augmentacija je končana 😄")


if __name__=="__main__":
    inputPath = Path("images_from_video/Video_007_25_4_2025")
    outputPath = "./augmented_images"
    os.makedirs(outputPath, exist_ok=True)
    jpg_files = [f for f in inputPath.iterdir()  #pogledamo koliko slik je v tej mapi
             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']]
    main(jpg_files, outputPath)

