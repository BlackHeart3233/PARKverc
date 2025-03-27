import albumentations as A
import cv2
import os

input_folder = 'results/video1'
output_folder = 'augmentacija/video1'
os.makedirs(output_folder, exist_ok=True)

# transformacija
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1.0),
], seed=137, strict=True)

# za vsako sliko v input_folder naredi 3 transformacije
for filename in os.listdir(input_folder):

    # zaenkrat samo slike
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):

        input_path = os.path.join(input_folder, filename)

        # naloži image v RGB
        image = cv2.imread(input_path, cv2.IMREAD_COLOR_RGB)
        if image is None:
            print(f"Preskčimo {filename}")
            continue

        for i in range(3):
            # naredi transformacijo
            augmented = transform(image=image)['image']

            # shrani image v BGR zaradi cv2
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

            output_path = os.path.join(output_folder, (filename + f"_aug_{i}." + filename.split('.')[-1]))
            cv2.imwrite(output_path, augmented_bgr)

            print(f"Saved: {output_path}_aug_{i}.jpg")