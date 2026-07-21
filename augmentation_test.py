import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A

DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"

image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")

image_name = sorted(os.listdir(image_dir))[0]
mask_name = sorted(os.listdir(mask_dir))[0]

image = cv2.imread(os.path.join(image_dir, image_name))
mask = cv2.imread(os.path.join(mask_dir, mask_name))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.HorizontalFlip(p=1.0),
])

augmented = transform(image=image, mask=mask)

aug_image = augmented["image"]
aug_mask = augmented["mask"]

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(mask)
plt.title("Original Mask")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(aug_image)
plt.title("Flipped Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(aug_mask)
plt.title("Flipped Mask")
plt.axis("off")

plt.tight_layout()
plt.show()