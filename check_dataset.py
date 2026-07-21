import os
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader

from dataset import CamVidDataset


DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"

image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

dataset = CamVidDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    class_csv=class_csv,
    transform=transform
)

print("데이터 개수:", len(dataset))
print("클래스 개수:", len(dataset.class_names))
print("클래스 이름:", dataset.class_names)

image, mask = dataset[0]

print("image shape:", image.shape)
print("mask shape:", mask.shape)
print("mask unique values:", mask.unique())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title("Image Tensor")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="tab20")
plt.title("Mask Tensor")
plt.axis("off")

plt.tight_layout()
plt.show()


loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(loader))

print("batch images shape:", images.shape)
print("batch masks shape:", masks.shape)