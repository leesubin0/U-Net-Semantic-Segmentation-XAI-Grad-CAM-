import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"

image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")

# class_dict.csv 읽기
class_df = pd.read_csv(class_csv)

print(class_df)

# RGB 색상표 만들기
class_names = class_df["name"].tolist()
class_rgb = class_df[["r", "g", "b"]].values

print("\n클래스 개수:", len(class_names))
print("클래스 이름:", class_names)

def rgb_to_class(mask_rgb, class_rgb):
    """
    RGB 마스크 이미지를 class index 마스크로 변환한다.
    예: Road 색상 픽셀 → 3
    """
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for class_idx, rgb in enumerate(class_rgb):
        match = np.all(mask_rgb == rgb, axis=-1)
        class_mask[match] = class_idx

    return class_mask

# 이미지/마스크 하나 선택
image_name = sorted(os.listdir(image_dir))[0]
mask_name = sorted(os.listdir(mask_dir))[0]

image_path = os.path.join(image_dir, image_name)
mask_path = os.path.join(mask_dir, mask_name)

image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

class_mask = rgb_to_class(mask, class_rgb)

print("\nimage shape:", image.shape)
print("mask RGB shape:", mask.shape)
print("class mask shape:", class_mask.shape)

print("class mask unique values:", np.unique(class_mask))

# 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("RGB Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(class_mask, cmap="tab20")
plt.title("Class Index Mask")
plt.axis("off")

plt.tight_layout()
plt.show()