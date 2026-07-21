import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"

image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")

print("이미지 폴더:", image_dir)
print("마스크 폴더:", mask_dir)
print("클래스 CSV:", class_csv)

print("train 이미지 개수:", len(os.listdir(image_dir)))
print("train label 개수:", len(os.listdir(mask_dir)))

class_df = pd.read_csv(class_csv)
print("\nclass_dict.csv 내용:")
print(class_df)

image_name = os.listdir(image_dir)[0]
mask_name = os.listdir(mask_dir)[0]

image_path = os.path.join(image_dir, image_name)
mask_path = os.path.join(mask_dir, mask_name)

print("\n선택된 이미지:", image_name)
print("선택된 마스크:", mask_name)

image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

print("image shape:", image.shape)
print("mask shape:", mask.shape)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()