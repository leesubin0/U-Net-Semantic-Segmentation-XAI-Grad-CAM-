import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

from dataset import CamVidDataset
from model import UNet


DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"
MODEL_PATH = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\results\best_model.pth"

test_image_dir = os.path.join(DATA_DIR, "test")
test_mask_dir = os.path.join(DATA_DIR, "test_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 32


def class_to_rgb(class_mask, class_rgb):
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, rgb in enumerate(class_rgb):
        rgb_mask[class_mask == class_idx] = rgb

    return rgb_mask


class_df = pd.read_csv(class_csv)
class_names = class_df["name"].tolist()
class_rgb = class_df[["r", "g", "b"]].values

transform = A.Compose([
    A.Resize(352, 480),
])

test_dataset = CamVidDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir,
    class_csv=class_csv,
    transform=transform
)

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

image, gt_mask = test_dataset[0]

with torch.no_grad():
    input_image = image.unsqueeze(0).to(DEVICE)
    output = model(input_image)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

image_np = image.permute(1, 2, 0).cpu().numpy()
gt_mask_np = gt_mask.cpu().numpy()

gt_rgb = class_to_rgb(gt_mask_np, class_rgb)
pred_rgb = class_to_rgb(pred_mask, class_rgb)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(gt_rgb)
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(pred_rgb)
axes[2].set_title("Prediction")
axes[2].axis("off")


def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    if x < 0 or x >= pred_mask.shape[1] or y < 0 or y >= pred_mask.shape[0]:
        return

    gt_idx = int(gt_mask_np[y, x])
    pred_idx = int(pred_mask[y, x])

    gt_name = class_names[gt_idx]
    pred_name = class_names[pred_idx]

    print("=" * 50)
    print(f"Clicked Pixel: x={x}, y={y}")
    print(f"Ground Truth : {gt_idx} - {gt_name}")
    print(f"Prediction   : {pred_idx} - {pred_name}")

    if gt_idx == pred_idx:
        print("Result       : Correct")
    else:
        print("Result       : Wrong")

    print("=" * 50)


fig.canvas.mpl_connect("button_press_event", onclick)

plt.tight_layout()
plt.show()