import os
import torch
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamVidDataset
from model import UNet


DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"
MODEL_PATH = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\results\best_model.pth"

test_image_dir = os.path.join(DATA_DIR, "test")
test_mask_dir = os.path.join(DATA_DIR, "test_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 32
BATCH_SIZE = 2

transform = A.Compose([
    A.Resize(352, 480),
])

test_dataset = CamVidDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir,
    class_csv=class_csv,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def calculate_iou_and_dice(pred, target, num_classes):
    ious = []
    dices = []

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        pred_sum = pred_cls.sum()
        target_sum = target_cls.sum()

        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union

        if pred_sum + target_sum == 0:
            dice = np.nan
        else:
            dice = (2 * intersection) / (pred_sum + target_sum)

        ious.append(iou)
        dices.append(dice)

    return ious, dices


total_ious = []
total_dices = []

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        ious, dices = calculate_iou_and_dice(preds, masks, NUM_CLASSES)

        total_ious.append(ious)
        total_dices.append(dices)

mean_ious = np.nanmean(np.array(total_ious), axis=0)
mean_dices = np.nanmean(np.array(total_dices), axis=0)

print("\nClass-wise IoU / Dice")
print("-" * 40)

for idx, class_name in enumerate(test_dataset.class_names):
    print(f"{idx:02d} {class_name:20s} IoU: {mean_ious[idx]:.4f} | Dice: {mean_dices[idx]:.4f}")

print("-" * 40)
print(f"mIoU:  {np.nanmean(mean_ious):.4f}")
print(f"mDice: {np.nanmean(mean_dices):.4f}")