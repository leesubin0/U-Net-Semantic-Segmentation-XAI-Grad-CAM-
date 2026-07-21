import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamVidDataset
from model import UNet


DATA_DIR = r"C:\Users\IVSP\Desktop\SB\구현프로젝트\unet_practice\data\CamVid"

train_image_dir = os.path.join(DATA_DIR, "train")
train_mask_dir = os.path.join(DATA_DIR, "train_labels")

val_image_dir = os.path.join(DATA_DIR, "val")
val_mask_dir = os.path.join(DATA_DIR, "val_labels")

class_csv = os.path.join(DATA_DIR, "class_dict.csv")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 32
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-4


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


train_transform = A.Compose([
    A.Resize(352, 480),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

val_transform = A.Compose([
    A.Resize(352, 480),
])


train_dataset = CamVidDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    class_csv=class_csv,
    transform=train_transform
)

val_dataset = CamVidDataset(
    image_dir=val_image_dir,
    mask_dir=val_mask_dir,
    class_csv=class_csv,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


train_losses = []
val_losses = []
best_val_loss = float("inf")


def train_one_epoch():
    model.train()
    total_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(val_loader)


print("사용 장치:", DEVICE)
print("Epochs:", EPOCHS)
print("Learning Rate:", LR)
print("Batch Size:", BATCH_SIZE)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_loss = validate()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            model.state_dict(),
            os.path.join(RESULT_DIR, "best_model.pth")
        )
        print("Best Model Updated!")


torch.save(
    model.state_dict(),
    os.path.join(RESULT_DIR, "last_model.pth")
)

history = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
})

history.to_csv(
    os.path.join(RESULT_DIR, "loss.csv"),
    index=False
)

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("UNet CamVid Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png"), dpi=300)
plt.close()

print("학습 완료")
print("Best Val Loss:", best_val_loss)
print("저장 위치:", RESULT_DIR)