import os
import cv2
import torch
import numpy as np
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

# CamVid class index 기준
TARGET_CLASS = 17   # Road


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


activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Grad-CAM을 적용할 layer
target_layer = model.enc4

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


image, gt_mask = test_dataset[0]

input_image = image.unsqueeze(0).to(DEVICE)
input_image.requires_grad = True

output = model(input_image)

pred_mask = torch.argmax(output, dim=1).squeeze(0).detach().cpu().numpy()

# Road class score
# Road로 예측된 영역만 평균내서 Grad-CAM 생성
class_score_map = output[:, TARGET_CLASS, :, :]

road_region = (torch.argmax(output, dim=1) == TARGET_CLASS)

if road_region.sum() > 0:
    score = class_score_map[road_region].mean()
else:
    score = class_score_map.mean()

model.zero_grad()
score.backward()


# Grad-CAM 계산
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1).squeeze()

cam = torch.relu(cam)
cam = cam.detach().cpu().numpy()

cam = cv2.resize(cam, (480, 352))

cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)

image_np = image.permute(1, 2, 0).cpu().numpy()
image_uint8 = (image_np * 255).astype(np.uint8)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(image_uint8, 0.6, heatmap, 0.4, 0)


plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(pred_mask, cmap="tab20")
plt.title("Prediction Mask")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cam, cmap="jet")
plt.title("Grad-CAM for Road Class")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(overlay)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_road.png", dpi=300)
plt.show()

print("gradcam_road.png 저장 완료")