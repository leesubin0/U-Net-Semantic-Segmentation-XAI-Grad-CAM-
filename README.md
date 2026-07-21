# U-Net-Semantic-Segmentation-and-XAI-using-Grad-CAM
Paper: 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
[Grad-CAM - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2017)](https://velog.io/@keepbini366/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Grad-CAM-Grad-CAM-Visual-Explanations-from-Deep-Networks-via-Gradient-based-Localization-2017)

Paper Review: 
[U-Net - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://velog.io/@keepbini366/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-U-Net-U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation-2015)
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

---
## Semantic Segmentation
자율주행 인지 분야 中 객체 탐지(YOLO, Faster R-CNN 등)뿐만 아니라 Semantic Segmentation도 함께 공부해 보기로 함. 

![](https://velog.velcdn.com/images/keepbini366/post/211a4b20-14e5-419e-9993-a0c1cc13ff5d/image.png)

Object Detection(객체 탐지)에서는 차량 위치, 보행자 위치 기반으로 **Bounding Box**를 출력하는데,
Semantic Segmentation에서는 모든 픽셀을 **Class**로 분류함.

그렇기에 자율주행 분야에서 차량이 어디를 주행할 수 있는지를 판단하거나, 주행 도로 환경 이해를 하기 위해 **Semantic Segmentation**이 중요한 역할을 하는 것.

그리고 이번 실습에서는 Segmentation 모델인 **UNet**을 이용하고자 함. 

### U-Net Architecture

![](https://velog.velcdn.com/images/keepbini366/post/3b92beb7-7dba-4763-86ec-a5d2f2aaf36b/image.png)

---
### Dataset
> ### CamVid Dataset
: 입문용 데이터셋으로 가장 많이 사용되는 Dataset
[CamVid Dataset](https://www.kaggle.com/datasets/carlolepelaars/camvid)
- 데이터 크기가 크지 않아서 학습에 용이
- 다양한 도로 환경 포함
- 차량, 도로, 보행자 등 여러 클래스 제공
- Semantic Segmentation 연구에서 활용 多

### Dataset 구성
- 367 training pairs
- 101 validation pairs
- 233 test pairs

![](https://velog.velcdn.com/images/keepbini366/post/569cf9a1-a22f-4409-8430-726e83d7769d/image.png)

| 폴더             | 설명                     |
| -------------- | ----------------------- |
| train          | 학습 이미지                |
| train_labels   | 학습용 Ground Truth        |
| val            | Validation 이미지          |
| val_labels     | Validation Ground Truth |
| test           | 테스트 이미지               |
| test_labels    | 테스트 Ground Truth        |
| class_dict.csv | 클래스 & RGB 색상 정보      |


![](https://velog.velcdn.com/images/keepbini366/post/3e0e2983-d032-4ebc-a740-12b33ef63e70/image.png)

---

### 환경 구축
Anaconda 환경 새롭게 만들었음.
```
conda create -n unet_camvid python=3.10 -y
conda activate unet_camvid
```

필요한 라이브러리 설치도 진행함. 
```
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install pandas
pip install albumentations
pip install tqdm
```
---

### Data 확인
U-Net 학습 전, 데이터가 정상적으로 읽히는지 확인하는 것이 가장 중요하기에, 
OpenCV로 이미지를 읽고, BGR을 RGB로 변환함. 
- 원본 이미지를 Matplotlib에서 올바른 색으로 표시하기 위해
- Semantic Segmentation의 라벨(RGB 클래스 정보)을 정확하게 유지하기 위해
```
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
```
그리고 이미지, label 읽기, class_dict 확인, 이미지 출력 진행

> #### check_camvid.py
```
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
>
DATA_DIR = r"C:\Users\IVSP\unet_practice\data\CamVid"
>
image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")
>
print("이미지 폴더:", image_dir)
print("마스크 폴더:", mask_dir)
print("클래스 CSV:", class_csv)
>
print("train 이미지 개수:", len(os.listdir(image_dir)))
print("train label 개수:", len(os.listdir(mask_dir)))
>
class_df = pd.read_csv(class_csv)
print("\nclass_dict.csv 내용:")
print(class_df)
>
image_name = os.listdir(image_dir)[0]
mask_name = os.listdir(mask_dir)[0]
>
image_path = os.path.join(image_dir, image_name)
mask_path = os.path.join(mask_dir, mask_name)
>
print("\n선택된 이미지:", image_name)
print("선택된 마스크:", mask_name)
>
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)
>
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
>
print("image shape:", image.shape)
print("mask shape:", mask.shape)
>
plt.figure(figsize=(12, 5))
>
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
>
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title("Ground Truth Mask")
plt.axis("off")
>
plt.tight_layout()
plt.show()
```

출력 결과는 하단과 같음.
- 좌측: 원본
- 우측: Ground Truth Label

![](https://velog.velcdn.com/images/keepbini366/post/dc8fd977-8bcf-4e26-b148-b219aa1c7ce8/image.png)
- 보라색 → 차량
- 연보라색 → 도로
- 회색 → 하늘 등등 색상별 각각의 클래스 의미

그리고 RGB 이미지가 Class 번호를, 그리고 U-Net의 입력되는 과정을 거쳐서 모델이 학습하게 되는 것. 

### U-Net 모델 돌리기 전, 데이터 이해를 위해 진행했던 과정
### Semantic Segmentation에서는 Bounding Box가 아니라 모든 픽셀을 분류한다는 것 = Object Detection과 가장 큰 차이

---
### RGB Mask를 Class Index로 변환

Ground Truth 이미지 내 각 색이 하나의 클래스를 의미함. 

그리고 Semantic Segmentation 모델은 RGB 이미지를 직접 학습하는 게 아니라,
RGB를 Class Index로 변환한 후 학습함.

그래서 하단 코드 기반 출력 결과에서
Matplotlib는 이 Index 숫자를 보기 쉽게 하기 위해 tab20 컬러맵을 적용하여 임의의 색으로 표현한 것(가장 우측)

> #### convert_mask.py
```
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
>
DATA_DIR = r"C:\Users\IVSP\unet_practice\data\CamVid"
>
image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")
>
# class_dict.csv 읽기
class_df = pd.read_csv(class_csv)
>
print(class_df)
>
# RGB 색상표 만들기
class_names = class_df["name"].tolist()
class_rgb = class_df[["r", "g", "b"]].values
>
print("\n클래스 개수:", len(class_names))
print("클래스 이름:", class_names)
>
def rgb_to_class(mask_rgb, class_rgb):
    """
    RGB 마스크 이미지를 class index 마스크로 변환한다.
    예: Road 색상 픽셀 → 3
    """
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
>
    for class_idx, rgb in enumerate(class_rgb):
        match = np.all(mask_rgb == rgb, axis=-1)
        class_mask[match] = class_idx
>
    return class_mask
>
# 이미지/마스크 하나 선택
image_name = sorted(os.listdir(image_dir))[0]
mask_name = sorted(os.listdir(mask_dir))[0]
>
image_path = os.path.join(image_dir, image_name)
mask_path = os.path.join(mask_dir, mask_name)
>
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)
>
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
>
class_mask = rgb_to_class(mask, class_rgb)
>
print("\nimage shape:", image.shape)
print("mask RGB shape:", mask.shape)
print("class mask shape:", class_mask.shape)
>
print("class mask unique values:", np.unique(class_mask))
>
# 시각화
plt.figure(figsize=(15, 5))
>
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
>
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("RGB Ground Truth Mask")
plt.axis("off")
>
plt.subplot(1, 3, 3)
plt.imshow(class_mask, cmap="tab20")
plt.title("Class Index Mask")
plt.axis("off")
>
plt.tight_layout()
plt.show()
```

![](https://velog.velcdn.com/images/keepbini366/post/f47dd5ad-65fa-4327-9975-cb6f9cc9571e/image.png)

---
### Augmentation 진행

자율주행 환경은 항상 동일하지 않음. 
학습 데이터만으로 모든 상황을 수집하는 건 어렵기 때문에, 
기존 데이터를 변형해서 다양한 환경 만들고 모델의 일반화 성능을 높이는 것. 
그게 **Data Augmentation**

Semantic Segmentation에서는 이미지와 마스크를 동시에 변환하기 위해 **`Albumentations`** 라이브러리를 많이 사용함. 

그리고 하단 코드가 가장 기본적인 **Horizontal Flip**을 적용한 모습임. 

> #### augmentation_test.py
```
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
>
DATA_DIR = r"C:\Users\IVSP\unet_practice\data\CamVid"
>
image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
>
image_name = sorted(os.listdir(image_dir))[0]
mask_name = sorted(os.listdir(mask_dir))[0]
>
image = cv2.imread(os.path.join(image_dir, image_name))
mask = cv2.imread(os.path.join(mask_dir, mask_name))
>
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
>
transform = A.Compose([
    A.HorizontalFlip(p=1.0),
])
>
augmented = transform(image=image, mask=mask)
>
aug_image = augmented["image"]
aug_mask = augmented["mask"]
>
plt.figure(figsize=(12,8))
>
plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
>
plt.subplot(2,2,2)
plt.imshow(mask)
plt.title("Original Mask")
plt.axis("off")
>
plt.subplot(2,2,3)
plt.imshow(aug_image)
plt.title("Flipped Image")
plt.axis("off")
>
plt.subplot(2,2,4)
plt.imshow(aug_mask)
plt.title("Flipped Mask")
plt.axis("off")
>
plt.tight_layout()
plt.show()
```

결과적으로 원본 이미지와 Ground Truth가 모두 동일하게 좌우 반전된 것을 확인할 수 있음. 

![](https://velog.velcdn.com/images/keepbini366/post/7b0fd065-f3bd-4b0f-a212-7f8761f1739e/image.png)

실제 많이 쓰는 기법
- A.HorizontalFlip()
- A.RandomBrightnessContrast()
- A.Rotate(limit=10)
- A.GaussianBlur()
- A.RandomFog()
- A.RandomRain()
- A.MotionBlur()

결론적으로 

>**Data Augmentation**
: 데이터만 늘리는 과정이 아니라, 다양한 주행 환경을 모사해서 모델의 일반화 성능을 높이는 과정

---
### Dataset 클래스 구현

> #### dataset.py
```
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
>
class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_csv, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
>
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
>
        class_df = pd.read_csv(class_csv)
        self.class_rgb = class_df[["r", "g", "b"]].values
        self.class_names = class_df["name"].tolist()
>
    def __len__(self):
        return len(self.image_names)
>
    def rgb_to_class(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)
>
        for class_idx, rgb in enumerate(self.class_rgb):
            match = np.all(mask_rgb == rgb, axis=-1)
            class_mask[match] = class_idx
>
        return class_mask
>
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        mask_name = self.mask_names[idx]
>
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
>
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
>
        mask = self.rgb_to_class(mask)
>
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
>
        image = image.astype(np.float32) / 255.0
>
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
>
        return image, mask
```


> #### check_dataset.py
```
import os
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
>
from dataset import CamVidDataset
>
>
DATA_DIR = r"C:\Users\IVSP\unet_practice\data\CamVid"
>
image_dir = os.path.join(DATA_DIR, "train")
mask_dir = os.path.join(DATA_DIR, "train_labels")
class_csv = os.path.join(DATA_DIR, "class_dict.csv")
>
>
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])
>
dataset = CamVidDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    class_csv=class_csv,
    transform=transform
)
>
print("데이터 개수:", len(dataset))
print("클래스 개수:", len(dataset.class_names))
print("클래스 이름:", dataset.class_names)
>
image, mask = dataset[0]
>
print("image shape:", image.shape)
print("mask shape:", mask.shape)
print("mask unique values:", mask.unique())
>
plt.figure(figsize=(10, 4))
>
plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title("Image Tensor")
plt.axis("off")
>
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="tab20")
plt.title("Mask Tensor")
plt.axis("off")
>
plt.tight_layout()
plt.show()
>
>
loader = DataLoader(dataset, batch_size=4, shuffle=True)
>
images, masks = next(iter(loader))
>
print("batch images shape:", images.shape)
print("batch masks shape:", masks.shape)
```

그리고 `check_dataset.py` 돌리면, 
>
```
(unet_camvid) PS C:\Users\IVSP\unet_practice> python check_dataset.py      
데이터 개수: 369
클래스 개수: 32
클래스 이름: ['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall']
image shape: torch.Size([3, 720, 960])
mask shape: torch.Size([720, 960])
mask unique values: tensor([ 2,  4,  5,  8, 10, 12, 16, 17, 19, 21, 22, 24, 26, 30, 31])
batch images shape: torch.Size([4, 3, 720, 960])
batch masks shape: torch.Size([4, 720, 960])
```
이렇게 나옴.

---
### U-Net 모델 구현

데이터셋 준비가 끝났으므로 이제 실제 U-Net 모델을 구현하겠음. 

U-Net은 크게 **Encoder, Decoder, Skip Connection**으로 구성되어 있음. 

![](https://velog.velcdn.com/images/keepbini366/post/f24c7d35-2bfd-4722-b012-17f60b122f6a/image.png)


#### [Encoder]
- Encoder에서는 Convolution과 MaxPooling을 반복하면서 Feature를 추출함. 
- 해당 과정에서 이미지 크기는 점점 줄어들고, Feature Channel은 증가하게 됨. 
- ex. ```352×480``` > ```176×240``` > ```88×120``` > ```44×60```처럼 해상도는 감소하지만, 모델은 더 높은 수준의 semantic feature를 학습하게 되는 것. 

#### [Decoder]
- Encoder의 반대 과정.
- Upsampling을 이용해서 Feature Map을 다시 원래 크기로 복원하는 과정임. 
- 최종적으로 Input 이미지랑 같은 크기의 Segmentation Mask를 출력하게 됨. 

#### [Skip Connection]
- U-Net에서의 핵심 구조라고 할 수 있음.
- Encoder에서 추출한 Feature Map을 Decoder로 직접 전달해서, Encoder 과정 당시 Pooling을 진행하면서 손실되었던 위치 정보와 경계 정보를 복원하는 과정임. 
- 따라서 차량이나 차선같이 작은 객체도 보다 정확하게 분할할 수 있음. 

> #### model.py
```
import torch
import torch.nn as nn
import torch.nn.functional as F
>
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
>
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
>
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
>
    def forward(self, x):
        return self.conv(x)
>
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=32):
        super().__init__()
>
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
>
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
>
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
>
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
>
        self.bottleneck = DoubleConv(512, 1024)
>
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
>
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
>
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
>
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
>
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
>
    def pad_to_match(self, decoder_feature, encoder_feature):
        diffY = encoder_feature.size(2) - decoder_feature.size(2)
        diffX = encoder_feature.size(3) - decoder_feature.size(3)
>
        decoder_feature = F.pad(
            decoder_feature,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2
            ]
        )
>
        return decoder_feature
>
    def forward(self, x):
        e1 = self.enc1(x)
>
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
>
        b = self.bottleneck(self.pool4(e4))
>
        d4 = self.up4(b)
        d4 = self.pad_to_match(d4, e4)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
>
        d3 = self.up3(d4)
        d3 = self.pad_to_match(d3, e3)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
>
        d2 = self.up2(d3)
        d2 = self.pad_to_match(d2, e2)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
>
        d1 = self.up1(d2)
        d1 = self.pad_to_match(d1, e1)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
>
        return self.out(d1)
```

---

### Image Resize
처음에는 ``CamVid`` 데이터셋의 원본 이미지 크기인 ```720 X 960``` 그대로 학습을 진행하려고 했음. 
다만, 원본 해상도를 그대로 쓰기에는 연산량,  GPU 메모리 사용량이 크기에 학습 속도가 좀 느려질 수 있을 것이라 판단해, 우선적으로 이미지 크기를 줄여서 학습을 진행해 보기로 했음. 

따라서, ``720 X 960`` 사이즈를 ``360 X 480``으로 Resize해서 진행했음. 

다만, 이렇게 진행하니, 학습 과정에서 하단과 같은 오류가 발생했음. 
>
```
RuntimeError:
Sizes of tensors must match
Expected size 44 but got size 45
```

U-Net의 경우, MaxPooling을 여러 번 수행하기 때문에, 
``output = floor(input / 2)``에 기반해 ``360``이라는 Input size가 계속 2로 나눠지면서, 
``360`` > ``180`` > ``90`` > ``45`` > ``22.5`` 이렇게 마지막에 정수가 아닌 size가 나오게 됨. 

근데 이때 MaxPooling이 출력 크기를 정수로 계산하면서 소수점 이하를 버려 버리기에, 
``floor(22.5) = 22``가 됨. 

그리고 Decoder에서 Feature Map 복원 시에, ``Transposed Convolution``으로 2배 확대하기 때문에, ``22``라는 size가 ``44``가 됨. 
이렇게 되면, ``Encoder Feature: 45, Decoder Feature: 44``으로 1픽셀 차이가 발생하게 됨. 

> H_out = floor((H_in + 2P - D(K-1) - 1) / S + 1)
(현재 설정: K = 2, S = 2, P = 0, D = 1)

#### => Padding 적용
따라서, 이 문제를 해결하기 위해 Decoder Feature Map에 Zero Padding 적용 진행. 

Padding을 통해 Decoder Feature Map의 크기를 Encoder Feature Map과 동일하게 맞춘 후,
Skip Connection을 수행하도록 수정했음. 

또한 입력 이미지의 크기도 352 × 480으로 변경해서,
Pooling 이후에도 모든 Feature Map의 크기가 일정하게 유지되도록 수정했음.

이후에는 Shape Mismatch 문제가 발생하지 않았고,
정상적으로 학습을 진행할 수 있었음.

---
### Training
우선 이번 학습 과정에서는 Epoch 값 차이만 두고(5 or 50) 진행했음, 

### i) Epoch 5 based Training
> ### 초기 학습 조건
- Epoch: 5
- Batch Size: 2
- Learning Rate: 1e-4
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Input Size: 352×480

> ### 로그
```
Epoch [1/5] Train Loss: 2.0059 | Val Loss: 1.7390
Epoch [2/5] Train Loss: 1.3876 | Val Loss: 1.3112
Epoch [3/5] Train Loss: 1.1290 | Val Loss: 0.9962
Epoch [4/5] Train Loss: 0.9744 | Val Loss: 0.8683
Epoch [5/5] Train Loss: 0.8883 | Val Loss: 0.8317
```

Train Loss와 Validation Loss가 모두 지속적으로 감소하면서 모델이 정상적으로 학습되고 있음을 확인했음.
다만 Epoch가 5로 짧았기 때문에 작은 객체나 경계 영역에서는 충분한 성능을 확보하지 못했을 가능성이 있다고 판단했음. 


![](https://velog.velcdn.com/images/keepbini366/post/19f7fdff-4f02-4265-a03a-b0d31458ae6c/image.png)

### Prediction 결과 확인

![](https://velog.velcdn.com/images/keepbini366/post/c3ffee51-cdc6-45d7-9329-f1982e7f2352/image.png)

Road, Sky, Building과 같은 큰 객체는 비교적 잘 분할하는 것을 확인할 수 있었으나, 
Pole, Traffic Light, Pedestrian 같은 크기가 작은 객체는 제대로 예측하지 못하는 경우 존재. 

또한 차량의 경계가 Ground Truth보다 매우 흐리게 예측되는 것도 확인 가능. 

Prediction 결과만 보면 모델이 전체적으로 잘 동작하는 것처럼 보이지만,
어떤 부분에서 틀렸는지는 정확히 알기 어려움. 

그래서 다음으로 Difference Map을 생성하여 오분류 영역을 시각적으로 분석해봄.

---

### Difference Map 결과 확인
Prediction 이미지만으로는 모델이 정확히 어느 영역에서 오분류를 발생시켰는지 직관적으로 파악하기 어렵기에, 
Ground Truth와 Prediction을 비교하여 서로 다른 픽셀을 시각화하는 Difference Map을 생성함. 

Difference Map은 모델이 틀린 픽셀만을 표시하기 때문에, 오분류가 집중적으로 발생하는 위치를 쉽게 확인할 수 있다.

![](https://velog.velcdn.com/images/keepbini366/post/e7acfb7e-c8fb-4ac6-9943-4b8a0e491822/image.png)

결과를 보자면, 
- **왼쪽 건물**
거의 빨간색 X
→ Building segmentation 거의 성공
- **Sky**
대부분 일치
Boundary만 조금 틀리는 수준(다소 정상)
- **Road**
대부분 일치
Road <-> Sidewalk 경계에서 오류 多 (다소 정상)
- **Vehicle**
버스, 자동차, SUV 경계 뭉개
(epoch 5라서 발생하는 문제라고 생각)
- **Traffic Light**
거의 못 맞춤
epoch 문제도 있지만, 352 × 480으로 Resize해서 신호등이 거의 5~10 pixel밖에 안 되어서 그런 듯

---

### Grad-CAM 결과 확인

모델은 어떤 근거를 바탕으로 해당 영역을 판단한 걸까?

Semantic Segmentation 모델은 결과(Prediction)는 보여주지만, 왜 그러한 예측을 수행했는지는 알려주지 않기에
이번에는 모델의 판단 근거를 시각적으로 확인하기 위해 Grad-CAM(Gradient-weighted Class Activation Mapping)을 적용해봄.

![](https://velog.velcdn.com/images/keepbini366/post/58386a79-8a4c-44ab-8160-2c483d969f43/image.png)

다만, 기존에 기대했던 도로를 보고 있다는 결과는 나오지 않았음. 
도로 전체를 중심으로 heatmap이 붉게 나오기를 기대 했으나, 실제로는 차량/버스/차선 일부, 프레임 하단 부분에 heatmap 집중한 모습을 볼 수 있었음. 

---

### ii) Epoch 50 based Training
학습 조건은 epoch 5일 때와 동일했고, epoch만 50으로 변경했음. 

> ### 로그
```
Epoch [45/50] Train Loss: 0.2729 | Val Loss: 0.3672
Best Model Updated!
Epoch [46/50] Train Loss: 0.2696 | Val Loss: 0.4167
Epoch [47/50] Train Loss: 0.2682 | Val Loss: 0.3963
Epoch [48/50] Train Loss: 0.2661 | Val Loss: 0.3590
Best Model Updated!
Epoch [49/50] Train Loss: 0.2511 | Val Loss: 0.3738
Epoch [50/50] Train Loss: 0.2734 | Val Loss: 0.3833
```


![](https://velog.velcdn.com/images/keepbini366/post/f87e5c6e-640d-4816-9b83-864863e959f6/image.png)

epoch 5 대비 확실히 Loss가 많이 감소한 것을 확인할 수 있었음. 

---

### Prediction 결과 확인

![](https://velog.velcdn.com/images/keepbini366/post/2964f2e3-af05-4f19-b0de-08261c2d3782/image.png)

Epoch 5와 비교했을 때 Prediction 결과가 전반적으로 개선된 것을 확인할 수 있었다.

- 비교적 잘 맞춘 부분
	- ``Car``: 앞 차량, 오른쪽 차량,  버스 잘 분리(epoch 5와 확연한 변화)(뭉개짐 완화)
	- ``Road/Sidewalk``: 경계 상대적으로 자연스러워짐
	- ``Sky``: 하늘과 건물 경계 잘 분리
	- ``Building``: 왼쪽/오른쪽 건물 모두 Ground Truth와 유사
	- ``Small Object``: Traffic Light, Signal Pole 분리하기 시작함
다만, ``Pedestrian``, ``Pole`` 부분이 아직 아쉽긴 함
(CamVid 자체가 작은 객체 비율이 낮기도 하지만...)
---

### Difference Map 결과 확인

![](https://velog.velcdn.com/images/keepbini366/post/e46e3cd5-a93d-45f6-a057-bf9b0b682d83/image.png)

Difference Map을 통해 모델가 어느 위치에서 오분류가 발생했는지는 확인할 수 있었지만, 모델은 해당 픽셀을 어떤 클래스로 잘못 예측한 것인지도 중요함. 
따라서, 하단 오분류 사례를 크게 세 가지 가지고 왔음. 

Building Class, Bus Class를 어떻게 인식한 것일까.

![](https://velog.velcdn.com/images/keepbini366/post/8d33b032-d232-47a6-8c4a-211cba155576/image.png)

결과를 확인해 보니,

**1. Building > Tree**
```
Ground Truth : Building
Prediction   : Tree
```
건물과 나무가 인접한 경계 영역에서 발생한 오분류임. 
해당 위치는 건물 외벽과 나무가 맞닿아 있고, 
어두운 조명과 그림자의 영향으로 두 객체의 Texture가 유사하게 나타난 것으로 판단됨. 

**2. Building > Wall**
```
Ground Truth : Building
Prediction   : Wall
```
Building과 Wall은 모두 구조물(Structure)에 속하는 객체임. 

따라서, 모델은 구조물이라는 큰 개념은 올바르게 이해했지만,
세부 클래스를 정확하게 구분하지는 못한 것으로 볼 수 있음. 

**3. Truck_Bus > SUVPickupTruck**
```
Ground Truth : Truck_Bus
Prediction   : SUVPickupTruck
```

상위 개념은 올바르게 인식했으나, 세부 차량 종류를 구분하는 데에는 어려움을 보였음.

즉, 모델이 차량이라는 의미(Semantic)는 충분히 학습했지만,
세부적인 Vehicle Type을 구분하는 데에는 한계가 있음을 확인할 수 있었음.

> 상단 세 사례를 통해, 
모두 **의미적으로는 유사한 클래스(Semantically Similar Classes) 사이에서 오분류가 발생**했음을 확인할 수 있었음. 

>
모델이 객체의 상위 의미(Semantic Category)는 비교적 잘 이해하고 있지만,
세부 클래스를 구분하는 능력은 아직 부족함을 보여줌. 

> 결과적으로 완전히 맞추진 못했지만, 유의미한 오답을 출력했다는 것. 

---

### Grad-CAM 결과 확인
![](https://velog.velcdn.com/images/keepbini366/post/4ea53e6a-d475-4458-8400-e4e024320622/image.png)


epoch 50일 때, 
전체적으로 Road 자체보다는, Road임을 예측하는 데 도움이 되는 주변 특징들을 많이 보고 있음.

- **왼쪽 차선 Heatmap 비교적 강하게 활성화**
-> 학습: 차선이 있는 곳 = Road일 가능성 높겠군
- **버스  뒷 부분 Heatmap 강한 활성화**
-> Road 예측 시, 차량도 Road의 Context가 됨
- 전체적으로 대개 Vehicle 부분의 Heatmap이 강하게 활성화 > **주요 Context**


다만 왜 좌측 Building을 강하게 보았을지에 대한 의문과 한계가 존재함. 

아무래도 가장 큰 이유는, **Classification GradCAM의 한계**라고 생각함. 

> #### GradCAM과 U-Net의 정합성 문제
- ``Grad-CAM``: 원래 이미지 전체를 하나의 클래스로 classification하는데 특화된 기법
- ``U-Net``: 이미지 한 장 전체에서 32개 클래스로 나눠서 Segmentation하는데 특화된 모델
>
원래 이미지 분류(Classification)를 위해 제안된 Grad-CAM을 
그대로 Semantic Segmentation 모델에 적용한 것임. 
>
다만 Semantic Segmentation은
이미지 전체를 하나의 클래스로 분류하는 문제가 아니라,
모든 픽셀을 각각 분류하는 문제.
>
따라서 Grad-CAM은 Road 전체를 설명하기보다는 출력에 큰 영향을 준 일부 Feature만 강조하게 되는 것임. 
>
결과적으로 실제로는 Road를 예측하기 위한 주변 정보(Context)가 함께 강조되거나, Building과 같은 영역에도 Heatmap이 나타날 수 있게 됨. 

> #### 이미지 전체 기준으로 Grad-CAM 적용
Semantic Segmentation용 Grad-CAM에서는 
일반적으로 특정 픽셀, 특정 객체 영역, 특정 클래스 마스크 등을 선택한 뒤 
해당 영역에 대해서만 Backpropagation을 수행함. 
> 
다만, 이번 실습에서는 전체 영역으로 진행했기에,
Road 뿐만 아니라, Building, 차선, 차량 등 Context Information도 함께 강조되고 있음. 

> #### Image Resize
이번 실습에서는 학습 효율과 GPU 메모리 사용량을 고려해 ``720 X 960`` input 이미지를 ``352 X 480``으로 Resize 했음. 
그렇기에, Encoder에서 MaxPooling을 진행하면서,Feature Map 크기가 ``22 X 30``까지 줄어들게 된 것임. 
>
그렇게 Grad-CAM은 Encor 마지막 Feature Map(22 × 30)을 기반으로 Heatmap을 생성하게 된 것임. 
>
따라서 Heatmap이 객체의 경계를 세밀하게 표현하기보다는 거칠고 넓은 영역을 강조하는 형태로 나타나게 된 것임. 

---
### Epoch 5 vs Epoch 50
> ### Prediction
![](https://velog.velcdn.com/images/keepbini366/post/0dadf3a6-d4b8-4253-a89b-5928fabfa893/image.png)

> ### Difference Map
![](https://velog.velcdn.com/images/keepbini366/post/41961db1-aa30-46d7-b51c-02be83ad1243/image.png)

> ### GradCAM 
![](https://velog.velcdn.com/images/keepbini366/post/648783ed-189b-4581-8501-62c4633d6c7d/image.png)

---

### Conclusion
이번 실습에서는 U-Net 기반 Semantic Segmentation 모델을 직접 구현하고, Prediction부터 Difference Map, Grad-CAM까지 적용하며 모델을 학습하고 해석하는 전 과정을 경험했음. 

특히 오분류 원인과 모델의 판단 근거를 분석해보며 XAI(Explainable AI)의 필요성을 확인할 수 있었음. 
앞으로는 Semantic Segmentation에 적합한 다양한 XAI 기법을 적용하여 모델의 설명 가능성을 더욱 향상시켜 볼 예정임. 

