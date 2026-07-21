import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_csv, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

        class_df = pd.read_csv(class_csv)
        self.class_rgb = class_df[["r", "g", "b"]].values
        self.class_names = class_df["name"].tolist()

    def __len__(self):
        return len(self.image_names)

    def rgb_to_class(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for class_idx, rgb in enumerate(self.class_rgb):
            match = np.all(mask_rgb == rgb, axis=-1)
            class_mask[match] = class_idx

        return class_mask

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        mask_name = self.mask_names[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = self.rgb_to_class(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return image, mask