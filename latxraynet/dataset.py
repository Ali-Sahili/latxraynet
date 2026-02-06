import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from .utils.data_utils import get_labels

def get_row(d, i):
    return {k: v[i] for k, v in d.items()}

#-------------------------------------------------------------------------
class AdenoidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, labels_path, img_size = 512, train=True, is_rgb=False):
        
        self.image_dir = image_dir
        self.masks_dir = mask_dir

        labels_pd = get_labels(labels_path)
        self.labels_pd = {
            "id": [int(k) for k in labels_pd.keys()],
            "label": list(labels_pd.values())
        }

        self.train = train
        self.is_rgb = is_rgb
        
        if train:
            self.transform = A.Compose([
                # Resize
                A.Resize(img_size, img_size),
                # Geometric Augmentation
                A.Rotate(limit=7, p=0.5),  # small rotation
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),  # small jitter
                # Photometric Augmentation
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),  # enhance soft tissue contrast
                # Elastic / Anatomical Variability
                A.ElasticTransform(alpha=1, sigma=20, alpha_affine=5, p=0.2),
                # Convert to Tensor
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.labels_pd["id"])

    def __getitem__(self, idx):
        id = self.labels_pd["id"][idx]

        img_path = os.path.join(self.image_dir, str(id).zfill(8) + ".png")
        mask_path = os.path.join(self.masks_dir, str(id).zfill(8) + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert image.shape == mask.shape

        if self.is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = (image / 255.0).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0)

        label = torch.tensor(self.labels_pd["label"][idx], dtype=torch.float32)
        
        return image, mask, label


#-------------------------------------------------------------------------
def get_datasets(image_dir, mask_dir, labels_path, img_size = 512, is_rgb=False):
    labels_df = pd.read_csv(labels_path)

    train_df, val_df = train_test_split(
        labels_df, test_size=0.15, stratify=labels_df["label"], random_state=42
    )
    print("Training vs Validation sets size: ", len(train_df), len(val_df))

    pos = sum(train_df['label'] == 1)
    neg = sum(train_df['label'] == 0)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

    train_set = AdenoidDataset(image_dir, mask_dir, "train.csv", train=True, img_size=img_size, is_rgb=is_rgb)
    val_set = AdenoidDataset(image_dir, mask_dir, "val.csv", train=False, img_size=img_size, is_rgb=is_rgb)

    return train_set, val_set, pos, neg

#-------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "data/2026/"
    LABELS_PATH = "data/2026/labels.csv"
    IMG_SIZE = 512

    image_dir = os.path.join(ROOT_DIR, "xray_images")
    mask_dir = os.path.join(ROOT_DIR, "xray_masks")
    dataset = AdenoidDataset(image_dir, mask_dir, LABELS_PATH, IMG_SIZE, train=True)
    print("Size of the dataset: ", len(dataset))
    
    train_set, val_set = get_datasets(image_dir, mask_dir, LABELS_PATH)
    print(f"Training Set Size: {len(train_set)} | Validation Set Size: {len(val_set)}")
