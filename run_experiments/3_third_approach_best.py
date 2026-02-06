import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from latxraynet.dataset import get_datasets
from latxraynet.models.heads import BasicClassifier
from latxraynet.models.segmentation import BasicUNet
from latxraynet.training.train_geo_head import train_cls
from latxraynet.training.train_segmentation import train_seg


device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "data/2026/"
LABELS_PATH = "data/2026/labels.csv"
IMG_SIZE = 224
N_EPOCHS = 50
SEED = 123

RES_DIR = "results"
res_path = os.path.join(RES_DIR, "basic_head_classifier_best")
os.makedirs(res_path, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

image_dir = os.path.join(ROOT_DIR, "xray_images")
mask_dir = os.path.join(ROOT_DIR, "xray_masks")

train_set, val_set, pos, neg = get_datasets(image_dir, mask_dir, LABELS_PATH, IMG_SIZE)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True)

pos_weight = torch.tensor([neg / pos]).to(device)

if not os.path.exists(os.path.join(RES_DIR, "best_segmentation.pth")):
    print("Training Segmentation...")
    model = BasicUNet().to(device)
    train_seg(model, train_loader, val_loader, device, num_epochs=N_EPOCHS,
                    res_path=RES_DIR)

print("Training Classifier...")
seg_model = BasicUNet().to(device)
seg_model.load_state_dict(torch.load("best_segmentation.pth"))
seg_model.eval()

clf_model = BasicClassifier(model_name="resnet34").to(device)
train_cls(seg_model, clf_model, train_loader, val_loader, pos_weight, 
            device, model_name = "frozen_resnet34_head_classifier", 
                    num_epochs=N_EPOCHS, res_path=res_path)
