import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from latxraynet.dataset import get_datasets
from latxraynet.models.transformer_based import TransformerBasedModel
from latxraynet.training.train_eval import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "data/2026/"
LABELS_PATH = "data/2026/labels.csv"
IMG_SIZE = 224
N_EPOCHS = 50
SEED = 123

res_path = "results/transformer_model"
os.makedirs(res_path, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

image_dir = os.path.join(ROOT_DIR, "xray_images")
mask_dir = os.path.join(ROOT_DIR, "xray_masks")

train_set, val_set, pos, neg = get_datasets(image_dir, mask_dir, LABELS_PATH, IMG_SIZE, is_rgb=True)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True)

pos_weight = torch.tensor([neg / pos]).to(device)

apply_freeze = False
for model_name in [
                    # "vit", 
                    # "deit",
                    "resnet50_transformer"
                    ]:
    print(f" ----------------- {model_name} ----------------- ")
    
    if model_name == "resnet50_transformer":
        apply_freeze = True
    
    model = TransformerBasedModel(backbone_name=model_name)

    train_model(model, train_loader, val_loader, device, pos_weight=pos_weight,
                    epochs=N_EPOCHS, model_name="transformer-based_model_"+model_name,
                    res_path=res_path, apply_freeze=apply_freeze)