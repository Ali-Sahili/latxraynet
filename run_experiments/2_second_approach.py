import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from latxraynet.dataset import get_datasets

from latxraynet.models.basic_models import SpecializedBasicModel
from latxraynet.training.train_eval import train_model
from latxraynet.utils.get_weights import get_radimagenet_weights

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "data/2026/"
LABELS_PATH = "data/2026/labels.csv"
IMG_SIZE = 224
N_EPOCHS = 50
SEED = 123

res_path = "results/specialized_basic_model"
os.makedirs(res_path, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

image_dir = os.path.join(ROOT_DIR, "xray_images")
mask_dir = os.path.join(ROOT_DIR, "xray_masks")

train_set, val_set, pos, neg = get_datasets(image_dir, mask_dir, LABELS_PATH, IMG_SIZE)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, drop_last=True)

pos_weight = torch.tensor([neg / pos]).to(device)

state_dict = None
apply_feeze = True
prefix = "specialized_basic_model_" if not apply_feeze else "specialized_basic_model_frozen_backbone_"
for model_name in ["densenet-chex", "densenet-mimic_nb", 
                    "densenet-mimic_ch", "resnet18", "resnet50"]:
    
    print(f" ----------------- {model_name} ----------------- ")
    
    if model_name == "resnet18":
        print("Initialize Radimagenet weights...")
        state_dict = get_radimagenet_weights("latxraynet/weights/radimagenet_resnet18.pth")
    elif model_name == "resnet50":
        print("Initialize Radimagenet weights...")
        state_dict = get_radimagenet_weights("latxraynet/weights/radimagenet_resnet50.pth")
    
    model = SpecializedBasicModel(model_name = model_name, 
                                    state_dict=state_dict,
                                        apply_freeze=apply_feeze)

    train_model(model, train_loader, val_loader, device, pos_weight=pos_weight,
            epochs=N_EPOCHS, 
            model_name=prefix + model_name, res_path=res_path)