import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from latxraynet.dataset import get_datasets
from latxraynet.models.segmentation import BasicUNet
from latxraynet.training.train_segmentation import train_seg
from latxraynet.utils.get_weights import get_radimagenet_weights
from latxraynet.training.train_roi_head import train_cls_with_roi
from latxraynet.models.heads import BasicClassifier, SpecializedBasicClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "data/2026/"
LABELS_PATH = "data/2026/labels.csv"
IMG_SIZE = 224
N_EPOCHS = 50
SEED = 123

RES_DIR = "results"
res_path = "results/specialized_basic_head_classifier"
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

for name in ["specialized"]:
    print(f" ------------- {name} ------------- ")
    if name == "basic": cls_model = BasicClassifier().to(device)
    elif name == "specialized": 
        # print("Initialize Radimagenet weights...")
        # state_dict = get_radimagenet_weights("latxraynet/weights/radimagenet_resnet50.pth")
        state_dict = None
        cls_model = SpecializedBasicClassifier(model_name="densenet-mimic_ch", state_dict=state_dict).to(device)
    else: assert "NOT SUPPORTED MODEL!!"

    train_cls_with_roi(cls_model, seg_model, train_loader, val_loader, pos_weight, 
            device, model_name = name + "_head_roi_classifier", 
            num_epochs=N_EPOCHS, res_path=res_path)
