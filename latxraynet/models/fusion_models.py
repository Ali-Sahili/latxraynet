
import timm
import torch
import torch.nn as nn
from torchvision import models


#-------------------------------------------------------------------------
class LateFusionClassifier(nn.Module):
    def __init__(self, model_name = ""):
        super().__init__()

        assert model_name == "resnet18" or model_name == "resnet34"

        # ---------- GLOBAL BRANCH ----------
        if model_name == "resnet18":
            self.global_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.global_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.global_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.global_backbone.fc = nn.Identity()

        # ---------- ROI BRANCH ----------
        if model_name == "resnet18":
            self.roi_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet34":
            self.roi_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.roi_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.roi_backbone.fc = nn.Identity()

        # ---------- Freeze Backbones ----------
        for param in self.global_backbone.parameters():
            param.requires_grad = False

        for param in self.roi_backbone.parameters():
            param.requires_grad = False

        # ---------- FUSION HEAD ----------
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, full_img, roi_img, geom_feats):
        g_feat = self.global_backbone(full_img)
        r_feat = self.roi_backbone(roi_img)

        fused = torch.cat([g_feat, r_feat, geom_feats], dim=1)
        return self.classifier(fused).squeeze()

#-------------------------------------------------------------------------
class SpecializedLateFusionClassifier(nn.Module):
    def __init__(self, pretrained_state_dict=None):
        super().__init__()

        # -------- GLOBAL BRANCH --------
        self.global_backbone = timm.create_model('resnet18', num_classes=165, pretrained=False)
        if pretrained_state_dict is not None:
            self.global_backbone.load_state_dict(pretrained_state_dict, strict=False)

        self.global_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.global_backbone.fc = nn.Identity()

        # -------- ROI BRANCH --------
        self.roi_backbone = timm.create_model('resnet18', num_classes=165, pretrained=False)
        if pretrained_state_dict is not None:
            self.roi_backbone.load_state_dict(pretrained_state_dict, strict=False)

        self.roi_backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.roi_backbone.fc = nn.Identity()

        # -------- FUSION HEAD --------
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, full_img, roi_img, geom):
        g = self.global_backbone(full_img)
        r = self.roi_backbone(roi_img)
        x = torch.cat([g, r, geom], dim=1)
        return self.head(x).squeeze()
