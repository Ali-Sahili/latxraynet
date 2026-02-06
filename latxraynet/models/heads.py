
import timm
import torch
import torch.nn as nn
import torchxrayvision as xrv

#-------------------------------------------------------------------------
class BasicClassifier(nn.Module):
    def __init__(self, model_name = "resnet34"):
        super().__init__()

        if model_name == "":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            in_features = 32

        elif model_name == "resnet34":
            self.cnn = timm.create_model(
                "resnet34",
                pretrained=True,
                num_classes=0,       # removes classifier
                global_pool="avg",   # ensures consistent output
                in_chans=1           # grayscale
            )

            in_features = self.cnn.num_features

            for param in self.cnn.parameters():
                param.requires_grad = False

        # head
        self.fc = nn.Sequential(
            nn.Linear(in_features + 3, 64),  # CNN features + geometry
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, geom_features):
        features = self.cnn(x).view(x.size(0), -1)
        combined = torch.cat([features, geom_features], dim=1)
        return self.fc(combined)

#-------------------------------------------------------------------------
class SpecializedBasicClassifier(nn.Module):
    def __init__(self, model_name = "resnet50", state_dict=None):
        super().__init__()

        if model_name == "densenet-mimic_ch":
            self.cnn = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            self.cnn.fc = nn.Identity()
            in_features = 18

        elif model_name == "resnet50":
            self.cnn = timm.create_model('resnet50', num_classes=165, pretrained=False)

            if state_dict is not None:
                self.cnn.load_state_dict(state_dict, strict=False)

            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.fc = nn.Identity()
            in_features = 2048
            
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(in_features + 3, 64),  # CNN features + geometry
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, geom_features):
        features = self.cnn(x).view(x.size(0), -1)
        combined = torch.cat([features, geom_features], dim=1)
        return self.fc(combined)