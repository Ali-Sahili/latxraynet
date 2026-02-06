
import torch.nn as nn
import segmentation_models_pytorch as smp


#-------------------------------------------------------------------------
class BasicUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, x):
        return self.model(x)