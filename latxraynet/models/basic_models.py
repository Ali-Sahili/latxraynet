
import timm
import torch.nn as nn
import torchxrayvision as xrv

#-------------------------------------------------------------------------
class BasicModel(nn.Module):
  def __init__(
      self,
      model_name: str = "efficientnet_b4",
      dropout: float = 0.5,
  ):
      super().__init__()

      SUPPORTED_MODELS = {
          "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
          "efficientnet_b3", "efficientnet_b4",
          "resnet18", "resnet34", "resnet50",
          "densenet121", "densenet201"
      }

      if model_name not in SUPPORTED_MODELS:
          raise ValueError(f"Unsupported model: {model_name}")

      # Create backbone without classification head
      self.backbone = timm.create_model(
          model_name,
          pretrained=True,
          num_classes=0,       # removes classifier
          global_pool="avg",   # ensures consistent output
          in_chans=1           # grayscale
      )

      in_features = self.backbone.num_features

      # Classification head
      self.head = nn.Sequential(
          nn.Linear(in_features, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(inplace=True),
          nn.Dropout(dropout),
          nn.Linear(128, 1)
      )

  def freeze_backbone(self):
      for param in self.backbone.parameters():
          param.requires_grad = False

  def unfreeze_last_n_layers(self, n: int):
      """
      Unfreezes the last n layers (blocks) of the backbone.
      """
      children = list(self.backbone.children())

      for layer in children[-n:]:
          for param in layer.parameters():
              param.requires_grad = True

  def forward(self, x):
      x = self.backbone(x)
      x = self.head(x)
      return x.squeeze(1)

#-------------------------------------------------------------------------
class SpecializedBasicModel(nn.Module):
    def __init__(self, model_name="", state_dict = None, apply_freeze=False):
        super().__init__()

        # CheXpert pretrained DenseNet
        if model_name == "densenet-chex":
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-chex", 
                                                    in_channels=1)
            in_features = 18
        elif model_name == "densenet-mimic_nb":
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
            in_features = 18
        elif model_name == "densenet-mimic_ch":
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            in_features = 18
        elif model_name == "resnet18":
            self.backbone = timm.create_model('resnet18', num_classes=165, pretrained=False)
            
            if state_dict is not None:
                self.backbone.load_state_dict(state_dict, strict=False)
            
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = 512
        elif model_name == "resnet50":
            self.backbone = timm.create_model('resnet50', num_classes=165, pretrained=False)
            
            if state_dict is not None:
                self.backbone.load_state_dict(state_dict, strict=False)
            
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = 2048

        else:
            assert "MODEL NOT SUPPORTED!!"

        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        if apply_freeze:
            self.freeze_backbone()

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True


#-------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    B, C, H, W = 8, 1, 224, 224
    imgs = torch.randn(B, C, H, W)

    ## Test Basic Model
    # for model_name in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    #                     "efficientnet_b3", "efficientnet_b4",
    #                     "resnet18", "resnet34", "resnet50",
    #                     "densenet121", "densenet201"]:
    #     model = BasicModel(model_name=model_name)
    #     out = model(imgs)
    #     print(model_name, out.shape)

    ## Test SpecializedBasicModel
    for model_name in ["densenet-chex", "densenet-mimic_nb", 
                            "densenet-mimic_ch", "resnet18", "resnet50"]:
        if model_name == "resnet18":
            state_dict = None # get weights
        model = SpecializedBasicModel(model_name = model_name, state_dict=None)
        out = model(imgs)
        print(out.shape)