
import timm
import torch.nn as nn


class TransformerBasedModel(nn.Module):
    """
    Transformer-based model for X-ray image classification.
    Suitable for small datasets (600-1000 images) with frozen backbone and trainable classification head.
    
    Args:
        backbone_name (str): Name of the backbone to use. Options: 'vit', 'deit'.
        num_classes (int): Number of output classes. Default is 1 (binary classification).
        pretrained (bool): Whether to use ImageNet pretrained weights. Default is True.
    """
    def __init__(self, backbone_name='vit', num_classes=1, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        # Select backbone
        if self.backbone_name == 'vit':
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif self.backbone_name == 'deit':
            self.backbone = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
            backbone_out_features = self.backbone.head.in_features
            self.backbone.head_dist = nn.Identity()
            self.backbone.head = nn.Identity()
        elif self.backbone_name == 'resnet50_transformer':
            # CNN backbone
            resnet = timm.create_model('resnet50', pretrained=pretrained, features_only=True)
            self.backbone = resnet
            cnn_out_features = resnet.feature_info[-1]['num_chs']
            
            # Small transformer encoder for global context
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cnn_out_features, nhead=4, dim_feedforward=512, batch_first=True
            )
            self.backbone_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            backbone_out_features = cnn_out_features
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported.")
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if self.backbone_name != 'resnet50_transformer':
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone_transformer.parameters():
                param.requires_grad = False

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if self.backbone_name == 'resnet50_transformer':
            features = self.backbone(x)[-1]  # Last feature map: (B, C, H, W)
            B, C, H, W = features.shape
            features = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            features = self.backbone_transformer(features)  # Transformer encoding
            features = features.mean(dim=1)  # Global average pooling
        else:
            features = self.backbone(x)

        out = self.classifier(features)
        return out.squeeze()

    def unfreeze_backbone_transformer(self):
        print(" Unfreezing transformer backbone... ")
        for param in self.backbone_transformer.parameters():
            param.requires_grad = True