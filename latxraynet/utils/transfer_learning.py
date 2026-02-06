

#-------------------------------------------------------------------------
def freeze_backbone(model):
    """
    Freeze all backbone parameters.
    Head remains trainable.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

#-------------------------------------------------------------------------
def unfreeze_last_backbone_layer(model):
    """
    Unfreeze only the last block/layer of the backbone.
    Works across timm models (EfficientNet, ResNet, DenseNet).
    """
    # First, freeze everything
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze last child module
    last_module = list(model.backbone.children())[-3]
    
    # # Get last BasicBlock
    # last_block = last_module[-1]

    # # Unfreeze conv2 and bn2
    # for layer_name in ["conv2", "bn2"]:
    #     layer = getattr(last_block, layer_name)
    #     for p in layer.parameters():
    #         p.requires_grad = True

    for param in last_module[-1].parameters():
        param.requires_grad = True

#-------------------------------------------------------------------------
def unfreeze_last_n_backbone_layers(model, n=3):
    """
    Unfreeze last N backbone layers/blocks.
    """
    # First, freeze everything
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    children = list(model.backbone.children())

    for layer in children[-n:]:
        for param in layer.parameters():
            param.requires_grad = True

#-------------------------------------------------------------------------
def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")

#-------------------------------------------------------------------------
def print_nb_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}/{total:,}")