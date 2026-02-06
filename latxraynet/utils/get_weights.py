

import torch
from collections import OrderedDict



#-------------------------------------------------------------------------
def get_radimagenet_weights(cpkt_path="weights/radimagenet_resnet18.pth"):
    radimagenet_weights = torch.load(cpkt_path, 
                                      map_location=torch.device("cuda"),
                                        weights_only=False)['model']
    state_dict = OrderedDict()
    for k, v in radimagenet_weights.items():
        state_dict[k[10:]] = v
    return state_dict