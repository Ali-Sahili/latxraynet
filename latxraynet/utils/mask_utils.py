
import cv2
import torch
import numpy as np

#-------------------------------------------------------------------------
def crop_to_mask(img, mask, out_size=224, margin=0.25):
    """
    img: (1, H, W)
    mask: (1, H, W)
    """
    img = img.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # fallback: center crop
        print("Center crop...")
        h, w = img.shape
        crop = img[h//4:3*h//4, w//4:3*w//4]
    else:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        h, w = img.shape
        dy = int((y2 - y1) * margin)
        dx = int((x2 - x1) * margin)

        y1 = max(0, y1 - dy)
        y2 = min(h, y2 + dy)
        x1 = max(0, x1 - dx)
        x2 = min(w, x2 + dx)

        crop = img[y1:y2, x1:x2]

    if crop.shape[1] < 5 or crop.shape[0] < 5:
        print("Center crop...")
        crop = img[h//4:3*h//4, w//4:3*w//4]

    crop = cv2.resize(crop, (out_size, out_size))
    crop = torch.tensor(crop).unsqueeze(0)

    return crop