import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------
def remove_borders(img):
    if img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Remove black background
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    # Find largest component (excluding background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create mask of largest component
    body_mask = np.zeros_like(labels, dtype=np.uint8)
    body_mask[labels == largest_label] = 255

    # Apply mask to original image
    body_only = cv2.bitwise_and(img, img, mask=body_mask)

    # Crop tightly to body
    coords = cv2.findNonZero(body_mask)
    x, y, w, h = cv2.boundingRect(coords)
    body_only = body_only[y:y+h, x:x+w]

    return x, y, w, h # body_only

#-------------------------------------------------------------------------
def visualize_mask(img_path, mask, bbox=None, is_remove_borders=False):
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)

    if is_remove_borders:
        x, y, w, h = remove_borders(img)
        img = img[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]
    
    # Create colored overlay
    overlay = img.copy()
    overlay[mask] = [255, 0, 0]

    if bbox:
        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])), (0, 255, 0), 2)  # green bbox

    # Blend image + mask
    alpha = 0.5
    visual = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Show
    plt.imshow(visual)
    plt.title("Mask Overlay")
    plt.axis("off")
    plt.show()

#-------------------------------------------------------------------------
def save_mask(img_path, img_id, mask, root_dir, flag = True, no_borders = True):
    img = cv2.imread(img_path)
    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8) * 255

    if flag:
        # Split into left and right X-rays
        h, w, _ = img.shape
        left_xray = img[:, :w//2]
        right_xray = img[:, w//2:]
        
        # Find coordinates where mask is white (non-zero)
        ys, xs = np.where(mask > 0)

        # Bounding box indices
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        if y_min >= 0 and y_max <= h:
            if x_max <= w//2 and x_min >= 0:
                img = left_xray
                mask = mask[:, :w//2]
            elif x_max > w//2 and x_min <= w: 
                img = right_xray
                mask = mask[:, w//2:]
            else:
                raise ValueError
        else:
            raise ValueError

    if no_borders:
        x, y, w, h = remove_borders(img)
        img = img[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    cv2.imwrite(os.path.join(root_dir, "xray_images", str(img_id).zfill(8) +".png"), img)
    cv2.imwrite(os.path.join(root_dir, "xray_masks", str(img_id).zfill(8) +".png"), mask)

#-------------------------------------------------------------------------
def visualize_img_with_mask(img_id, root_dir):
    img_path = os.path.join(root_dir, "xray_images", str(img_id).zfill(8) +".png")
    mask_path = os.path.join(root_dir, "xray_masks", str(img_id).zfill(8) +".png")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, 0) / 255.
    mask = mask.astype(bool)

        # Create colored overlay
    overlay = img.copy()
    overlay[mask] = [255, 0, 0]

    # Blend image + mask
    alpha = 0.5
    visual = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Show
    plt.imshow(visual)
    plt.title("Mask Overlay")
    plt.axis("off")
    plt.show()