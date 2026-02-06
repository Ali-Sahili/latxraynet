import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pycocotools import mask as maskUtils

from .utils.img_utils import save_mask, visualize_img_with_mask, visualize_mask
from .utils.data_utils import (is_two_xrays_per_img, extract_id, get_labels, 
                            folder_is_empty, check_numbers)

#-------------------------------------------------------------------------
def build_data(data_dir, labels_path, visualize=False):
    
    os.makedirs(os.path.join(data_dir, "xray_images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "xray_masks"), exist_ok=True)
    
    masks_dir = os.path.join(data_dir, "Masked data/2026.v1i.sam2/train")
    folder_path = Path(masks_dir)
    json_filenames = [file.name for file in folder_path.glob("*.json")]

    labels_pd = get_labels(labels_path)
    samples = []
    not_found_labels = 0
    for json_filename in json_filenames:
        json_path = os.path.join(masks_dir, json_filename)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data["annotations"]) == 1:
                bbox = data["annotations"][0]["bbox"]
                mask = maskUtils.decode(data["annotations"][0]["segmentation"])
                file_name = data["image"]["file_name"]
                img_id = extract_id(file_name)
                img_label = labels_pd.get(img_id)
                flag = is_two_xrays_per_img(file_name, data_dir)

                if img_label is None or flag is None:
                    not_found_labels += 1
                    continue

                img_path = os.path.join(masks_dir, file_name)
                save_mask(img_path, img_id, mask, data_dir, flag, no_borders = True)
                samples.append((img_path, bbox, mask, flag, img_label))
                if visualize: visualize_mask(img_path, mask, bbox)
    
    check_numbers(samples)
    print(f"Number of non-found-labels: {not_found_labels}")

#-------------------------------------------------------------------------
def save_labels(data_dir, labels_path):
    img_paths = os.path.join(data_dir, "xray_images")
    png_files = [f for f in os.listdir(img_paths) if f.lower().endswith(".png")]
    
    labels_pd = get_labels(labels_path)

    ids, labels = [], []
    for filename in png_files:
        id  = str(filename.split(".")[0]).zfill(8)
        
        ids.append(id)
        labels.append(labels_pd.get(id))
    
    # Create DataFrame
    df = pd.DataFrame({
        "id": ids,
        "label": labels
    })

    # Save to CSV
    df.to_csv(os.path.join(data_dir, "labels.csv"), index=False)



#-------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "data/2026/"
    LABELS_PATH = "data/2026/ORL-2026.xlsx"

    DEBUG = False

    if folder_is_empty(os.path.join(ROOT_DIR, "xray_images")):
        build_data(ROOT_DIR, LABELS_PATH)
        save_labels(ROOT_DIR, LABELS_PATH)
    else:
        png_files = [f for f in os.listdir(os.path.join(ROOT_DIR, "xray_images")) if f.lower().endswith(".png")]
        print("Number of xray images: ", len(png_files))

        labels = get_labels(os.path.join(ROOT_DIR, "labels.csv"))
        print(labels.shape)

        if DEBUG:
            for filename in png_files:
                visualize_img_with_mask(img_id=filename.split(".")[0], root_dir=ROOT_DIR)