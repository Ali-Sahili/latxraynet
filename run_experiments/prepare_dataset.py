import os
 
 
from latxraynet.prepare_data import build_data, save_labels
from latxraynet.utils.img_utils import visualize_img_with_mask
from latxraynet.utils.data_utils import get_labels, folder_is_empty

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
        print(len(labels.keys()))

        if DEBUG:
            for filename in png_files:
                visualize_img_with_mask(img_id=filename.split(".")[0], root_dir=ROOT_DIR)