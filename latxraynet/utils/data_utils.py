import os
import re
import pandas as pd
from cdifflib import CSequenceMatcher


#-------------------------------------------------------------------------
def folder_is_empty(path):
    return (os.path.isdir(path) and not os.listdir(path)) or (not os.path.isdir(path))

#-------------------------------------------------------------------------
def extract_id(filename):
    filename_no_ext = os.path.splitext(filename)[0]    
    match = re.search(r'(\d{8})', filename_no_ext)
    return match.group(1) if match else None

#-------------------------------------------------------------------------
def get_labels(file_path):
    suffix = file_path.split(".")[-1]

    if suffix == "xlsx":
        df = pd.read_excel(file_path)
        df = df.dropna()
        df["NIP"] = df["NIP"].astype(int).astype(str).str.zfill(8)
        return dict(zip(df["NIP"], df["Value"]))    
    elif suffix == "csv":
        df = pd.read_csv(file_path)
        return dict(zip(df["id"], df["label"])) 
    else:
        raise ValueError

#-------------------------------------------------------------------------
def check_numbers(samples):
    count_normal = 0
    count_anormal = 0
    for s in samples:
        if s[-1] == 0:
            count_normal += 1
        else:
            count_anormal += 1
    print("Total samples:", len(samples))
    print(f"Nb_Normal: {count_normal} | Nb_Anormal: {count_anormal}")

#-------------------------------------------------------------------------
def normalize_filename(name):
    """
    Normalize filename by removing extension, hashes, and unifying separators
    """
    name = os.path.splitext(name)[0]          # remove extension
    name = name.lower()

    # remove rf hash part (e.g. rf.47fff4c3f9f93b994530152e56ab433a)
    name = re.sub(r'rf\.[a-f0-9]+', '', name)

    # replace separators with space
    name = re.sub(r'[-_.]', ' ', name)

    # remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()

    return name

#-------------------------------------------------------------------------
def similarity(a, b):
    return CSequenceMatcher(None, a, b).ratio()

#-------------------------------------------------------------------------
def is_two_xrays_per_img(img_name, data_dir):
    img_norm = normalize_filename(img_name)

    search_folder = os.path.join(data_dir, "2 images")
    for file in os.listdir(search_folder):
        file_norm = normalize_filename(file)
        if similarity(img_norm, file_norm) > 0.75:
            return True
    
    search_folder = os.path.join(data_dir, "1 image")
    for file in os.listdir(search_folder):
        file_norm = normalize_filename(file)
        if similarity(img_norm, file_norm) > 0.75:
            return False

    return None

#-------------------------------------------------------------------------
if __name__ == "__main__":
    path = "data/2026/Masked data/2026.v1i.sam2/train"
    jpg_files = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]

    c = []
    for i, img_name in enumerate(jpg_files):
        print(i)
        flag = is_two_xrays_per_img(img_name, "data/2026/")
        if flag is None:
            print(i, img_name)