import glob
import os.path

import cv2
from tqdm import tqdm

file_path_list = glob.glob("/mnt/c/dataset_tmp/gt/*.tif")
dst_dir = r"/mnt/c/dataset_tmp/fix_gt"
os.makedirs(dst_dir, exist_ok=True)
results_list = []
for file_path in tqdm(file_path_list):
    img = cv2.imread(file_path, -1)
    if img.min() != 0:
        img[img == 200] = 0
        if os.path.basename(file_path) == "2929.tif":
            img[img == 150] = 0
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(file_path)), img)
    else:
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(file_path)), img)
