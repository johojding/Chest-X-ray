import os
from pathlib import Path
from PIL import Image
import utils
import pandas as pd

pd.set_option("display.max_columns", None)
dataset_names = ["train_normal", "train_pneumonia", "test_normal", "test_pneumonia", "val_normal", "val_pneumonia"]
data_stat_columns = ["dataset", "nbr_images"]
general_data_stat_columns = ["tot_nbr_imgs", "nbr_train_imgs", "nbr_val_imgs", "nbr_test_imgs", "train/val/test_split",
                      "nbr_normal_imgs", "nbr_pneumonia_imgs", "train_class_split", "val_class_split",
                      "test_class_split", "img_size", "color_channel", "aspect_ratio"]

data_path = "/Users/josephinehojding/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
root = Path(data_path)
paths = [
    root / "train" / "NORMAL",
    root / "train" / "PNEUMONIA",
    root / "test" / "NORMAL",
    root / "test" / "PNEUMONIA",
    root / "val" / "NORMAL",
    root/ "val" / "PNEUMONIA",
    ]

image_datasets = {name:utils.image_list(image_path) for name,image_path in zip(dataset_names, paths)}

# -------------- Dataset statistics ------------------- #
# how many images?
# train/val/test split?
# class distribution
# size, color channel, aspect ratio
# missing files, corrupt files,

image_statistics = pd.DataFrame(columns=data_stat_columns)
image_statistics["dataset"] = dataset_names
nbr_images = {name:len(image_datasets[name]) for name in image_datasets}
image_statistics["nbr_images"] = image_statistics["dataset"].map(nbr_images)
print(image_statistics)

general_img_stats = pd.DataFrame(columns=general_data_stat_columns)
general_img_stats["tot_nbr_imgs"] = utils.sum_list(image_statistics["nbr_images"])
general_img_stats["nbr_train_imgs"] = utils.sum_list(image_statistics["nbr_images"][:2])
general_img_stats["nbr_test_imgs"] = utils.sum_list(image_statistics["nbr_images"][2:4])
general_img_stats["nbr_val_imgs"] = utils.sum_list(image_statistics["nbr_images"][4:])
