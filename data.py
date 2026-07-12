import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)

# ------------- PATHS ---------------- #
DATA_PATH = (
    "/Users/josephinehojding/.cache/kagglehub/datasets/"
    "paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
)

root = Path(DATA_PATH)
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from collections import Counter

pd.set_option("display.max_columns", None)

# ------------- PATHS ---------------- #
DATA_PATH = (
    "/Users/josephinehojding/.cache/kagglehub/datasets/"
    "paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
)

root = Path(DATA_PATH)

dataset_paths = {
    "train_normal": root / "train" / "NORMAL",
    "train_pneumonia": root / "train" / "PNEUMONIA",
    "test_normal": root / "test" / "NORMAL",
    "test_pneumonia": root / "test" / "PNEUMONIA",
    "val_normal": root / "val" / "NORMAL",
    "val_pneumonia": root / "val" / "PNEUMONIA",
}

# ------------------ HELPER FUNCTIONS ---------------- #

def get_image_paths(path: Path):
    """Return all jpeg image paths."""
    return list(path.rglob("*.jpeg"))


def percent(part, total):
    return round(part / total * 100, 2)

# ------------- LOAD IMAGE PATHS -------------- #
image_datasets = {
    name: get_image_paths(path)
    for name, path in dataset_paths.items()
}

# --------------- DATASET LEVEL STATS -------------- #
image_statistics = pd.DataFrame({
    "dataset": image_datasets.keys(),
    "nbr_images": [len(files) for files in image_datasets.values()]
})

print("\nDATASET STATISTICS")
print(image_statistics)

# ----------------- GLOBAL COUNTS ----------------- #
total_imgs = image_statistics["nbr_images"].sum()

train_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("train"),
    "nbr_images"
].sum()

test_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("test"),
    "nbr_images"
].sum()

val_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("val"),
    "nbr_images"
].sum()

normal_imgs = image_statistics.loc[
    image_statistics.dataset.str.contains("normal"),
    "nbr_images"
].sum()

pneumonia_imgs = image_statistics.loc[
    image_statistics.dataset.str.contains("pneumonia"),
    "nbr_images"
].sum()

# -----------------CLASS SPLITS -------------- #
train_normal = image_statistics.loc[
    image_statistics.dataset == "train_normal",
    "nbr_images"
].iloc[0]

train_pneumonia = image_statistics.loc[
    image_statistics.dataset == "train_pneumonia",
    "nbr_images"
].iloc[0]

test_normal = image_statistics.loc[
    image_statistics.dataset == "test_normal",
    "nbr_images"
].iloc[0]

test_pneumonia = image_statistics.loc[
    image_statistics.dataset == "test_pneumonia",
    "nbr_images"
].iloc[0]

val_normal = image_statistics.loc[
    image_statistics.dataset == "val_normal",
    "nbr_images"
].iloc[0]

val_pneumonia = image_statistics.loc[
    image_statistics.dataset == "val_pneumonia",
    "nbr_images"
].iloc[0]

train_class_split = (
    f"{percent(train_normal, train_imgs)}/"
    f"{percent(train_pneumonia, train_imgs)}"
)

test_class_split = (
    f"{percent(test_normal, test_imgs)}/"
    f"{percent(test_pneumonia, test_imgs)}"
)

val_class_split = (
    f"{percent(val_normal, val_imgs)}/"
    f"{percent(val_pneumonia, val_imgs)}"
)

train_val_test_split = (
    f"{percent(train_imgs, total_imgs)}/"
    f"{percent(val_imgs, total_imgs)}/"
    f"{percent(test_imgs, total_imgs)}"
)

# ------------------ IMAGE LEVEL STATS --------------- #
size_counter = Counter()
channel_counter = Counter()
aspect_ratios = []

corrupt_images = []

image_metadata = []

print("\nScanning images...")

for dataset_name, image_paths in image_datasets.items():

    split, label = dataset_name.split("_")

    for image_path in image_paths:

        try:
            with Image.open(image_path) as img:

                width, height = img.size
                mode = img.mode
                aspect_ratio = width / height

                # Summary statistics
                size_counter[(width, height)] += 1
                channel_counter[mode] += 1
                aspect_ratios.append(aspect_ratio)

                # Per-image statistics
                image_metadata.append({
                    "dataset": dataset_name,
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 3),
                    "pixels": width * height,
                    "mode": mode,
                    "size_bytes": image_path.stat().st_size,
                    "filepath": str(image_path),
                    "filename": image_path.name,
                })

        except Exception:

            corrupt_images.append(str(image_path))

            image_metadata.append({
                "dataset": dataset_name,
                "split": split,
                "label": label,
                "width": np.nan,
                "height": np.nan,
                "aspect_ratio": np.nan,
                "pixels": np.nan,
                "mode": None,
                "size_bytes": image_path.stat().st_size,
                "corrupt": True,
                "filepath": str(image_path),
                "filename": image_path.name,
            })
image_df = pd.DataFrame(image_metadata)

image_df["corrupt"] = image_df.get("corrupt", False)

# -------------------- SUMMARIES ------------------ #
most_common_sizes = size_counter.most_common(5)

most_common_channels = dict(channel_counter)

aspect_ratio_summary = (
    f"mean={np.mean(aspect_ratios):.3f}, "
    f"std={np.std(aspect_ratios):.3f}, "
    f"min={np.min(aspect_ratios):.3f}, "
    f"max={np.max(aspect_ratios):.3f}"
)

# --------------- GENERAL STATISTICS ----------------- #

general_img_stats = pd.DataFrame([{
    "tot_nbr_imgs": total_imgs,
    "nbr_train_imgs": train_imgs,
    "nbr_val_imgs": val_imgs,
    "nbr_test_imgs": test_imgs,
    "train/val/test_split": train_val_test_split,

    "nbr_normal_imgs": normal_imgs,
    "nbr_pneumonia_imgs": pneumonia_imgs,

    "train_class_split": train_class_split,
    "val_class_split": val_class_split,
    "test_class_split": test_class_split,

    "img_size": str(most_common_sizes),
    "color_channel": str(most_common_channels),
    "aspect_ratio": aspect_ratio_summary,

    "nbr_corrupt_imgs": len(corrupt_images)
}])

# --------------------- OUTPUT -------------------- #
print("\nGENERAL IMAGE STATISTICS")
print(general_img_stats)

print("\nPER-IMAGE DATA")
print(image_df.head())

image_df.to_csv("image_metadata.csv", index=False)
dataset_paths = {
    "train_normal": root / "train" / "NORMAL",
    "train_pneumonia": root / "train" / "PNEUMONIA",
    "test_normal": root / "test" / "NORMAL",
    "test_pneumonia": root / "test" / "PNEUMONIA",
    "val_normal": root / "val" / "NORMAL",
    "val_pneumonia": root / "val" / "PNEUMONIA",
}

# ------------------ HELPER FUNCTIONS ---------------- #

def get_image_paths(path: Path):
    """Return all jpeg image paths."""
    return list(path.rglob("*.jpeg"))


def percent(part, total):
    return round(part / total * 100, 2)

# ------------- LOAD IMAGE PATHS -------------- #
image_datasets = {
    name: get_image_paths(path)
    for name, path in dataset_paths.items()
}

# --------------- DATASET LEVEL STATS -------------- #
image_statistics = pd.DataFrame({
    "dataset": image_datasets.keys(),
    "nbr_images": [len(files) for files in image_datasets.values()]
})

print("\nDATASET STATISTICS")
print(image_statistics)

# ----------------- GLOBAL COUNTS ----------------- #
total_imgs = image_statistics["nbr_images"].sum()

train_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("train"),
    "nbr_images"
].sum()

test_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("test"),
    "nbr_images"
].sum()

val_imgs = image_statistics.loc[
    image_statistics.dataset.str.startswith("val"),
    "nbr_images"
].sum()

normal_imgs = image_statistics.loc[
    image_statistics.dataset.str.contains("normal"),
    "nbr_images"
].sum()

pneumonia_imgs = image_statistics.loc[
    image_statistics.dataset.str.contains("pneumonia"),
    "nbr_images"
].sum()

# -----------------CLASS SPLITS -------------- #
train_normal = image_statistics.loc[
    image_statistics.dataset == "train_normal",
    "nbr_images"
].iloc[0]

train_pneumonia = image_statistics.loc[
    image_statistics.dataset == "train_pneumonia",
    "nbr_images"
].iloc[0]

test_normal = image_statistics.loc[
    image_statistics.dataset == "test_normal",
    "nbr_images"
].iloc[0]

test_pneumonia = image_statistics.loc[
    image_statistics.dataset == "test_pneumonia",
    "nbr_images"
].iloc[0]

val_normal = image_statistics.loc[
    image_statistics.dataset == "val_normal",
    "nbr_images"
].iloc[0]

val_pneumonia = image_statistics.loc[
    image_statistics.dataset == "val_pneumonia",
    "nbr_images"
].iloc[0]

train_class_split = (
    f"{percent(train_normal, train_imgs)}/"
    f"{percent(train_pneumonia, train_imgs)}"
)

test_class_split = (
    f"{percent(test_normal, test_imgs)}/"
    f"{percent(test_pneumonia, test_imgs)}"
)

val_class_split = (
    f"{percent(val_normal, val_imgs)}/"
    f"{percent(val_pneumonia, val_imgs)}"
)

train_val_test_split = (
    f"{percent(train_imgs, total_imgs)}/"
    f"{percent(val_imgs, total_imgs)}/"
    f"{percent(test_imgs, total_imgs)}"
)

# ------------------ IMAGE LEVEL STATS --------------- #
size_counter = Counter()
channel_counter = Counter()
aspect_ratios = []

corrupt_images = []

image_metadata = []

print("\nScanning images...")

for dataset_name, image_paths in image_datasets.items():

    split, label = dataset_name.split("_")

    for image_path in image_paths:

        try:
            with Image.open(image_path) as img:

                width, height = img.size
                mode = img.mode
                aspect_ratio = width / height

                # Summary statistics
                size_counter[(width, height)] += 1
                channel_counter[mode] += 1
                aspect_ratios.append(aspect_ratio)

                # Per-image statistics
                image_metadata.append({
                    "dataset": dataset_name,
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 3),
                    "pixels": width * height,
                    "mode": mode,
                    "size_bytes": image_path.stat().st_size,
                    "filepath": str(image_path),
                    "filename": image_path.name,
                })

        except Exception:

            corrupt_images.append(str(image_path))

            image_metadata.append({
                "dataset": dataset_name,
                "split": split,
                "label": label,
                "width": np.nan,
                "height": np.nan,
                "aspect_ratio": np.nan,
                "pixels": np.nan,
                "mode": None,
                "size_bytes": image_path.stat().st_size,
                "corrupt": True,
                "filepath": str(image_path),
                "filename": image_path.name,
            })
image_df = pd.DataFrame(image_metadata)

image_df["corrupt"] = image_df.get("corrupt", False)

# -------------------- SUMMARIES ------------------ #
most_common_sizes = size_counter.most_common(5)

most_common_channels = dict(channel_counter)

aspect_ratio_summary = (
    f"mean={np.mean(aspect_ratios):.3f}, "
    f"std={np.std(aspect_ratios):.3f}, "
    f"min={np.min(aspect_ratios):.3f}, "
    f"max={np.max(aspect_ratios):.3f}"
)

# --------------- GENERAL STATISTICS ----------------- #

general_img_stats = pd.DataFrame([{
    "tot_nbr_imgs": total_imgs,
    "nbr_train_imgs": train_imgs,
    "nbr_val_imgs": val_imgs,
    "nbr_test_imgs": test_imgs,
    "train/val/test_split": train_val_test_split,

    "nbr_normal_imgs": normal_imgs,
    "nbr_pneumonia_imgs": pneumonia_imgs,

    "train_class_split": train_class_split,
    "val_class_split": val_class_split,
    "test_class_split": test_class_split,

    "img_size": str(most_common_sizes),
    "color_channel": str(most_common_channels),
    "aspect_ratio": aspect_ratio_summary,

    "nbr_corrupt_imgs": len(corrupt_images)
}])

# --------------------- OUTPUT -------------------- #
print("\nGENERAL IMAGE STATISTICS")
print(general_img_stats)

print("\nPER-IMAGE DATA")
print(image_df.head())

image_df.to_csv("image_metadata.csv", index=False)
