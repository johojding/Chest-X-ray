import pandas as pd
from PIL import Image
import numpy as np

def load_image(filepath):
    image = (Image.open(filepath))
    if image.mode == "L":
        image = Image.merge("RGB", (image, image, image))
    resized_im = image.resize((224,224))
    #image.show()
    im_arr = np.array(resized_im)
    im_arr = im_arr / 255
    return im_arr

def image_check(im):
    print(f"shape: {im.shape}")
    print(f"dtype: {im.dtype}")
    print(f"min: {im.min()}")
    print(f"max: {im.max()}")

image_df = pd.read_csv("image_metadata.csv")
clean_image_df = image_df[["split", "label", "filepath"]].copy()
clean_image_df["label_id"] = clean_image_df["label"].map({"normal": 0, "pneumonia": 1})

""" images_idxs = [0, 1000, 2000, 3000 ,4000, 5000]

for idx in images_idxs:
    row = clean_image_df.loc[idx]
    test_im = load_image(row["filepath"])
    image_check(test_im) """

train_images = []
train_labels = []

val_images = []
val_labels = []

test_images = []
test_labels = []

for i in range(clean_image_df.shape[0]):
    row = clean_image_df.loc[i]
    im = load_image(row["filepath"])
    split = row["split"]
    if split == "train":
        train_labels.append(row["label_id"])
        train_images.append(im)
    elif split == "val":
        val_labels.append(row["label_id"])
        val_images.append(im)
    elif split == "test":
        test_labels.append(row["label_id"])
        test_images.append(im)

X_train = np.array(train_images)
y_train = np.array(train_labels)

X_val = np.array(val_images)
y_val = np.array(val_labels)

X_test = np.array(test_images)
y_test = np.array(test_labels)

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)

print(X_test.shape)
print(y_test.shape)

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)
y_test = y_test.astype(np.int64)

np.savez_compressed(
    "chest_xray_processed.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)