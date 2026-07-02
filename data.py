import os
from pathlib import Path
from PIL import Image
data_path = "/Users/josephinehojding/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
root = Path(data_path)
train_normal = root / "train" / "NORMAL"
train_pneumonia = root / "train" / "PNEUMONIA"
test = root/ "test"
val = root/"val"

train_normal = list(train_pneumonia.rglob("*jpeg"))
print(len(train_normal))
img = Image.open(train_normal[0])
img.show()