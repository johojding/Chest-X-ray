
import numpy as np
data = np.load("chest_xray_processed.npz")

X_train_loaded = data["X_train"]
y_train_loaded = data["y_train"]

print(X_train_loaded.shape)
print(y_train_loaded.shape)