import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Đọc dữ liệu từ CSV
X_train = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_data.csv").values
X_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_data.csv").values
y_train = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_labels.csv").values.ravel()
y_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_labels.csv").values.ravel()

# Xây dựng mô hình mạng nơ-ron
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Dự đoán giá nhà
])

# Biên dịch mô hình
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Sử dụng EarlyStopping để tránh overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Huấn luyện mô hình
model.fit(
    X_train, y_train,
    epochs=100, batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Đánh giá mô hình trên tập test
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Lưu mô hình đã huấn luyện
model.save("E:/TNTT/BTL_DuDoanGiaNha/models/model.keras")

print("Mo hinh da duoc huan luyen va luu thanh cong!")
