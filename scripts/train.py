import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# Load dữ liệu đã xử lý
X = joblib.load("E:/TNTT/BTL_DuDoanGiaNha/data/processed_data.keras")
y = joblib.load("E:/TNTT/BTL_DuDoanGiaNha/data/labels.keras")

# Xây dựng mô hình mạng nơ-ron
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Dự đoán giá nhà
])

# Biên dịch mô hình
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Huấn luyện mô hình
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

# Lưu mô hình đã huấn luyện
model.save("E:/TNTT/BTL_DuDoanGiaNha/models/model.keras")

print("Mo hinh da duoc huan luyen va duoc luu!")
