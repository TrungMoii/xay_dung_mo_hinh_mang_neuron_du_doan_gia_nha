import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu từ CSV
X_train = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_data.csv").values
X_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_data.csv").values
y_train = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_labels.csv").values.ravel()
y_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_labels.csv").values.ravel()
scaler = StandardScaler()

# Tải y_scaler
y_scaler = joblib.load("E:/TNTT/BTL_DuDoanGiaNha/models/y_scaler.keras")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Xây dựng mô hình mạng nơ-ron
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],),
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Thêm Dropout
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Thêm Dropout
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1)  # Dự đoán giá nhà
])

# Biên dịch mô hình
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Sử dụng EarlyStopping để tránh overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train_scaled,  # Sử dụng dữ liệu đã chuẩn hóa
    epochs=50, batch_size=16,
    validation_data=(X_test_scaled, y_test_scaled),
    callbacks=[early_stopping]
)

test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

model.save("E:/TNTT/BTL_DuDoanGiaNha/models/model.keras")
joblib.dump(scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/scaler.pkl")
joblib.dump(y_scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/y_scaler.keras")
print("Luu!")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE During Training')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()