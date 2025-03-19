import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Đọc dữ liệu test
X_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_data.csv").values
y_test = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_labels.csv").values.ravel()

# Load mô hình đã huấn luyện
model = load_model("E:/TNTT/BTL_DuDoanGiaNha/models/model.keras")

# Load scaler đã fit trên tập train
scaler = joblib.load("E:/TNTT/BTL_DuDoanGiaNha/models/scaler.pkl")
y_scaler = joblib.load("E:/TNTT/BTL_DuDoanGiaNha/models/y_scaler.keras")

# Chuẩn hóa X_test
X_test_scaled = scaler.transform(X_test)

# Dự đoán giá nhà
y_pred_scaled = model.predict(X_test_scaled).flatten()

# Chuyển đổi nhãn và dự đoán về giá trị gốc
y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Tính toán sai số
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)

print(f"MAE (gia tri goc): {mae:.2f}")
print(f"MSE (gia tri goc): {mse:.2f}")

# Trực quan hóa kết quả
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.6)
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--')
plt.xlabel("Giá trị thực")
plt.ylabel("Giá trị dự đoán")
plt.title("So sánh giá trị thực và giá trị dự đoán")
plt.show()