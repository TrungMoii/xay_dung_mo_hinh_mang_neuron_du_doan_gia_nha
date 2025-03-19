import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Đọc dữ liệu kiểm tra
X_test = pd.read_csv("data/test_data.csv").values
y_test = pd.read_csv("data/test_labels.csv").squeeze().values  # Đọc nhãn dạng mảng 1D

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("models/model.keras")

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá hiệu suất
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
