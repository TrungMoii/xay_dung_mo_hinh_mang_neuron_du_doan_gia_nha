import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Đọc dữ liệu gốc
file_path = "E:/TNTT/BTL_DuDoanGiaNha/data/house_data.csv"
df = pd.read_csv(file_path)

# Loại bỏ cột ID nếu có
df.drop(columns=["id"], inplace=True)

# Tách dữ liệu đầu vào (features) và nhãn (labels)
X = df.drop(columns=["price"])
y = df["price"]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Lưu dữ liệu train/test dưới dạng CSV
pd.DataFrame(X_train, columns=X.columns).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_data.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_data.csv", index=False)
y_train.to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_labels.csv", index=False)
y_test.to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_labels.csv", index=False)

# Lưu scaler và dữ liệu chuẩn hóa
joblib.dump(scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/scaler.keras")

print("Du lieu da duoc xu ly va luu thanh cong!")
