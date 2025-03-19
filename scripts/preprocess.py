import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Đọc dữ liệu gốc
file_path = "E:/TNTT/BTL_DuDoanGiaNha/data/house_data.csv"
df = pd.read_csv(file_path)

# Loại bỏ cột ID nếu có
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Tách dữ liệu đầu vào (features) và nhãn (labels)
X = df.drop(columns=["price"])
y = df["price"]

# Chia tập train/test trước khi chuẩn hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu, chỉ fit trên tập train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
joblib.dump(y_scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/y_scaler.keras")
# Lưu dữ liệu train/test dưới dạng CSV
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_data.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_data.csv", index=False)
pd.DataFrame(y_train_scaled, columns=["price"]).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/train_labels.csv", index=False)
pd.DataFrame(y_test_scaled, columns=["price"]).to_csv("E:/TNTT/BTL_DuDoanGiaNha/data/test_labels.csv", index=False)

# Lưu scaler đã fit
joblib.dump(scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/scaler.pkl")

print("Da luu!")
