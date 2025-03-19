import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
# -*- coding: utf-8 -*-
df = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/house_data.csv")
df.drop(columns=["id"], inplace=True)
X = df.drop(columns=["price"])
y = df["price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(X_scaled, "E:/TNTT/BTL_DuDoanGiaNha/data/processed_data.keras")
joblib.dump(y, "E:/TNTT/BTL_DuDoanGiaNha/data/labels.keras")
joblib.dump(scaler, "E:/TNTT/BTL_DuDoanGiaNha/models/scaler.keras")

print("Du lieu da duoc xu ly va duoc luu!")

