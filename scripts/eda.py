import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/house_data.csv")
df.drop(columns=["id"], inplace=True)
# Hiển thị thông tin tổng quan
print(df.info())
print(df.describe())
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Phân bố giá nhà")
plt.xlabel("Giá nhà")
plt.ylabel("Số lượng")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các biến")
plt.show()

