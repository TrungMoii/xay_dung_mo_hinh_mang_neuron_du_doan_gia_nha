import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("E:/TNTT/BTL_DuDoanGiaNha/data/house_data.csv")
df.drop(columns=["id"], inplace=True)  # Bỏ cột ID vì không cần thiết

# Tạo khung vẽ 3x2 (3 hàng, 2 cột)
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Biểu đồ 1: Phân bố giá nhà
sns.histplot(df["price"], bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Phân bố giá nhà")

# Biểu đồ 2: Ma trận tương quan giữa các biến
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0, 1])
axes[0, 1].set_title("Ma trận tương quan giữa các biến")

# Biểu đồ 3: Số lượng nhà bán theo năm (Bar Plot)
sns.countplot(x=df["year_sold"], ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Số lượng nhà bán theo năm")
axes[1, 0].set_xlabel("Năm bán")
axes[1, 0].set_ylabel("Số lượng nhà")

# Biểu đồ 4: Giá nhà theo khoảng cách đến trung tâm (Scatter Plot)
sns.scatterplot(x=df["distance"], y=df["price"], ax=axes[1, 1])
axes[1, 1].set_title("Giá nhà theo khoảng cách đến trung tâm")
axes[1, 1].set_xlabel("Khoảng cách đến trung tâm (km)")
axes[1, 1].set_ylabel("Giá nhà")

# Biểu đồ 5: Giá nhà theo số lượng cửa hàng gần đó (Scatter Plot)
sns.scatterplot(x=df["num_shops"], y=df["price"], ax=axes[2, 0])
axes[2, 0].set_title("Giá nhà theo số lượng cửa hàng gần đó")
axes[2, 0].set_xlabel("Số lượng cửa hàng gần nhà")
axes[2, 0].set_ylabel("Giá nhà")

# Biểu đồ 6: Bản đồ vị trí nhà (Scatter Plot theo kinh độ và vĩ độ)
sns.scatterplot(x=df["longitude"], y=df["latitude"], hue=df["price"], palette="coolwarm", ax=axes[2, 1])
axes[2, 1].set_title("Bản đồ vị trí nhà")
axes[2, 1].set_xlabel("Kinh độ")
axes[2, 1].set_ylabel("Vĩ độ")

# Căn chỉnh lại bố cục
plt.tight_layout()
plt.show()
