from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load mô hình và bộ chuẩn hóa
model = tf.keras.models.load_model("models/model.keras")
scaler = joblib.load("models/scaler.keras")

# Hàm dự đoán khoảng giá (TÁCH RIÊNG)
def predict_price_range(input_data, model, scaler, num_samples=10, noise_std=0.01):
    """
    Dự đoán giá nhà nhiều lần với nhiễu nhỏ để tính khoảng giá.
    """
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Chuẩn hóa đầu vào

    predictions = []
    for _ in range(num_samples):  # Dự đoán nhiều lần với dữ liệu có nhiễu nhẹ
        noisy_input = input_data + np.random.normal(0, noise_std, input_data.shape)
        price = model.predict(noisy_input)[0][0]
        predictions.append(price)

    return round(min(predictions), 2), round(max(predictions), 2)

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            year_sold = int(request.form["year_sold"])
            house_age = int(request.form["house_age"])
            distance = int(request.form["distance"])
            num_shops = int(request.form["num_shops"])
            longitude = float(request.form["longitude"])
            latitude = float(request.form["latitude"])

            # Chuẩn hóa dữ liệu
            input_data = [[year_sold, house_age, distance, num_shops, longitude, latitude]]

            # Dự đoán khoảng giá
            price_min, price_max = predict_price_range(input_data, model, scaler)

            # Định dạng giá tiền x.xxx.xxx VNĐ
            price_min *= 1000
            price_max *= 1000
            

            price_min = "{:,.0f}".format(price_min).replace(",", ".")
            price_max = "{:,.0f}".format(price_max).replace(",", ".")

            return render_template("index.html", price_min=price_min, price_max=price_max)

        except ValueError:
            return render_template("index.html", error="Vui lòng nhập đúng số!")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
