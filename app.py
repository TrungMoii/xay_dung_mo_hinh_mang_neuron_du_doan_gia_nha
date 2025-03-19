from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Tải mô hình và scaler
model = tf.keras.models.load_model("models/model.keras")
scaler = joblib.load("models/scaler.keras")
y_scaler = joblib.load("models/y_scaler.keras")

def predict_price_range(input_data, model, scaler, y_scaler, num_samples=10, noise_std=0.01):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Chuẩn hóa đầu vào

    predictions = []
    for _ in range(num_samples):
        noisy_input = input_data + np.random.normal(0, noise_std, input_data.shape)
        price = model.predict(noisy_input)[0][0]
        predictions.append(price)

    # Chuyển đổi giá trị dự đoán về giá trị gốc
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    return round(min(predictions), 2), round(max(predictions), 2)

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
            price_min, price_max = predict_price_range(input_data, model, scaler, y_scaler)

            # Định dạng giá trị USD
            price_min_usd = "{:,.2f}".format(price_min).replace(",", ".")
            price_max_usd = "{:,.2f}".format(price_max).replace(",", ".")

            return render_template("index.html", price_min=price_min_usd, price_max=price_max_usd)

        except ValueError:
            return render_template("index.html", error="Vui lòng nhập đúng số!")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)