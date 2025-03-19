function predictPrice() {
    let yearBuilt = document.getElementById("year_built").value;
    let houseAge = document.getElementById("house_age").value;
    let distance = document.getElementById("distance").value;
    let shops = document.getElementById("shops").value;
    let latitude = document.getElementById("latitude").value;
    let longitude = document.getElementById("longitude").value;
    let actualPrice = document.getElementById("actual_price").value;
    let predictionText = document.getElementById("prediction");

    if (!yearBuilt || !houseAge || !distance || !shops || !latitude || !longitude || !actualPrice) {
        predictionText.innerHTML = "⚠️ Vui lòng nhập đầy đủ thông tin!";
        predictionText.style.color = "red";
        return;
    }

    let basePrice = 500_000_000;
    let estimatedPrice = basePrice + (2025 - yearBuilt) * 300_000_000 + houseAge * 200_000_000 + distance * (-30_000_000) + shops * 100_000_000;

    estimatedPrice = Math.min(estimatedPrice, 20_000_000_000);
    estimatedPrice = Math.max(estimatedPrice, 100_000_000);

    predictionText.innerHTML = `Giá nhà dự đoán: <b>${estimatedPrice.toLocaleString()} VNĐ</b>`;
    predictionText.style.color = "green";
}
