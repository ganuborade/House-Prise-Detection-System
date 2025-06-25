from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and features
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")
usd_to_inr = 82

# Load dataset once for average price
df = pd.read_csv("Data/train.csv")
average_price = df['SalePrice'].mean()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    data["TotalArea"] = data["LotArea"] + data["1stFlrSF"] + data["2ndFlrSF"] + data["GarageArea"]

    input_df = pd.DataFrame([data])[features]
    predicted_price = model.predict(input_df)[0]
    price_inr = round(predicted_price * usd_to_inr, 2)
    avg_inr = round(average_price * usd_to_inr, 2)

    # Generate plot
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Average Price', 'Predicted Price'], [avg_inr, price_inr], color=['gray', 'green'])
    ax.set_ylabel("Price (INR)")
    ax.set_title("Predicted vs Average Price")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"₹{int(height):,}", ha='center', va='bottom')

    # Convert plot to image string
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({"price_inr": price_inr, "graph": graph_url})

if __name__ == "__main__":
    app.run(debug=True)
