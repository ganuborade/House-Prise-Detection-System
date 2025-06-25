import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and features
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

# Load dataset to calculate average price
df = pd.read_csv("Data/train.csv")
average_price = df['SalePrice'].mean()
usd_to_inr = 82

# Minimal user input
print("📝 Please enter house details below:\n")
inputs = {}
inputs['LotArea'] = float(input("Lot Area (sq ft): "))
inputs['1stFlrSF'] = float(input("1st Floor Area (sq ft): "))
inputs['2ndFlrSF'] = float(input("2nd Floor Area (sq ft): "))
inputs['GarageArea'] = float(input("Garage Area (sq ft): "))
inputs['YearBuilt'] = float(input("Year Built (e.g., 2005): "))
inputs['TotRmsAbvGrd'] = float(input("Total Rooms Above Ground: "))
inputs['GarageCars'] = float(input("Number of Garage Cars: "))
inputs['OverallQual'] = float(input("Overall Quality (1-10): "))
inputs['OverallCond'] = float(input("Overall Condition (1-10): "))

# Derived Feature
inputs['TotalArea'] = inputs['LotArea'] + inputs['1stFlrSF'] + inputs['2ndFlrSF'] + inputs['GarageArea']

# Prepare input DataFrame
input_df = pd.DataFrame([inputs])[features]

# Predict
predicted_price = model.predict(input_df)[0]
price_inr = predicted_price * usd_to_inr

print(f"\n🏠 Predicted House Price: ₹{price_inr:,.2f} INR")

# 🧾 Show Graph: Predicted vs Average
plt.figure(figsize=(6, 4))
bars = plt.bar(['Average Price', 'Predicted Price'], [average_price * usd_to_inr, price_inr], color=['gray', 'green'])
plt.title("Predicted vs Average House Price")
plt.ylabel("Price (INR)")
plt.tight_layout()

# Show values on bars
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, y + 100000, f"₹{int(y):,}", ha='center', va='bottom')

plt.show()
