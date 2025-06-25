import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Data/train.csv")


# Feature engineering: create TotalArea and drop GrLivArea
df['TotalArea'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea'] + df['LotArea']
df = df.drop(columns=['GrLivArea'])

# Select features and target
features = [
    "TotalArea", "YearBuilt", "TotRmsAbvGrd", "GarageCars",
    "1stFlrSF", "2ndFlrSF", "GarageArea", "LotArea",
    "OverallQual", "OverallCond"
]
X = df[features]
y = df['SalePrice']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, "models/model.pkl")
joblib.dump(features, "models/features.pkl")

print("✅ Model and features saved successfully.")
