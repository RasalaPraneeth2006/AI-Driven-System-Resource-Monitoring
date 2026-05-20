import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 3 features only
X = np.random.rand(200, 3)
y = X[:, 0]*50 + X[:, 1]*30 + X[:, 2]*20

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_scaled, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/resource_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ New model trained with 3 features")