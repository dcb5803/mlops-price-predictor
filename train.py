import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# ✅ Embedded dataset
data = {
    "feature1": [100, 150, 200, 250, 300],
    "feature2": [200, 180, 160, 140, 120],
    "price":    [300, 330, 360, 390, 420]
}
df = pd.DataFrame(data)

# ✅ MLflow setup
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("price_prediction")

# ✅ Model training
X = df[["feature1", "feature2"]]
y = df["price"]
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)
mse = mean_squared_error(y, preds)

# ✅ MLflow logging
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"✅ Model trained and logged with MSE: {mse:.2f}")
