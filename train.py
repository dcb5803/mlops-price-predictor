import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/price_data.csv")
X = df[["feature1", "feature2"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)
mse = mean_squared_error(y, preds)

# Log with MLflow
mlflow.set_experiment("price_prediction")
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print(f"Logged model with MSE: {mse}")
