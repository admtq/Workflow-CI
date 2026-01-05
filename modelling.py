from pathlib import Path
import joblib
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "amazon_preprocessing"

X_train = joblib.load(DATA_DIR / "X_train.pkl")
X_test  = joblib.load(DATA_DIR / "X_test.pkl")
y_train = joblib.load(DATA_DIR / "y_train.pkl")
y_test  = joblib.load(DATA_DIR / "y_test.pkl")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI-Amazon-Rating")

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="amazon-rating-model",
        input_example=X_train[:5]
    )

print("✅ Training & MLflow logging completed successfully")