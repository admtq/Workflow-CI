import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


X_train = joblib.load("amazon_preprocessing/X_train.pkl")
X_test  = joblib.load("amazon_preprocessing/X_test.pkl")
y_train = joblib.load("amazon_preprocessing/y_train.pkl")
y_test  = joblib.load("amazon_preprocessing/y_test.pkl")

mlflow.set_experiment("CI-Amazon-Rating")


with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")