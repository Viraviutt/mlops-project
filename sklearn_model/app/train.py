import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn_model.app.pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn_model.app.config import MLFLOW_TRACKING_URI, MODEL_NAME


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_and_log():
    df = pd.read_csv("./data/winequality-white.csv", sep=";")
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe = build_pipeline()
    with mlflow.start_run(run_name="whitewine_model_training"):
        mlflow.log_param("model", "RandomForest")
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(pipe, MODEL_NAME, registered_model_name=MODEL_NAME)
        print(f"Trained. accuracy={acc}")

if __name__ == "__main__":
    train_and_log()