
import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Charger le dataset Iris
df = pd.read_csv("data/raw/iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 2. Séparer train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Créer le modèle KNN
k = 5

model = KNeighborsClassifier(
    n_neighbors=k
)

# 4. Configurer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

with mlflow.start_run(run_name="KNN_Iris"):

    # 5. Entraîner
    model.fit(X_train, y_train)

    # 6. Faire les prédictions
    predictions = model.predict(X_test)

    # 7. Calculer les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),

        "precision": precision_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),

        "recall": recall_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test, predictions,
            average="weighted",
            zero_division=0
        )
    }

    # 8. Enregistrer dans MLflow
    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("n_neighbors", k)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metrics(metrics)

    # 9. Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/knn_model.pkl"
    )

    mlflow.log_artifact(
        "models/knn_model.pkl",
        artifact_path="model"
    )

    # 10. Sauvegarder les métriques
    os.makedirs("reports", exist_ok=True)

    with open("reports/knn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Résultats KNN :", metrics)
    print("Modèle KNN entraîné et enregistré.")
