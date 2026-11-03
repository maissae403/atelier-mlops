
import os
import yaml
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Charger les hyperparamètres
with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)["train"]

# 2. Charger le dataset
df = pd.read_csv("data/raw/iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 3. Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

# 4. Configurer MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

# 5. Démarrer un run MLflow
with mlflow.start_run(run_name="RandomForest_Iris"):

    # Créer le modèle
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )

    # Entraîner
    model.fit(X_train, y_train)

    # Prédire
    predictions = model.predict(X_test)

    # 6. Calculer les métriques
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

    # 7. Enregistrer les paramètres et métriques
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # 8. Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    mlflow.log_artifact(
        "models/model.pkl",
        artifact_path="model"
    )

    # 9. Enregistrer les métriques en JSON
    os.makedirs("reports", exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Métriques :", metrics)
    print("Modèle entraîné et enregistré.")
