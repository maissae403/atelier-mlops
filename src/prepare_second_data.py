
from sklearn.datasets import load_wine
import pandas as pd
import os

# Charger le dataset Wine
wine = load_wine()

# Transformer en DataFrame
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Ajouter la colonne cible
df["target"] = wine.target

# Créer le dossier si nécessaire
os.makedirs("data/raw", exist_ok=True)

# Enregistrer les données
df.to_csv("data/raw/wine.csv", index=False)

print("Dataset Wine généré avec succès")
print("Nombre de lignes :", len(df))
