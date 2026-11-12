import os                                # lire les variables d'environnement
from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API Churn")

# Configuration lue dans les variables d'environnement, avec les valeurs
# par défaut du TP4 : sur ton poste, rien ne change.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")

# Charger le modèle UNE SEULE FOIS, au démarrage
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
print(f"Modèle chargé depuis : {MODEL_URI}")

class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(client: Client):
    ligne = pd.DataFrame([client.model_dump()])
    ligne = pd.get_dummies(ligne)
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)
    resultat = int(model.predict(ligne)[0])
    return {"churn": resultat,
            "message": "Le client va partir" if resultat == 1 else "Le client reste"}