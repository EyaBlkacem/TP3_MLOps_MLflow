from typing import Literal            # limiter les valeurs possibles d'un champ
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI           # le cadre de travail de l'API
from pydantic import BaseModel        # décrire et valider les données reçues

app = FastAPI(title="API Churn")      # le titre s'affiche dans /docs

# Charger le champion UNE SEULE FOIS, au démarrage
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# Données attendues (validées automatiquement par Pydantic)
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# Route /health : l'API est-elle en vie ?
@app.get("/health")
def health():
    return {"status": "ok"}

# Route /predict : recevoir un client, renvoyer la prédiction
@app.post("/predict")
def predict(client: Client):
    # a) transformer le client en tableau d'une ligne
    ligne = pd.DataFrame([client.model_dump()])
    # b) transformer 'abonnement' en colonnes 0/1, comme à l'entraînement
    ligne = pd.get_dummies(ligne)
    # c) garder les colonnes du modèle, dans le bon ordre (absentes = 0)
    ligne = ligne.reindex(columns=model.feature_names_in_, fill_value=0)
    # d) prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])
    return {"churn": resultat,
            "message": "Le client va partir" if resultat == 1 else "Le client reste"}