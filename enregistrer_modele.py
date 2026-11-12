import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# Se connecter au serveur MLflow du conteneur (port publié : 5000)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")

# Lire et préparer les données
df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]

# Entraîner le modèle
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X, y)

# Enregistrer le modèle dans le Registry du serveur
with mlflow.start_run():
    mlflow.sklearn.log_model(
        modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )

# Poser l'alias champion sur la version créée
client = MlflowClient()
version = client.search_model_versions("name='churn-model'")[0].version
client.set_registered_model_alias("churn-model", "champion", version)
print(f"Version {version} de churn-model enregistrée avec l'alias champion")